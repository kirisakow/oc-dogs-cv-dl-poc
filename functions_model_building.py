from keras import layers
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.efficientnet_v2 import EfficientNetV2B0
from keras.applications.vgg16 import VGG16
from keras.src.callbacks.history import History
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from typing import Callable, Union
import keras.models
import keras.utils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
import seaborn as sns
import tempfile

WORDNET_ID_REGEX_PTRN = re.compile(r'^n\d+-')


def temp_dir_with_symlinks(train_paths, val_paths, test_paths,
                           ) -> str:
    subsets = {'train': train_paths, 'val': val_paths, 'test': test_paths}
    temp_dir = Path(tempfile.mkdtemp(prefix='ultralytics_tempdir_'))
    for subset_name, subset_items in subsets.items():
        for path in subset_items:
            class_label = WORDNET_ID_REGEX_PTRN.split(str(path.parent.name))[1]
            dest = temp_dir / subset_name / class_label / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(path.resolve())
    return str(temp_dir)


def build_model_from_pretrained(*,
                                pretrained_model: Union[EfficientNetB0, EfficientNetV2B0, VGG16],
                                n_classes: int,
                                target_img_size: tuple[int],
                                dropout_rate: float = None,
                                layers_to_finetune: Union[str, int] = None,
                                experiment_name: str = 'CNN_model_from_pretrained',
                                ) -> keras.models.Model:
    inputs = layers.Input(shape=(*target_img_size, 3))
    model = pretrained_model(include_top=False, input_tensor=inputs, weights="imagenet")
    # Freeze the pretrained weights
    model.trainable = False
    # Finetune a targeted block of layers, if any
    if layers_to_finetune:
        if not isinstance(layers_to_finetune, (str, int)):
            raise ValueError("the layers_to_finetune parameter must be of type str or int")
        elif isinstance(layers_to_finetune, str):
            for layer in model.layers:
                if layers_to_finetune in layer.name and not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
        elif isinstance(layers_to_finetune, int):
            n_layers_to_finetune = min(layers_to_finetune, len(model.layers))
            for layer in model.layers[-n_layers_to_finetune:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
    # Rebuild top
    if 'VGG' in pretrained_model.__name__:
        # Use Flatten() for VGG* models
        x = layers.Flatten()(model.output)
    else:
        # Use GlobalAveragePooling2D() for EfficientNet* models
        x = layers.GlobalAveragePooling2D()(model.output)
    x = layers.BatchNormalization()(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="pred")(x)
    return keras.Model(inputs, outputs, name=experiment_name)


def build_model_from_scratch(*,
                             n_classes: int,
                             target_img_size: tuple[int],
                             data_augm: keras.models.Sequential = None,
                             dropout_rate: float = None,
                             filters: list[int] = [32, 64],
                             kernel_size: int = 3,
                             experiment_name: str = 'CNN_model',
                             ) -> keras.models.Model:
    inputs = keras.Input(shape=(*target_img_size, 3))

    x = inputs
    if data_augm:
        x = data_augm(inputs)

    x = layers.Rescaling(1. / 255)(x)

    for n_filters in filters:
        x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name=experiment_name)


def plot_accuracy_and_loss_values(history: History,
                                  *,
                                  suptitle: str,
                                  legend_location: dict,
                                  ) -> None:
    plt.figure(figsize=(12, 4))
    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    for i, metric_name in enumerate(['accuracy', 'loss'], start=1):
        plt.subplot(1, 2, i)
        plt.plot(history.history[f'{metric_name}'])
        plt.plot(history.history[f'val_{metric_name}'])
        plt.title(f'Model {metric_name}')
        plt.ylabel(f'{metric_name.capitalize()}')
        plt.xlabel(None)
        plt.xlim(1, len(history.history[f'{metric_name}']))
        plt.legend(['Train', 'Validation'], loc=legend_location[metric_name])
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_pred, y_test, class_labels,
                          title: str = None,
                          ) -> None:
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    cm = confusion_matrix(y_true, y_pred_cls)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='top')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_test, y_pred, class_labels,
                                ) -> None:
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    y_pred_cls = np.argmax(y_pred, axis=1)
    print(
        classification_report(y_true, y_pred_cls, target_names=class_labels)
    )


class MyKerasSequence(keras.utils.Sequence):
    """Classe personnalisée, dérivée de keras.utils.Sequence, servant à
    charger les images (X) et leurs libellés (y), avec quelques options.

    Args:
        paths (tuple[pathlib.Path]): x_set as paths to images
        labels (tuple[str]): y_set as class labels
        batch_size (int): batch size
        target_size (tuple[int, int], optional): Defaults to (224, 224).
        preprocessing_func (Callable, optional): One of the matching Keras
            native preprocessing functions from `keras.applications.*.preprocess_input()`.
            Defaults to None.
    """
    def __init__(self,
                 paths: tuple[Path],
                 labels: tuple[str],
                 batch_size: int,
                 target_size: tuple[int, int] = (224, 224),
                 preprocessing_func: Callable = None,
                 ):
        self.xset = paths
        self.yset = LabelEncoder().fit_transform(labels)
        self.batch_size = batch_size
        self.target_size = target_size
        self.preprocessing_func = preprocessing_func

    def __len__(self):
        return int(np.ceil(len(self.xset) / self.batch_size))

    def __getitem__(self, idx):
        idx_from = self.batch_size * idx
        idx_to = self.batch_size * (idx + 1)
        batch_images = []
        for path in self.xset[idx_from:idx_to]:
            img = Image.open(path).convert('RGB')
            img = img.resize(self.target_size, Image.Resampling.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            batch_images.append(img_np)
        batch_images = np.array(batch_images)
        batch_encoded_labels = np.array(self.yset[idx_from:idx_to])
        if self.preprocessing_func:
            # Fix bad perf in transfer learning by implementing one of the Keras native preprocessing functions
            batch_images = self.preprocessing_func(batch_images * 255.0)
        return batch_images, batch_encoded_labels


@keras.saving.register_keras_serializable()
class MyF1Metric(keras.metrics.Metric):
    def __init__(self, name='f1_macro', average='macro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.y_true = []
        self.y_pred = []
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_true.extend(y_true.detach().cpu().numpy())
        self.y_pred.extend(np.argmax(y_pred.detach().cpu().numpy(), axis=1))

    def result(self):
        return f1_score(self.y_true, self.y_pred, average=self.average)

    def reset_state(self):
        self.y_true = []
        self.y_pred = []
