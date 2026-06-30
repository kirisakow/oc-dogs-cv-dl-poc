# Projet OpenClassrooms « Développez une preuve de concept »

Il s'agit, dans le cadre d'une étude (imaginaire) de migration vers un modèle *deep learning* de classification d'images plus récent, d'entraîner sur un même dataset puis de comparer les performances de deux modèles : un vieux (censé être notre baseline) et un récent (de moins de 5 ans) : nous avons opté, respectivement, pour [EfficientNetB0][1] de Keras et [YOLO26 spécial classification][2] d'Ultralytics.

Enfin, c'est le célèbre dataset [ImageNetDogs][3] de Stanford qui a servi de matière pour notre étude. Le dataset est composé de 20 580 images annotées et réparties entre 120 classes (races de chiens) à raison de 150 à 250 images par classe.

## Notebook 1 ... Notebook 3

## Notebook 4 : Le dashboard, une IHM streamlit pour effectuer l'inférence de la classe d'une image donnée

L'IHM du dashboard permet :

- de sélectionner un model YOLO depuis le répertoire `models/` ;
- de sélectionner et prévisualiser une image depuis le répertoire `images/` ;
- d'effectuer l'inférence et d'en afficher les résultats ;
- de visualiser l'interprétabilité (depuis le répertoire `runs/classify/predict/`) à l'aide de la bibliothèque [EigenCAM pour YOLO26][4].

### Instructions pour lancer l'application streamlit

1. Cloner le repo et instaler les dépendances.

2. Activer le venv :

   ```bash
   source .venv/bin/activate
   ```

3. Lancer l'appli streamlit

   - soit via l'exécutable streamlit :

      ```bash
      streamlit run ISAKOV_Kiril_4_dashboard_052026.py
      ```

   - soit via votre gestionnaire de paquets préféré :

      ```bash
      # si votre gestionnaire de paquets est uv :
      uv run --no-project streamlit run ISAKOV_Kiril_4_dashboard_052026.py

      # si votre gestionnaire de paquets est poetry :
      poetry run streamlit run ISAKOV_Kiril_4_dashboard_052026.py
      ```

4. Attendre que l'appli s'ouvre dans le navigateur et charge complètement.

[1]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights
[2]: https://docs.ultralytics.com/tasks/classify#where-can-i-find-pretrained-yolo26-classification-models
[3]: http://vision.stanford.edu/aditya86/ImageNetDogs
[4]: https://github.com/rigvedrs/YOLO-26-CAM
