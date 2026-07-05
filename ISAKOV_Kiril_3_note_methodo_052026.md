# Note méthodologique : preuve de concept

## Dataset retenu

<!--
*Présentez le dataset en 1 page maximum.*
-->

Notre étude s'appuie sur le célèbre dataset [ImageNetDogs][1] de Stanford, spécialement conçu pour la classification fine de races de chiens et composé de 20 580 images annotées et réparties entre 120 classes à raison, plutôt bien équilibrée, de 150 à 250 images par classe, offrant une base solide pour l'entraînement et l'évaluation de modèles de classification. De plus, ce dataset présente un défi intéressant pour les modèles de vision par ordinateur en raison de la similarité visuelle entre certaines races, nécessitant une bonne capacité de discrimination fine des caractéristiques morphologiques.

Les images sont fournies au format JPEG avec une résolution variable, et sont organisées dans une arborescence de dossiers où chaque sous-dossier porte le nom de la classe correspondante, précédée d'un [WordNet ID indiquant sa catégorie synonymique][2] (ex: `n02088094-Afghan_hound`). Cette structure facilite le chargement et le prétraitement des données.

Pour notre preuve de concept, nous avons commencé par entraîner notre modèle d'abord sur un échantillon réduit de trois classes selon le nombre d'images qui les composent : deux plus nombreuses et une la moins nombreuse. Le développement une fois complet et exempt d'erreurs, nous avons réentraîné un modèle final sur l'intégralité du dataset sans filtration supplémentaire.

Pour ce qui est de la répartition des images pour l'entraînement, la validation et le test, nous avons procédé par une proportion assez classique en appliquant un split stratifié en trois ensembles :

- 80% pour l'entraînement (16 464 images pour l'entraînement du modèle final)
- 10% pour la validation (2 058 images pour l'entraînement du modèle final)
- 10% pour le test (2 058 images pour l'entraînement du modèle final)

## Les concepts de l’algorithme récent

<!--
*Présentez, en 2 pages maximum, les principes de fonctionnement du nouvel algorithme.*

- Faire une synthèse commune des 3 articles pertinents énumérés ci-dessous, qui sera incluse dans les slides et/ou la note méthodo et qui servira à répondre aux questions lors de la soutenance :

  - YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection ([arXiv:2509.25164·Ranjan Sapkota et al.][a])
    - copie locale: ./arxiv.2509.25164v5.pdf

  - Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition ([arXiv:2510.09653·Ranjan Sapkota et al.][b])
    - copie locale: ./arxiv.2510.09653v3.pdf

  - YOLO26: A Comprehensive Architecture Overview and Key Improvements ([arXiv:2602.14582·Priyanto Hidayatullah et al.][c])
    - copie locale: ./arxiv.2602.14582v1.pdf

[a]: https://arxiv.org/abs/2509.25164
[b]: https://arxiv.org/abs/2510.09653
[c]: https://arxiv.org/abs/2602.14582
-->

La famille YOLO (You Only Look Once), introduite en 2016 par Redmon et al., a révolutionné la détection d'objets en proposant une approche unifiée en une seule passe (*one-stage*), réduisant ainsi la complexité des pipelines traditionnels en deux étapes (comme R-CNN ou Faster R-CNN). Cette approche permet d'atteindre des vitesses d'inférence en temps réel tout en maintenant une précision compétitive, ce qui a rendu YOLO particulièrement attractif pour les applications où la latence est critique, telles que la robotique, la navigation autonome ou l'analyse vidéo en direct.

L'évolution de YOLO s'est faite par itérations successives, chacune adressant des limitations spécifiques tout en intégrant des avancées en matière de design de réseaux de neurones, de fonctions de perte et d'efficacité de déploiement. YOLOv2 (2017) a introduit les *anchor boxes* et la normalisation par lots (*batch normalization*), YOLOv3 (2018) a approfondi l'architecture avec Darknet-53 et des cartes de caractéristiques multi-échelles, tandis que YOLOv4 (2020) a intégré CSPDarknet et des stratégies d'augmentation avancées. Ultralytics a ensuite popularisé une implémentation native en PyTorch avec YOLOv5 (2020), rendant le framework plus accessible et modulaire.

YOLO26, sorti en septembre 2025, représente la dernière étape de cette évolution et incarne une philosophie de conception axée sur la simplicité, l'efficacité et l'innovation. Contrairement aux versions précédentes qui ajoutaient de la complexité architecturale, YOLO26 se distingue par des simplifications stratégiques conçues pour optimiser le déploiement sur des appareils *edge* et à faible consommation.

### Innovations architecturales clés

YOLO26 introduit cinq améliorations majeures qui le différencient de ses prédécesseurs :

#### Suppression de la Distribution Focal Loss (DFL)

La DFL, utilisée dans les versions précédentes pour améliorer la régression des *bounding boxes* en prédisant une distribution de positions possibles plutôt qu'une valeur unique, a été supprimée. Cette simplification allège le graphe computationnel, facilite l'export vers différents formats (ONNX, TensorRT, CoreML, TFLite) et accélère l'inférence, particulièrement sur CPU où une amélioration jusqu'à 43% est rapportée.

#### Inférence native sans NMS (*Non-Maximum Suppression*)

Traditionnellement, les modèles de détection d'objets génèrent de multiples prédictions redondantes pour un même objet, nécessitant une étape de post-traitement NMS pour filtrer ces doublons. YOLO26 élimine ce goulot d'étranglement en adoptant une approche *end-to-end* où le réseau produit directement un ensemble compact de prédictions non redondantes. Cette innovation supprime non seulement la latence associée au NMS, mais aussi la nécessité de régler des hyperparamètres spécifiques au déploiement (seuils IoU, seuils de score).

#### ProgLoss et STAL

Pour garantir la stabilité de l'entraînement et améliorer la détection des petits objets, YOLO26 intègre deux mécanismes complémentaires :

- *ProgLoss* (*Progressive Loss Balancing*) : rééquilibre adaptativement les objectifs de perte pour éviter la domination des exemples faciles en fin d'entraînement
- *STAL* (*Small-Target-Aware Label Assignment*) : priorise l'assignation des étiquettes pour les instances minuscules ou occluses, améliorant significativement le rappel dans des conditions difficiles (encombrement, feuillage, flou de mouvement)

#### Optimiseur MuSGD

YOLO26 utilise un nouvel optimiseur hybride, MuSGD, qui combine les avantages de SGD (généralisation robuste) avec des comportements inspirés des méthodes de type Muon (convergence rapide et stable). Cet optimiseur permet une convergence plus rapide et plus fiable, avec un meilleur comportement sur les plateaux d'apprentissage.

### Architecture unifiée multi-tâches

YOLO26 est le premier modèle de la famille à supporter nativement cinq tâches de vision par ordinateur au sein d'une architecture unifiée :

- Détection d'objets (*object detection*)
- Segmentation d'instances (*instance segmentation*)
- Estimation de pose/détection de points clés (*pose/keypoints detection*)
- Détection orientée (*oriented detection*, pour les objets obliques ou allongés)
- Classification (*classification*)

Cette conception consolidée permet un entraînement multi-tâches ou un fine-tuning spécifique à une tâche sans nécessiter de réarchitecture, tout en préservant la portabilité à travers différents accélérateurs matériels. Le *backbone* et le *neck* partagés, couplés à des têtes de prédiction simplifiées, assurent une réutilisation efficace des caractéristiques tout en minimisant la redondance computationnelle.

### Optimisation pour le déploiement *edge*

YOLO26 a été conçu avec une approche *edge-first*, c'est-à-dire optimisé dès le départ pour les contraintes des appareils embarqués et à faible consommation. Les choix architecturaux reflètent cette priorité :

- **Export multi-format** : support natif pour ONNX, TensorRT, CoreML et TFLite
- **Quantification** : support de la précision FP16 et INT8 pour accélérer l'inférence sur CPU, NPU et GPU avec une dégradation minimale de la précision
- **Graphes simplifiés** : la suppression de DFL et du NMS réduit la complexité du graphe computationnel, facilitant la compilation et l'optimisation sur différents *backends*
- **Latence réduite** : les benchmarks montrent que YOLO26 atteint un meilleur compromis précision-latence que les versions précédentes, notamment sur des appareils comme le NVIDIA Jetson Nano et Orin

Ces innovations positionnent YOLO26 comme un modèle de nouvelle génération, particulièrement adapté aux applications en temps réel sur des appareils à ressources limitées, tout en maintenant une précision de pointe pour la détection d'objets.

## La modélisation

<!--
*Présentez la méthodologie de modélisation, la métrique d'évaluation retenue et sa démarche d'optimisation, en 2 pages maximum.*

Présenter pour les deux modèles :

- Le jeu de données,
- Les étapes d'entrainement
- Métriques
-->

### Contexte et objectifs

Notre étude s'inscrit dans le cadre d'une migration hypothétique vers un modèle de classification d'images plus récent. L'objectif est de comparer les performances de deux approches : un modèle *baseline* représentatif des architectures traditionnelles, et un modèle *state-of-the-art* (SoTA) incarnant les dernières avancées en vision par ordinateur. Nous avons sélectionné EfficientNetB0 comme modèle de référence, pour son équilibre éprouvé entre précision et efficacité, et YOLO26 dans sa version classification, pour ses innovations architecturales récentes et son optimisation pour les déploiements *edge*.

Cette comparaison vise à évaluer non seulement la précision brute, mais aussi des critères pratiques tels que le temps d'entraînement, le temps d'inférence, et la facilité de déploiement, afin de déterminer si la migration vers YOLO26 se justifie dans un contexte industriel.

### Méthodologie commune

#### Jeu de données

Les deux modèles ont été entraînés et évalués sur le dataset ImageNetDogs dans sa totalité, soit 20 580 images réparties en 120 classes (races de chiens). Pour valider notre approche de manière itérative, nous avons initialement testé l'entraînement sur un sous-ensemble de 3 classes (deux classes nombreuses et une classe moins représentée) avant de procéder à l'entraînement final sur l'intégralité des 120 classes.

#### Répartition des données

Un *split* stratifié a été appliqué pour conserver la distribution des classes dans chacun des ensembles :

- 80% pour l'entraînement (16 464 images)
- 10% pour la validation (2 058 images)
- 10% pour le test (2 058 images)

Cette stratification est cruciale pour un dataset comme ImageNetDogs où le nombre d'images par classe varie entre 150 et 250, garantissant que chaque classe est représentée de manière proportionnelle dans chaque ensemble.

#### Métriques d'évaluation

Trois métriques principales ont été suivies pour évaluer les performances des modèles :

- **Accuracy** : précision classique, mesurant le pourcentage de prédictions correctes
- **Top-5 Accuracy** : précision élargie aux 5 prédictions les plus probables, pertinente pour un problème à 120 classes où la distinction fine entre races similaires peut être ambiguë
- **F1-score macro** : moyenne non pondérée des F1-scores par classe, offrant une mesure équilibrée entre précision et rappel, particulièrement importante pour évaluer les performances sur les classes minoritaires

#### Autres indicateurs mesurés

- Temps d'entraînement par époque
- Temps total d'entraînement
- Temps d'inférence par image et par lot (*batch*)

### Modèle Baseline : EfficientNetB0

#### Architecture

EfficientNetB0 est une variante de la famille EfficientNet, conçue selon la méthode *Compound Scaling* qui optimise simultanément la profondeur, la largeur et la résolution des images. Ce modèle utilise un *backbone* convolutionnel avec des blocs MBConv (Mobile Inverted Bottleneck Convolutions) et une fonction d'activation Swish. Pour notre tâche de classification sur 120 classes, nous avons adapté le modèle pré-entraîné sur ImageNet en remplaçant la couche de classification finale.

#### Prétraitement des images

Les images ont été redimensionnées à 224×224 pixels et normalisées selon les statistiques ImageNet (moyenne=[0.485, 0.456, 0.406], écart-type=[0.229, 0.224, 0.225]) avant d'être passées au modèle.

#### Étapes d'entraînement

L'entraînement a suivi une approche classique de *transfer learning* en deux étapes :

##### Stage 1 : *Rebuild Top* (20 époques)

- Seule la nouvelle tête de classification (*top layer*) est entraînée
- Le *backbone* reste gelé (*frozen*) pour préserver les caractéristiques générales apprises sur ImageNet
- *Learning rate* régulier : 1e-3
- Taille des *batches* : 4 images
- Taille des images : 224×224 pixels
- Fonction de perte : *sparse categorical crossentropy*
- Optimiseur : Adam
- *Dropout rate* : 0.2 pour réduire le surapprentissage

##### Stage 2 : *Fine-Tuning* (20 époques)

- Dégel des couches profondes du *backbone* (à partir de *block7* pour EfficientNetB0)
- La tête de classification, déjà entraînée, continue d'être affinée
- *Learning rate* réduit : 1e-5 pour éviter de perturber les caractéristiques déjà apprises
- Tous les autres paramètres restent identiques au Stage 1

Cette approche en deux étapes permet d'abord adapter le classificateur aux nouvelles classes, puis d'affiner progressivement les caractéristiques du *backbone* pour qu'elles soient plus spécifiques à notre dataset.

### Modèle SoTA : YOLO26 (version classification)

#### Architecture

YOLO26 en version classification conserve les innovations architecturales de la version détection, mais adaptées pour la tâche de classification. Le modèle repose sur un *backbone* CSPNet optimisé, des blocs C3k2 avec attention spatiale, et une tête de classification simplifiée. Contrairement aux versions précédentes de YOLO, YOLO26 élimine la DFL et adopte une inférence *end-to-end* sans NMS, même pour la classification.

#### Gestion des données

Contrairement à EfficientNetB0 qui utilise des séquences Keras personnalisées, YOLO26 nécessite une structure de répertoires spécifique. Nous avons donc créé un répertoire temporaire avec des liens symboliques organisant les images selon la structure attendue par Ultralytics : un dossier par ensemble (train/val/test) contenant des sous-dossiers par classe.

#### Prétraitement des images

YOLO26 applique ses propres transformations internes, incluant le redimensionnement, la normalisation, et des augmentations optionnelles (mosaïque, mixup) que nous avons désactivées pour une comparaison équitable avec EfficientNetB0.

#### Étapes d'entraînement

L'entraînement de YOLO26 a également suivi une approche de *transfer learning* en deux étapes, avec des paramètres adaptés :

##### Stage 1 : *Freeze Backbone, Train Head and Neck* (20 époques)

- Le *backbone* est entièrement gelé
- Seules la tête (*head*) et le *neck* (couches de fusion de caractéristiques) sont entraînés
- *Learning rate* régulier : 1e-3
- Taille des *batches* : 8 images (supérieure à EfficientNetB0 grâce à l'optimisation mémoire de YOLO26)
- Taille des images : 224×224 pixels
- *Freeze depth* : 10 couches (tout le *backbone* gelé)

##### Stage 2 : *Partial Unfreeze, Fine-Tune* (20 époques)

- Dégel partiel du *backbone* : 10 - 2 = 8 couches gelées, 2 couches dégélées
- La tête et le *neck* continuent d'être affinés
- *Learning rate* réduit : 1e-5
- Tous les autres paramètres restent identiques au Stage 1

### Démarche d'optimisation

Pour garantir une comparaison juste entre les deux modèles, nous avons veillé à :

1. **Uniformité des données** : Même *split* stratifié, mêmes images, mêmes étiquettes
2. **Uniformité des métriques** : Même jeu de métriques (accuracy, top-5 accuracy, F1-score macro)
3. **Uniformité des conditions** : Même taille d'images (224×224), même nombre d'époques (20 par stage)
4. **Reproductibilité** : Mêmes *random seeds* (42) pour les splits et initialisations

L'optimisation s'est concentrée sur :

- Le choix des hyperparamètres (*learning rates*, taille des *batches*) adaptés à chaque architecture
- La stratégie de *fine-tuning* (quelles couches dégeler, dans quel ordre)
- La prévention du surapprentissage (*dropout* pour EfficientNetB0, *regularization* intégrée pour YOLO26)

### Les temps d'entraînement et d'inférence

Les temps d'entraînement et d'inférence ont été mesurés de manière systématique pour évaluer l'efficacité computationnelle de chaque approche. Les résultats obtenus sur l'intégralité du dataset (120 classes, 20 époques par stage) révèlent des différences significatives :

#### EfficientNetB0 (baseline)

- Temps total d'entraînement (2 stages) : 10h09m (5h01m pour le *Rebuild Top* + 5h08m pour le *Fine-Tuning*)
- Temps d'inférence moyen sur un échantillon de 100 images : 4,75s

#### YOLO26 (SoTA) selon la taille du modèle

- **yolo26n-cls** (nano) : entraînement en 53m, inférence sur 100 images en 0,98s
- **yolo26s-cls** (small) : entraînement en 1h03m, inférence sur 100 images en 1,72s
- **yolo26m-cls** (medium) : entraînement en 1h33m, inférence sur 100 images en 1,54s

Ces mesures démontrent un avantage conséquent de YOLO26 en termes d'efficacité : un entraînement jusqu'à 10 fois plus rapide et une inférence 3 à 5 fois plus rapide que le modèle *baseline*, tout en maintenant une précision compétitive. Les tests préliminaires sur 3 classes ont confirmé cette tendance avec des temps proportionnellement réduits.

## Une synthèse des résultats

<!--
*Présentez une synthèse des résultats comparés entre la technique récente et les techniques utilisées précédemment et une conclusion, en 2 pages maximum.*

- Courbes d'apprentissage
- Métriques
-->

## L’analyse de la feature importance globale et locale du nouveau modèle

<!--
*Présentez l’analyse de la feature importance globale et locale du nouveau modèle, en 2 pages maximum.*

- L'interprétabilité : EigenCAM
-->

## Les limites et les améliorations possibles

<!--
*Présentez les limites et les améliorations envisageables pour gagner en performance et en interprétabilité de l'approche de modélisation, en 1 page maximum.*

- Optimiser chacun des 2 modèles pour les comparer dans leur meilleure version -- pas un système deux poids, deux mesures inique !
-->

[1]: http://vision.stanford.edu/aditya86/ImageNetDogs
[2]: https://en.wikipedia.org/wiki/ImageNet#Categories
