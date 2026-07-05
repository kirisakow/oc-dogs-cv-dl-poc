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
