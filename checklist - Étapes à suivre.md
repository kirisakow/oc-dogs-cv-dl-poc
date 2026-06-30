# Étapes à suivre

## Notebook 1 : Modéle baseline : EfficientNet

- [x] Transfer learning (le réentraînement) depuis le modèle EfficientNet
  - [x] d'abord réentraînement de l'embout,
  - [x] puis le réentraînement plus en profondeur (dit le *fine tuning*), avec un *learning rate* faible ;
  - [x] d'abord sur 3 classes,
  - [x] ensuite sur 120 classes.
- [x] À logguer :
  - [x] 3 métriques (accuracy, top5 accuracy, f1_macro),
  - [x] temps d'entraînement et
  - [x] temps d'inférence.

## Notebook 2 : Modéle SoTA : YOLO26

- [x] Transfer learning (le réentraînement) depuis le modèle YOLO26 en cls de tailles nano, small et medium.
  - idem
- [x] À logguer :
  - idem
- [x] Ajouter interprétabilité ~~avec GradCAM~~ avec [EigenCAM pour YOLO26][3].

## Plan de travail

- [ ] Un plan de travail prévisionnel, respectant [le modèle fourni][1] et transformé en pdf (1 page).

## Note méthodologique

- [ ] Un note méthodologique présentant la preuve de concept respectant [le template fourni][2] et transformé en pdf (10 pages maximum).

- [ ] Faire une synthèse commune des 3 articles pertinents énumérés ci-dessous, qui sera incluse dans les slides et/ou la note méthodo et qui servira à répondre aux questions lors de la soutenance :
  - YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection (arXiv:2509.25164·Ranjan Sapkota et al.)
  - Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition (arXiv:2510.09653·Ranjan Sapkota et al.)
  - YOLO26: A Comprehensive Architecture Overview and Key Improvements (arXiv:2602.14582·Priyanto Hidayatullah et al.)

## Dashboard sous Steamlit

(Instructions : https://course.oc-static.com/projects/MLE_V2_P7/Spécifications+dashboard.pdf)

- [ ] Inclure au préalable une analyse exploratoire des données permettant d’illustrer le contenu du jeu de données, selon le type de données.
  - [ ] En cas de données non structurées de type image, cette analyse se formalise par la présentation d’exemples d’images du dataset, selon les éventuelles catégories ou données structurelles associées, de leur comptage et de leur transformation (par exemple : equalization, floutage).
- [x] Inclure la sélection de données en entrée du moteur de prédiction (par exemple, liste déroulante ou saisie de champs).
- [x] Inclure le résultat de la prédiction.
- [ ] Être déployé dans le cloud.
- [ ] Prendre en compte le besoin des personnes en situation de handicap dans la réalisation des graphiques, en couvrant des critères d'accessibilité du WCAG essentiels.

💡 Afficher seulement les perf du modèle SoTA, pas du modèle *baseline*.

## Présentation

- [ ] Un support de présentation pour la soutenance détaillant le travail réalisé, autant sur le dashboard que la modélisation (Powerpoint ou équivalent, transformé en pdf, 30 slides maximum).
- [ ] Piste d'amélioration à mentionner (sans l'implémenter pour autant) : choisir une fonction de loss adaptée à des classes dont la taille varie fortement.

[1]: https://course.oc-static.com/projects/MLE_V2_P7/Mode%CC%80le_Plan_pre%CC%81visionnel.docx
[2]: https://course.oc-static.com/projects/MLE_V2_P7/Mode%CC%80le_Note_Me%CC%81thodologique.docx
[3]: https://github.com/rigvedrs/YOLO-26-CAM
