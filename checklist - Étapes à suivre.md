# Étapes à suivre

## Notebook 1 : Modéle baseline : EfficientNet

- [ ] Mettre en place une méthode classique :
  - [ ] EfficientNet déjà utilisé, avec les 120 classes,
  - [ ] transfer learning
  - [ ] puis fine tuning,
  - [ ] À logguer :
    - [ ] 3 métriques (accuracy, top5 accuracy, f1_macro),
    - [ ] temps d'entraînement et
    - [ ] temps d'inférence.

## Notebook 2 : Modéle SoTA : YOLO26

- [ ] YOLO26 en cls de tailles nano et small.

## Note méthodologique

- [ ] Faire une synthèse commune des 3 articles pertinents énumérés ci-dessous, qui sera incluse dans les slides et/ou la note méthodo et qui servira à répondre aux questions lors de la soutenance :
  - [ ] YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection
    - [ ] arXiv:2509.25164·Ranjan Sapkota et al.·↑ 55
  - [ ] Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition
    - [ ] arXiv:2510.09653·Ranjan Sapkota et al.·↑ 27
  - [ ] YOLO26: A Comprehensive Architecture Overview and Key Improvements
    - [ ] arXiv:2602.14582·Priyanto Hidayatullah et al.·↑ 1

...

## Présentation

- [ ] Piste d'amélioration à mentionner (sans l'implémenter pour autant) : choisir une fonction de loss adaptée à une situation de la disparité en termes de taille de classes.
