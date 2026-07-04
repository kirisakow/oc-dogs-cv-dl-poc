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

  - Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition ([arXiv:2510.09653·Ranjan Sapkota et al.][b])

  - YOLO26: A Comprehensive Architecture Overview and Key Improvements ([arXiv:2602.14582·Priyanto Hidayatullah et al.][c])

[a]: https://arxiv.org/abs/2509.25164
[b]: https://arxiv.org/abs/2510.09653
[c]: https://arxiv.org/abs/2602.14582
-->

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
