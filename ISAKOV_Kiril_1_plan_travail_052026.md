![dataspace_logo.png](./dataspace_logo.png)

# Plan prévisionnel

## Dataset retenu

<!--
Présentez le dataset en quelques lignes.
 -->

C'est le célèbre dataset [ImageNetDogs][1.01] de Stanford qui a servi de matière pour notre étude. Le dataset est composé de 20 580 images annotées et réparties entre 120 classes (races de chiens) à raison de 150 à 250 images par classe.

## Modèle envisagé

<!--
Présentez le ou les arguments qui justifient le choix de l’algorithme, en particulier en quoi il serait susceptible d’apporter de la performance (par exemple, résultat d’études présentées dans un des articles en référence).

Présentez en quelques lignes l’objectif de l’algorithme et le contexte dans lequel il peut être utilisé.
-->

Il s'agit, dans le cadre d'une étude (imaginaire) de migration vers un modèle *deep learning* de classification d'images plus récent, d'entraîner puis de comparer les performances de deux modèles : un vieux (censé être notre baseline) et un récent (de moins de 5 ans) : nous avons opté, respectivement, pour [EfficientNetB0][1.02] de Keras et [YOLO26 spécial classification][1.03] d'Ultralytics.

| 👍 Arguments en faveur de la migration (motivations) | 👎 Arguments contre la migration (freins) |
| --- | --- |
| - Temps d'entraînement extrêmement long. | - Performances toujours très correctes du modèle actuel. Risque de régression. |
| - Temps d'inférence extrêmement long. | - Une *codebase* compatible avec Keras, ce qui n'est pas le cas de YOLO26, tant pour l'entraînement que pour l'inférence. |
| - Taille du modèle sauvegardé | |

## Références bibliographiques

<!--
Présentez deux ou trois références (posts de blog ou articles de recherche) vous permettant de présenter un état de l’art sur le problème étudié et sur lesquels votre travail futur s’appuiera.

Sources conseillées :

- [fastml][1], [machine learning mastery][2], [kdnuggets][3], [import AI][4], [MIT tech review][5], [MIT news ML][6]

- Newsletters de qualité comme [data elixir][7] et [data science weekly][8]

- Twitter, en suivant de grands noms de la discipline

- Articles de recherche : [https://arxiv.org/,][9] [https://scholar.google.fr/][10]...

Il est obligatoire de s’appuyer sur au moins un article de recherche parmi les 2 à 3 sources du projet.

Si lire un article de recherche vous intimide, choisir l’article de recherche illustré dans un article de blog peut vous aider ! Dans ce cas, ils compteront ensemble pour une référence bibliographique. Si vous aimez les vidéos, beaucoup de conférences proposent des tutoriels (NIPS, ICML, ICCV…) qui sont des revues du domaine et peuvent vous aider à identifier des sources pertinentes.
-->

- YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection ([arXiv:2509.25164·Ranjan Sapkota et al.][1.04])

- Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition ([arXiv:2510.09653·Ranjan Sapkota et al.][1.05])

- YOLO26: A Comprehensive Architecture Overview and Key Improvements ([arXiv:2602.14582·Priyanto Hidayatullah et al.][1.06])

## Explication de votre démarche de test du nouvel algorithme (votre preuve de concept)

<!--
Présentez en quelques lignes votre démarche, notamment la méthode baseline pour comparer les performances, et la méthode que vous souhaitez mettre en œuvre.

Dans le contexte de la data science et du machine learning, une preuve de concept (Proof Of Concept or POC) peut être utilisée pour tester si un modèle de machine learning ou une analyse de données est viable et pour évaluer sa performance avec un ensemble de données limité. Son utilité est souvent démontrée via la création d'une interface graphique très simple afin d’interroger le modèle en question.
 -->

### Modéle baseline : EfficientNetB0

- [x] Effectuer le réentraînement d'un modèle préalablement entraîné, dit [le *transfer learning*][1.02] :
  - [x] d'abord réentraîner l'embout, dit *“rebuild top step”*, du classificateur, avec un learning rate régulier,
  - [x] puis réentraîner les couches profondes, dit *“fine tuning step”*, avec un *learning rate* faible ;
  - [x] d'abord sur 3 classes,
  - [x] ensuite sur les 120 classes.
- [x] À logguer :
  - [x] 3 métriques (accuracy, top-5 accuracy, f1_macro),
  - [x] temps d'entraînement,
  - [x] temps d'inférence.

### Modéle SoTA : YOLO26

- [x] Transfer learning (le réentraînement) depuis le modèle YOLO26 en cls de tailles nano, small et medium.
  - idem
- [x] À logguer :
  - idem
- [x] Ajouter interprétabilité ~~avec GradCAM~~ avec [EigenCAM pour YOLO26][1.07].

<!--
[1]: http://fastml.com/
[2]: https://machinelearningmastery.com/
[3]: https://www.kdnuggets.com/
[4]: https://jack-clark.net/
[5]: https://www.technologyreview.com/
[6]: http://news.mit.edu/topic/machine-learning
[7]: https://dataelixir.com/
[8]: https://www.datascienceweekly.org/
[9]: https://arxiv.org/,
[10]: https://scholar.google.fr/
 -->

[1.01]: http://vision.stanford.edu/aditya86/ImageNetDogs
[1.02]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights
[1.03]: https://docs.ultralytics.com/tasks/classify#where-can-i-find-pretrained-yolo26-classification-models
[1.04]: https://arxiv.org/abs/2509.25164
[1.05]: https://arxiv.org/abs/2510.09653
[1.06]: https://arxiv.org/abs/2602.14582
[1.07]: https://github.com/rigvedrs/YOLO-26-CAM
