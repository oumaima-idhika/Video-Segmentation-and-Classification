# Segmentation et Classification de vidéo avec deploiement à l'aide de Streamlit
##Objectif
Ce projet a pour but d'appliquer de la segmentation sémantique et de la classification sur une vidéo 
par le biais des résaux de neurones convolutionnels et de le mettre en valeur en le déployant sur Streamlit
## Partie Segmentation sémantique
on a utilisé le jeu de données [The Cityscapes Dataset](https://www.cityscapes-dataset.com/) 
(à savoir qu'il faut se connecter sur le site web et attendre l'approbation de la part des proprietaires des données )
Comme son nom l'indique, le jeu de données Cityscapes comprend des exemples d'images e rues étiquetées (à partir d'une séquence vidéo)
qui peuvent être utilisées pour la compréhension des scènes urbaines, il comporte 34 classes dont Person , Sky , Sidewalk , Road , Wall ....

on a utilisé comme architecture de segmentation sémantique U-Net
elle a une structure simple. C'est une répétition de blocs de construction de base : convolutions, ReLu, max pooling pour
downsampling/encoding path , convolutions, ReLu pour upsampling/decoding path. Il ne devrait donc pas être trop compliqué à mettre en œuvre.

