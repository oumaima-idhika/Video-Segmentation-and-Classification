# Segmentation et Classification de vidéo avec deploiement à l'aide de Streamlit

## Objectif

Ce projet a pour but d'appliquer de la segmentation sémantique et de la classification sur une vidéo par le biais des résaux de neurones convolutionnels et de le mettre en valeur en le déployant sur Streamlit

## Partie Segmentation sémantique

on a utilisé le jeu de données [The Cityscapes Dataset](https://www.cityscapes-dataset.com/)  
(à savoir qu'il faut se connecter sur le site web et attendre l'approbation de la part des proprietaires des données )  
Comme son nom l'indique, le jeu de données Cityscapes comprend des exemples d'images e rues étiquetées (à partir d'une séquence vidéo)  
qui peuvent être utilisées pour la compréhension des scènes urbaines, il comporte 34 classes dont Person , Sky , Sidewalk , Road , Wall ....  
  
on a utilisé comme architecture de segmentation sémantique U-Net.  
elle a une structure simple. C'est une répétition de blocs de construction de base : convolutions, ReLu, max pooling pour  
downsampling/encoding path , convolutions, ReLu pour upsampling/decoding path. Il ne devrait donc pas être trop compliqué à mettre en œuvre.  
  
![UNet architecture](https://www.researchgate.net/publication/334287825/figure/fig2/AS:778191392210944@1562546694325/The-architecture-of-Unet.ppm)  
  
### Exemples  
  
![resultatdl](https://user-images.githubusercontent.com/74614342/171282645-57c69417-9535-4e19-a268-63dea53be2a4.PNG)
  
## Partie Classification  
  
La classification des vidéos consiste à attribuer une « action » a tout frame. Dans ce projet, on s’est intéressé aux activités sportives. 
On a commencé par télécharger le jeu de données.  
  
L'ensemble de données que nous avons utilisé est  UCF50 - Action Recognition .  
UCF50  est un ensemble de données de reconnaissance d'action qui contient :  
•	50  catégories d'action composées de vidéos YouTube réalistes  
•	25  groupes de vidéos par catégorie d'action  
•	133  vidéos en moyenne par catégorie d'action  
•	199  Nombre moyen d'images par vidéo  
•	320  largeur moyenne d'images par vidéo  
•	 Hauteur moyenne de 240 images par vidéo  
•	26  images moyennes par seconde par vidéo  
  
Apres, on a créé le modèle CNN, en effet, on a créé  un modèle de classification CNN simple avec deux couches CNN. Le modèle arrive  plus au moins a bien classifier les vidéos avec une précision de 97%.  
  
![resutatdl1 PNG](https://user-images.githubusercontent.com/74614342/171068780-f451dd88-5912-413f-b39e-516aeb41f359.jpeg)
  
### Exemples  
![resutatdl12](https://user-images.githubusercontent.com/74614342/171068775-b5bd6558-1dd9-446b-b2e3-fff17aa08dcc.jpeg)
  
## Partie Déploiement  

## Usage   

installez les bibliothèques suivantes :  
•	Streamlit  
•	Tensorflow  
•	Opencv  
  
to run the app  
```bash
streamlit run ./App_Streamlit.py
```







