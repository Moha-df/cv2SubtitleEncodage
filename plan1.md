# Partie 1 – Encodage des sous-titres dans une vidéo

## Objectif
Développer un programme qui prend en entrée une **vidéo** et des **sous-titres** (texte ou fichier `.srt`), et qui génère une nouvelle vidéo où les sous-titres sont encodés sous forme de **points invisibles** placés selon une grille logique.

---

## Étapes détaillées

### 1. Préparation des données
- Récupérer les sous-titres (texte brut ou fichier `.srt`).  
- Segmenter les sous-titres selon le **temps d’apparition** (timing).  
- Convertir les caractères en une **représentation binaire** (chaque bit sera encodé sous forme de point).  

### 2. Définition de la grille logique
- Définir une **grille virtuelle** (ex : 10×10 cases).  
- Chaque case correspond à une position possible pour un point.  
- Décider d’un **mapping binaire → position** :  
  - Exemple : `1` = point placé, `0` = pas de point.  

### 3. Encodage des points
- Pour chaque sous-titre :  
  - Convertir en binaire.  
  - Placer les points correspondants dans la grille virtuelle.  
- Associer chaque grille à la **durée du sous-titre** (affichage synchro avec la vidéo).  

### 4. Insertion dans la vidéo
- Utiliser **OpenCV 2** pour manipuler la vidéo image par image.  
- Superposer les points encodés sur chaque frame correspondante.  
- Les points doivent être :  
  - **Petits et discrets** (ex : variation légère de couleur ou intensité).  
  - Invisibles pour l’œil humain, mais détectables par la caméra et OpenCV.  

### 5. Génération de la vidéo finale
- Exporter la vidéo avec les points encodés.  
- Vérifier que visuellement → rien ne se voit.  
- Stocker aussi un **fichier de référence** (mapping utilisé), pour faciliter le décodage dans la partie 2.  

---

## Outils nécessaires
- **Python**  
- **OpenCV 2** pour :  
  - lecture/écriture de vidéos,  
  - ajout de points sur les frames.  
- Eventuellement **NumPy** pour manipuler la grille et le binaire.  

---

## Résultat attendu
- Une vidéo identique à l’originale pour un spectateur humain.  
- Mais contenant en réalité des **points invisibles** placés de manière logique et encodant les sous-titres.  
