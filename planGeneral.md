# Plan du projet – UE IHM (M1 Informatique)

## 1. Introduction
- Plan du projet en 2 etapes

---

## 2. Partie 1 – Encodage des sous-titres dans une vidéo
### 2.1 Objectif
Créer un programme qui prend une vidéo + des sous-titres en entrée, et produit une vidéo avec des points invisibles encodant ces sous-titres.  

### 2.2 Étapes
1. **Récupération des sous-titres** (texte brut ou fichier `.srt`).  
2. **Transformation en grille logique** :  
   - Chaque caractère/bit du texte correspond à une position dans une grille virtuelle.  
3. **Placement des points invisibles** :  
   - Insertion des points dans la vidéo (imperceptibles pour l’œil humain).  
4. **Export de la vidéo encodée**.  

### 2.3 Outils
- Python  
- OpenCV 2 (gestion des images et placement des points).  

---

## 3. Partie 2 – Décodage sur smartphone
### 3.1 Objectif
Développer une application mobile qui filme la vidéo encodée et en extrait les sous-titres.  

### 3.2 Étapes
1. **Capture vidéo avec la caméra du smartphone**.  
2. **Redressement de l’image** (correction de perspective avec OpenCV).  
3. **Détection des points invisibles**.  
4. **Lecture de la grille logique** :  
   - Décodage des positions → reconstruction du texte.  
5. **Affichage des sous-titres à l’écran en temps réel**.  

### 3.3 Outils
- Application mobile (Android de préférence).  
- OpenCV 2 pour :  
  - correction de perspective,  
  - détection et lecture des points.  

---

## 4. Tests et validation
- Vérifier que les points restent invisibles pour un spectateur humain.  
- Vérifier que le smartphone détecte correctement les sous-titres.  
- Tester en conditions variées (lumière, angle, qualité caméra).  
