# Projet UE IHM (M1 Informatique)

## Contexte
Dans le cadre de l’UE IHM, le but est de concevoir un système qui intègre des **sous-titres invisibles** dans une vidéo.  
Ils ne doivent pas apparaître quand on regarde directement la vidéo, mais être détectables et affichés via une application mobile.

---

## Fonctionnement attendu
1. **Encodage dans la vidéo**
   - Les sous-titres sont transformés en une série de **points invisibles pour l’œil humain**.
   - Ces points sont placés suivant une **grille logique** (structure virtuelle, non visible).
   - La vidéo finale garde un aspect normal.

2. **Décodage par l’application**
   - L’utilisateur filme la vidéo avec l’application mobile.
   - L’application redresse la vidéo (correction de perspective).
   - Elle détecte les points encodés et lit leur position sur la **grille logique**.
   - Chaque position correspond à un morceau d’information → l’application **reconstruit les sous-titres**.

---

## Outils utilisés
- **OpenCV 2** :  
  - Détection et suivi des points.  
  - Correction de perspective et redressement de la vidéo.  
  - Analyse des patterns encodés pour le décodage des sous-titres.  

---

## Résumé
- **Spectateur normal** : regarde la vidéo → aucun sous-titre visible.  
- **Spectateur via l’app** : filme la vidéo → l’app détecte les points cachés avec **OpenCV 2** → affiche les sous-titres.  

---

## Objectifs pédagogiques
- Comprendre la séparation entre **perception humaine** et **perception machine**.  
- Manipuler des techniques d’**encodage/décodage visuel**.  
- Utiliser **OpenCV 2** pour analyser en temps réel une vidéo et extraire des informations cachées.  
