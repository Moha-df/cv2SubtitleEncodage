# Projet IHM - Système de Sous-titres Invisibles

## Description
Ce projet implémente un système de sous-titres invisibles pour l'UE IHM en M1 Informatique. Il permet d'encoder des sous-titres dans une vidéo sous forme de points invisibles à l'œil nu, puis de les décoder via une application mobile.

## Structure du Projet

### Fichiers Sources
- `video.mp4` - Vidéo source originale
- `subtitle.srt` - Fichier de sous-titres SRT
- `video_encoded.mp4` - Vidéo avec sous-titres encodés
- `video_encoded_mapping.json` - Fichier de mapping pour le décodage

### Scripts Python
- `encode_subtitles.py` - Script d'encodage des sous-titres
- `decode_subtitles.py` - Script de décodage des sous-titres
- `test_encode.py` - Script de test pour l'encodage
- `test_decode.py` - Script de test pour le décodage

## Installation

1. Créer un environnement virtuel Python :
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Installer les dépendances :
```bash
pip install opencv-python numpy
```

## Utilisation

### Encodage
```bash
python encode_subtitles.py --video video.mp4 --srt subtitle.srt --output video_encoded.mp4
```

### Décodage
```bash
python decode_subtitles.py --video video_encoded.mp4 --mapping video_encoded_mapping.json --output decoded_subtitles.txt
```

### Tests
```bash
# Test d'encodage
python test_encode.py

# Test de décodage
python test_decode.py
```

## Principe de Fonctionnement

### Encodage
1. **Parsing SRT** : Lecture et parsing du fichier de sous-titres
2. **Conversion binaire** : Transformation du texte en représentation binaire
3. **Grille logique** : Placement des bits sur une grille virtuelle (10x10 par défaut)
4. **Points invisibles** : Insertion de points subtils dans la vidéo selon le mapping binaire
5. **Mapping** : Sauvegarde du mapping pour le décodage

### Décodage
1. **Détection** : Identification des points encodés dans chaque frame
2. **Reconstruction** : Conversion des positions en chaîne binaire
3. **Décodage** : Transformation binaire → texte
4. **Affichage** : Présentation des sous-titres décodés

## Paramètres Configurables

- `--grid-width` : Largeur de la grille (défaut: 10)
- `--grid-height` : Hauteur de la grille (défaut: 10)
- `point_size` : Taille des points en pixels (défaut: 2)
- `point_intensity` : Variation d'intensité des points (défaut: 5)

## Résultats

✅ **Encodage réussi** : 319 sous-titres encodés dans la vidéo
✅ **Vidéo générée** : `video_encoded.mp4` (307MB)
✅ **Mapping créé** : `video_encoded_mapping.json` (78MB)

## Prochaines Étapes

1. **Partie 2** : Développement de l'application mobile Android
2. **Amélioration** : Optimisation de la détection des points
3. **Tests** : Validation en conditions réelles

## Technologies Utilisées

- **Python 3**
- **OpenCV 2** : Manipulation vidéo et détection de points
- **NumPy** : Calculs matriciels
- **JSON** : Stockage du mapping de référence
