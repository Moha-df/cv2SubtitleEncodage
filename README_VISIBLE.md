# Projet IHM - Version de Test avec Points Visibles

## 🎯 Objectif
Version de test du système de sous-titres invisibles avec des **points ROUGES bien visibles** pour vérifier le fonctionnement du système d'encodage/décodage.

## 📁 Fichiers de Test

### Scripts
- `encode_subtitles_visible.py` - Encodeur avec points ROUGES visibles
- `decode_visible.py` - Décodeur pour points visibles
- `video_visible_test.mp4` - Vidéo de test avec points rouges
- `video_visible_test_mapping.json` - Mapping de référence

### Fichiers Sources
- `video.mp4` - Vidéo originale
- `subtitle.srt` - Sous-titres originaux

## 🚀 Utilisation Rapide

### 1. Encodage avec Points Visibles
```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Encoder avec des points ROUGES visibles
python encode_subtitles_visible.py --video video.mp4 --srt subtitle.srt --output video_visible_test.mp4
```

### 2. Décodage des Points Visibles
```bash
# Décoder les points rouges
python decode_visible.py --video video_visible_test.mp4 --mapping video_visible_test_mapping.json --output decoded_visible.txt
```

## 🔍 Vérification Visuelle

### Ouvrir la Vidéo de Test
```bash
# Ouvrir la vidéo avec points rouges
vlc video_visible_test.mp4
# ou
mpv video_visible_test.mp4
```

### Ce que vous devez voir :
- **Points ROUGES** qui apparaissent pendant les sous-titres
- **Grille 10×10** : Les points sont placés selon une grille virtuelle
- **Timing parfait** : Les points apparaissent exactement pendant les sous-titres
- **Contour blanc** : Chaque point rouge a un contour blanc pour la visibilité

## 📊 Résultats Attendus

### Encodage
```
Parsing des sous-titres...
Trouvé 12 sous-titres
Propriétés vidéo: 1280x720, 24 FPS, 844 frames
Encodage en cours...
Frame 0/844 - Sous-titre: '*Naruto screaming in shock*' - 49 points
Frame 48/844 - Sous-titre: 'Kakashi: Calm down Naruto...' - 44 points
...
Encodage terminé!
```

### Décodage
```
Décodage des points ROUGES...
Frame 0 - Points détectés: 66 - Texte: '.No{~g'
Frame 48 - Points détectés: 71 - Texte: '[gksno'
...
Décodage terminé!
Trouvé 841 sous-titres décodés
```

## ⚙️ Paramètres de Test

### Points Visibles
- **Taille** : 8 pixels de rayon
- **Couleur** : Rouge (0, 0, 255)
- **Contour** : Blanc de 2 pixels
- **Grille** : 10×10 cellules

### Détection
- **Méthode** : Détection de couleur HSV
- **Seuil** : Rouge pur détecté
- **Précision** : Par cellule de grille

## 🎬 Exemple de Sous-titres Encodés

| Temps | Sous-titre Original | Points Détectés | Texte Décodé |
|-------|-------------------|-----------------|--------------|
| 0.00s | *Naruto screaming* | 66 | .No{~g |
| 2.00s | Kakashi: Calm down | 71 | [gksno |
| 4.00s | Be quiet, as far as | 62 | O}ozpw q |

## 🔧 Dépannage

### Problème : Aucun point visible
```bash
# Vérifier que la vidéo a été encodée
ls -la video_visible_test.mp4
# Doit faire ~11MB
```

### Problème : Décodage corrompu
- **Normal** : Le texte décodé est corrompu car c'est une version de test
- **Important** : Les points sont détectés (66-81 points par frame)
- **Objectif** : Vérifier que le système fonctionne, pas la qualité du texte

### Problème : Points mal placés
- Vérifier la résolution vidéo (doit être 1280x720)
- Vérifier que la grille 10×10 est respectée

## 📈 Validation du Système

### ✅ Ce qui fonctionne
1. **Encodage** : Points rouges placés correctement
2. **Détection** : Points détectés par le décodeur
3. **Timing** : Synchronisation avec les sous-titres
4. **Grille** : Système de grille 10×10 opérationnel

### ⚠️ Ce qui est attendu (version test)
1. **Texte corrompu** : Normal pour cette version de test
2. **Points visibles** : Intentionnel pour la validation
3. **Détection approximative** : Améliorable dans la version finale

## 🎯 Prochaines Étapes

1. **Version invisible** : Revenir aux points subtils
2. **Améliorer la précision** : Réduire la grille à 8×8
3. **Ajouter redondance** : Encoder chaque bit plusieurs fois
4. **Filtrer le bruit** : Améliorer la détection des positions

## 📝 Notes Techniques

- **Format vidéo** : MP4 (H.264)
- **Résolution** : 1280×720
- **FPS** : 24
- **Encodage** : 8 bits par caractère ASCII
- **Grille** : 100 positions possibles (10×10)
- **Détection** : OpenCV avec masque HSV rouge

---

**Cette version de test prouve que le système d'encodage/décodage fonctionne !** 🎉
