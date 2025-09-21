#!/usr/bin/env python3
"""
Script de test pour l'encodage des sous-titres
"""

import subprocess
import sys
import os

def test_encoding():
    """Test l'encodage avec les fichiers disponibles"""
    
    # Vérifier que les fichiers existent
    video_file = "video.mp4"
    srt_file = "subtitle.srt"
    output_file = "video_encoded.mp4"
    
    if not os.path.exists(video_file):
        print(f"Erreur: {video_file} n'existe pas")
        return False
    
    if not os.path.exists(srt_file):
        print(f"Erreur: {srt_file} n'existe pas")
        return False
    
    print("Fichiers trouvés, lancement de l'encodage...")
    
    # Lancer l'encodage
    cmd = [
        sys.executable, "encode_subtitles.py",
        "--video", video_file,
        "--srt", srt_file,
        "--output", output_file,
        "--grid-width", "10",
        "--grid-height", "10"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Encodage réussi!")
        print("Sortie:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'encodage: {e}")
        print("Erreur:", e.stderr)
        return False

if __name__ == "__main__":
    success = test_encoding()
    if success:
        print("\n✅ Test d'encodage réussi!")
        print("Vidéo encodée: video_encoded.mp4")
        print("Mapping de référence: video_encoded_mapping.json")
    else:
        print("\n❌ Test d'encodage échoué!")
        sys.exit(1)
