#!/usr/bin/env python3
"""
Script de test pour le décodage des sous-titres
"""

import subprocess
import sys
import os

def test_decoding():
    """Test le décodage avec la vidéo encodée"""
    
    # Vérifier que les fichiers existent
    video_file = "video_encoded.mp4"
    mapping_file = "video_encoded_mapping.json"
    output_file = "decoded_subtitles.txt"
    
    if not os.path.exists(video_file):
        print(f"Erreur: {video_file} n'existe pas")
        return False
    
    if not os.path.exists(mapping_file):
        print(f"Erreur: {mapping_file} n'existe pas")
        return False
    
    print("Fichiers trouvés, lancement du décodage...")
    
    # Lancer le décodage
    cmd = [
        sys.executable, "decode_subtitles.py",
        "--video", video_file,
        "--mapping", mapping_file,
        "--output", output_file
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Décodage réussi!")
        print("Sortie:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors du décodage: {e}")
        print("Erreur:", e.stderr)
        return False

if __name__ == "__main__":
    success = test_decoding()
    if success:
        print("\n✅ Test de décodage réussi!")
        print("Sous-titres décodés: decoded_subtitles.txt")
    else:
        print("\n❌ Test de décodage échoué!")
        sys.exit(1)
