#!/usr/bin/env python3
"""
Script pour visualiser les images de debug
"""

import cv2
import os
import sys
from pathlib import Path

def view_debug_images():
    """Affiche les images de debug une par une"""
    
    # Trouver toutes les images de debug
    debug_files = sorted([f for f in os.listdir('.') if f.startswith('debug_frame_') and f.endswith('.jpg')])
    
    if not debug_files:
        print("Aucune image de debug trouvée!")
        return
    
    print(f"Trouvé {len(debug_files)} images de debug")
    print("Contrôles:")
    print("- ESPACE: image suivante")
    print("- 'q': quitter")
    print("- 's': sauvegarder l'image actuelle")
    
    current_idx = 0
    
    while True:
        if current_idx >= len(debug_files):
            print("Fin des images")
            break
            
        filename = debug_files[current_idx]
        print(f"\nImage {current_idx + 1}/{len(debug_files)}: {filename}")
        
        # Charger et afficher l'image
        img = cv2.imread(filename)
        if img is None:
            print(f"Impossible de charger {filename}")
            current_idx += 1
            continue
        
        # Redimensionner si trop grande
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        cv2.imshow('Debug - Cercles Détectés', img)
        
        # Attendre une touche
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            current_idx += 1
        elif key == ord('s'):
            # Sauvegarder une copie
            save_name = f"selected_{filename}"
            cv2.imwrite(save_name, img)
            print(f"Image sauvegardée: {save_name}")
        elif key == ord('b'):  # Back
            if current_idx > 0:
                current_idx -= 1
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_debug_images()
