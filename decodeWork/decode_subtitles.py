#!/usr/bin/env python3
"""
Script de décodage des sous-titres depuis une vidéo encodée
Test pour vérifier que l'encodage fonctionne correctement
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple, Dict

class SubtitleDecoder:
    def __init__(self, mapping_file: str):
        """
        Initialise le décodeur avec le fichier de mapping
        
        Args:
            mapping_file: Chemin vers le fichier de mapping JSON
        """
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        self.point_intensity = self.mapping_data['point_intensity']
        self.video_props = self.mapping_data['video_properties']
        
    def detect_points_in_frame(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Détecte les points encodés dans une frame
        
        Args:
            frame: Frame de la vidéo
            
        Returns:
            Liste des positions détectées
        """
        detected_positions = []
        frame_height, frame_width = frame.shape[:2]
        
        # Calculer la taille des cellules de la grille
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # Parcourir chaque cellule de la grille
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Calculer la position du centre de la cellule
                center_x = x * cell_width + cell_width // 2
                center_y = y * cell_height + cell_height // 2
                
                # Vérifier si un point est présent dans cette cellule
                if self._is_point_present(frame, center_x, center_y):
                    detected_positions.append((x, y))
        
        return detected_positions
    
    def _is_point_present(self, frame: np.ndarray, center_x: int, center_y: int) -> bool:
        """
        Vérifie si un point est présent à une position donnée
        
        Args:
            frame: Frame de la vidéo
            center_x: Position X du centre
            center_y: Position Y du centre
            
        Returns:
            True si un point est détecté
        """
        # Zone de recherche autour du centre
        search_radius = self.point_size + 1
        
        # Vérifier les pixels dans la zone de recherche
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                px = center_x + dx
                py = center_y + dy
                
                if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                    # Vérifier si la variation d'intensité est suffisante
                    pixel_value = np.mean(frame[py, px])
                    if pixel_value > 200:  # Seuil de détection
                        return True
        
        return False
    
    def positions_to_binary(self, positions: List[Tuple[int, int]]) -> str:
        """
        Convertit les positions détectées en chaîne binaire
        
        Args:
            positions: Liste des positions détectées
            
        Returns:
            Chaîne binaire reconstituée
        """
        binary_str = ['0'] * (self.grid_width * self.grid_height)
        
        for x, y in positions:
            index = y * self.grid_width + x
            if index < len(binary_str):
                binary_str[index] = '1'
        
        return ''.join(binary_str)
    
    def binary_to_text(self, binary_str: str) -> str:
        """
        Convertit une chaîne binaire en texte
        
        Args:
            binary_str: Chaîne binaire
            
        Returns:
            Texte reconstitué
        """
        text = ""
        # Grouper par 8 bits (1 caractère)
        for i in range(0, len(binary_str), 8):
            if i + 8 <= len(binary_str):
                char_binary = binary_str[i:i+8]
                try:
                    char_code = int(char_binary, 2)
                    if char_code > 0:  # Ignorer les caractères nuls
                        text += chr(char_code)
                except:
                    pass
        return text
    
    def decode_video(self, video_path: str, output_file: str = None):
        """
        Décode les sous-titres d'une vidéo encodée
        
        Args:
            video_path: Chemin vers la vidéo encodée
            output_file: Fichier de sortie pour les sous-titres décodés
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        decoded_subtitles = []
        
        print("Décodage en cours...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Détecter les points dans la frame
            detected_positions = self.detect_points_in_frame(frame)
            
            if detected_positions:
                # Convertir en binaire puis en texte
                binary_str = self.positions_to_binary(detected_positions)
                decoded_text = self.binary_to_text(binary_str)
                
                if decoded_text.strip():
                    decoded_subtitles.append({
                        'time': current_time,
                        'frame': frame_count,
                        'text': decoded_text.strip(),
                        'positions': detected_positions
                    })
                    
                    if frame_count % (fps * 2) == 0:  # Afficher toutes les 2 secondes
                        print(f"Frame {frame_count} - Texte décodé: '{decoded_text.strip()}'")
            
            frame_count += 1
        
        cap.release()
        
        # Sauvegarder les résultats
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for subtitle in decoded_subtitles:
                    f.write(f"{subtitle['time']:.2f}s: {subtitle['text']}\n")
        
        print(f"\nDécodage terminé!")
        print(f"Trouvé {len(decoded_subtitles)} sous-titres décodés")
        
        return decoded_subtitles

def main():
    parser = argparse.ArgumentParser(description='Décoder des sous-titres depuis une vidéo encodée')
    parser.add_argument('--video', required=True, help='Chemin vers la vidéo encodée')
    parser.add_argument('--mapping', required=True, help='Chemin vers le fichier de mapping')
    parser.add_argument('--output', help='Fichier de sortie pour les sous-titres décodés')
    
    args = parser.parse_args()
    
    # Vérifier que les fichiers existent
    if not os.path.exists(args.video):
        print(f"Erreur: La vidéo {args.video} n'existe pas")
        return
    
    if not os.path.exists(args.mapping):
        print(f"Erreur: Le fichier de mapping {args.mapping} n'existe pas")
        return
    
    # Créer le décodeur
    decoder = SubtitleDecoder(args.mapping)
    
    try:
        # Décoder la vidéo
        decoded_subtitles = decoder.decode_video(args.video, args.output)
        
        # Afficher quelques exemples
        print("\nExemples de sous-titres décodés:")
        for i, subtitle in enumerate(decoded_subtitles[:10]):
            print(f"{i+1}. {subtitle['time']:.2f}s: {subtitle['text']}")
        
    except Exception as e:
        print(f"Erreur lors du décodage: {e}")

if __name__ == "__main__":
    main()
