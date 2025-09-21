#!/usr/bin/env python3
"""
Décodeur pour les points VISIBLES (version de test)
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple, Dict

class SubtitleDecoderVisible:
    def __init__(self, mapping_file: str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        
    def detect_red_points(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Détecte les points ROUGES dans la frame
        """
        detected_positions = []
        frame_height, frame_width = frame.shape[:2]
        
        # Calculer la taille des cellules
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # Convertir en HSV pour détecter le rouge
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Définir la plage de couleurs pour le rouge
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Créer les masques
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Parcourir chaque cellule
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Zone de la cellule
                start_x = x * cell_width
                end_x = start_x + cell_width
                start_y = y * cell_height
                end_y = start_y + cell_height
                
                # Vérifier s'il y a du rouge dans cette cellule
                cell_mask = red_mask[start_y:end_y, start_x:end_x]
                if np.any(cell_mask > 0):
                    detected_positions.append((x, y))
        
        return detected_positions
    
    def positions_to_binary(self, positions: List[Tuple[int, int]]) -> str:
        """Convertit les positions en binaire"""
        binary_str = ['0'] * (self.grid_width * self.grid_height)
        
        for x, y in positions:
            index = y * self.grid_width + x
            if index < len(binary_str):
                binary_str[index] = '1'
        
        return ''.join(binary_str)
    
    def binary_to_text(self, binary_str: str) -> str:
        """Convertit binaire en texte"""
        text = ""
        for i in range(0, len(binary_str), 8):
            if i + 8 <= len(binary_str):
                char_binary = binary_str[i:i+8]
                try:
                    char_code = int(char_binary, 2)
                    if char_code > 0 and char_code < 128:  # Caractères ASCII valides
                        text += chr(char_code)
                except:
                    pass
        return text
    
    def decode_video(self, video_path: str, output_file: str = None):
        """Décode la vidéo avec points visibles"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        decoded_subtitles = []
        
        print("Décodage des points ROUGES...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Détecter les points rouges
            detected_positions = self.detect_red_points(frame)
            
            if detected_positions:
                # Convertir en texte
                binary_str = self.positions_to_binary(detected_positions)
                decoded_text = self.binary_to_text(binary_str)
                
                if decoded_text.strip():
                    decoded_subtitles.append({
                        'time': current_time,
                        'frame': frame_count,
                        'text': decoded_text.strip(),
                        'positions': detected_positions
                    })
                    
                    if frame_count % (fps * 2) == 0:
                        print(f"Frame {frame_count} - Points détectés: {len(detected_positions)} - Texte: '{decoded_text.strip()}'")
            
            frame_count += 1
        
        cap.release()
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for subtitle in decoded_subtitles:
                    f.write(f"{subtitle['time']:.2f}s: {subtitle['text']}\n")
        
        print(f"\nDécodage terminé!")
        print(f"Trouvé {len(decoded_subtitles)} sous-titres décodés")
        
        return decoded_subtitles

def main():
    parser = argparse.ArgumentParser(description='Décoder points ROUGES')
    parser.add_argument('--video', required=True, help='Vidéo avec points visibles')
    parser.add_argument('--mapping', required=True, help='Fichier de mapping')
    parser.add_argument('--output', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Erreur: {args.video} n'existe pas")
        return
    
    if not os.path.exists(args.mapping):
        print(f"Erreur: {args.mapping} n'existe pas")
        return
    
    decoder = SubtitleDecoderVisible(args.mapping)
    
    try:
        decoded_subtitles = decoder.decode_video(args.video, args.output)
        
        print("\nExemples de sous-titres décodés:")
        for i, subtitle in enumerate(decoded_subtitles[:10]):
            print(f"{i+1}. {subtitle['time']:.2f}s: {subtitle['text']}")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
