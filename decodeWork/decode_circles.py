#!/usr/bin/env python3
"""
Décodeur utilisant la détection de cercles OpenCV
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple, Dict

class SubtitleDecoderCircles:
    def __init__(self, mapping_file: str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        
    def detect_circles(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Détecte les cercles rouges avec OpenCV HoughCircles
        """
        detected_positions = []
        frame_height, frame_width = frame.shape[:2]
        
        # Calculer la taille des cellules
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détecter les cercles avec HoughCircles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,                    # Résolution inverse
            minDist=cell_width//2,   # Distance minimale entre centres
            param1=50,               # Seuil haut pour Canny
            param2=30,               # Seuil pour la détection de centre
            minRadius=self.point_size-2,  # Rayon minimum
            maxRadius=self.point_size+4   # Rayon maximum
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Vérifier que c'est bien un cercle rouge
                if self._is_red_circle(frame, x, y, r):
                    # Convertir en coordonnées de grille
                    grid_x = x // cell_width
                    grid_y = y // cell_height
                    
                    # Vérifier que c'est dans la grille
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        detected_positions.append((grid_x, grid_y))
        
        return detected_positions
    
    def _is_red_circle(self, frame: np.ndarray, x: int, y: int, r: int) -> bool:
        """
        Vérifie si le cercle détecté est bien rouge
        """
        # Créer un masque circulaire
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Calculer la couleur moyenne dans le cercle
        mean_color = cv2.mean(frame, mask=mask)
        
        # Vérifier si c'est rouge (BGR: B faible, G faible, R élevé)
        b, g, r = mean_color[:3]
        return r > 100 and g < 100 and b < 100
    
    def positions_to_binary(self, positions: List[Tuple[int, int]]) -> str:
        """Convertit les positions en binaire"""
        binary_str = ['0'] * (self.grid_width * self.grid_height)
        
        for x, y in positions:
            index = y * self.grid_width + x
            if index < len(binary_str):
                binary_str[index] = '1'
        
        return ''.join(binary_str)
    
    def binary_to_text(self, binary_str: str) -> str:
        """Convertit binaire en texte avec validation"""
        text = ""
        for i in range(0, len(binary_str), 8):
            if i + 8 <= len(binary_str):
                char_binary = binary_str[i:i+8]
                try:
                    char_code = int(char_binary, 2)
                    # Filtrer les caractères valides
                    if 32 <= char_code <= 126:  # Caractères ASCII imprimables
                        text += chr(char_code)
                    elif char_code == 0:
                        text += ' '  # Espace pour les caractères nuls
                except:
                    pass
        return text.strip()
    
    def decode_video(self, video_path: str, output_file: str = None, debug: bool = False):
        """Décode la vidéo avec détection de cercles"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        decoded_subtitles = []
        
        print("Décodage avec détection de cercles...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Détecter les cercles
            detected_positions = self.detect_circles(frame)
            
            if detected_positions:
                # Convertir en texte
                binary_str = self.positions_to_binary(detected_positions)
                decoded_text = self.binary_to_text(binary_str)
                
                if decoded_text and len(decoded_text) > 2:
                    decoded_subtitles.append({
                        'time': current_time,
                        'frame': frame_count,
                        'text': decoded_text,
                        'positions': detected_positions,
                        'binary_length': len(binary_str)
                    })
                    
                    if frame_count % (fps * 2) == 0:
                        print(f"Frame {frame_count} - Cercles: {len(detected_positions)} - Texte: '{decoded_text}'")
            
            # Mode debug : sauvegarder les frames avec cercles détectés
            if debug and detected_positions and frame_count % (fps * 5) == 0:
                debug_frame = frame.copy()
                cell_width = frame.shape[1] // self.grid_width
                cell_height = frame.shape[0] // self.grid_height
                
                for x, y in detected_positions:
                    center_x = x * cell_width + cell_width // 2
                    center_y = y * cell_height + cell_height // 2
                    cv2.circle(debug_frame, (center_x, center_y), 5, (0, 255, 0), 2)
                
                cv2.imwrite(f"debug_frame_{frame_count}.jpg", debug_frame)
            
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
    parser = argparse.ArgumentParser(description='Décoder avec détection de cercles')
    parser.add_argument('--video', required=True, help='Vidéo avec points visibles')
    parser.add_argument('--mapping', required=True, help='Fichier de mapping')
    parser.add_argument('--output', help='Fichier de sortie')
    parser.add_argument('--debug', action='store_true', help='Mode debug avec images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Erreur: {args.video} n'existe pas")
        return
    
    if not os.path.exists(args.mapping):
        print(f"Erreur: {args.mapping} n'existe pas")
        return
    
    decoder = SubtitleDecoderCircles(args.mapping)
    
    try:
        decoded_subtitles = decoder.decode_video(args.video, args.output, args.debug)
        
        print("\nExemples de sous-titres décodés:")
        for i, subtitle in enumerate(decoded_subtitles[:10]):
            print(f"{i+1}. {subtitle['time']:.2f}s: '{subtitle['text']}' ({len(subtitle['positions'])} cercles)")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
