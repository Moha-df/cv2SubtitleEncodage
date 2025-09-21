#!/usr/bin/env python3
"""
Décodeur avec détection de cercles optimisée
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple, Dict

class SubtitleDecoderCirclesOptimized:
    def __init__(self, mapping_file: str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        
    def detect_circles_optimized(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Détection de cercles optimisée avec plusieurs méthodes
        """
        detected_positions = []
        frame_height, frame_width = frame.shape[:2]
        
        # Calculer la taille des cellules
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # Méthode 1: HoughCircles avec paramètres optimisés
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détecter les cercles avec des paramètres plus permissifs
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(cell_width//3, 10),  # Distance minimale réduite
            param1=30,                       # Seuil Canny réduit
            param2=20,                       # Seuil de détection réduit
            minRadius=max(3, self.point_size-3),
            maxRadius=self.point_size+6
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if self._is_red_circle(frame, x, y, r):
                    grid_x = x // cell_width
                    grid_y = y // cell_height
                    
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        detected_positions.append((grid_x, grid_y))
        
        # Méthode 2: Détection par contours si pas assez de cercles
        if len(detected_positions) < 20:  # Seuil arbitraire
            additional_positions = self._detect_by_contours(frame, cell_width, cell_height)
            detected_positions.extend(additional_positions)
        
        # Supprimer les doublons
        detected_positions = list(set(detected_positions))
        
        return detected_positions
    
    def _detect_by_contours(self, frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """
        Détection par contours pour compléter HoughCircles
        """
        positions = []
        
        # Convertir en HSV et détecter le rouge
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Trouver les contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < 200:  # Filtre par taille
                # Vérifier si c'est circulaire
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Seuil de circularité
                        # Trouver le centre
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            grid_x = cx // cell_width
                            grid_y = cy // cell_height
                            
                            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                                positions.append((grid_x, grid_y))
        
        return positions
    
    def _is_red_circle(self, frame: np.ndarray, x: int, y: int, r: int) -> bool:
        """Vérifie si le cercle est rouge"""
        # Créer un masque circulaire
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Calculer la couleur moyenne
        mean_color = cv2.mean(frame, mask=mask)
        b, g, r = mean_color[:3]
        
        # Vérifier si c'est rouge
        return r > 80 and g < 120 and b < 120
    
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
                    if 32 <= char_code <= 126:
                        text += chr(char_code)
                    elif char_code == 0:
                        text += ' '
                except:
                    pass
        return text.strip()
    
    def decode_video(self, video_path: str, output_file: str = None):
        """Décode la vidéo avec détection optimisée"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        decoded_subtitles = []
        
        print("Décodage avec détection de cercles optimisée...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Détecter les cercles
            detected_positions = self.detect_circles_optimized(frame)
            
            if detected_positions:
                # Convertir en texte
                binary_str = self.positions_to_binary(detected_positions)
                decoded_text = self.binary_to_text(binary_str)
                
                if decoded_text and len(decoded_text) > 2:
                    decoded_subtitles.append({
                        'time': current_time,
                        'frame': frame_count,
                        'text': decoded_text,
                        'positions': detected_positions
                    })
                    
                    if frame_count % (fps * 2) == 0:
                        print(f"Frame {frame_count} - Cercles: {len(detected_positions)} - Texte: '{decoded_text}'")
            
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
    parser = argparse.ArgumentParser(description='Décoder avec cercles optimisés')
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
    
    decoder = SubtitleDecoderCirclesOptimized(args.mapping)
    
    try:
        decoded_subtitles = decoder.decode_video(args.video, args.output)
        
        print("\nExemples de sous-titres décodés:")
        for i, subtitle in enumerate(decoded_subtitles[:10]):
            print(f"{i+1}. {subtitle['time']:.2f}s: '{subtitle['text']}' ({len(subtitle['positions'])} cercles)")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
