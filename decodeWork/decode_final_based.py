#!/usr/bin/env python3
"""
Décodeur basé sur decode_circles_optimized mais amélioré
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple, Dict, Set

class SubtitleDecoderFinalBased:
    def __init__(self, mapping_file: str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        
        print(f"Grille: {self.grid_width}×{self.grid_height}")
        print(f"Taille points: {self.point_size}")
        
    def detect_circles_optimized(self, frame: np.ndarray, frame_count: int) -> List[Tuple[int, int]]:
        """
        Détection optimisée basée sur decode_circles_optimized
        """
        detected_positions = []
        frame_height, frame_width = frame.shape[:2]
        
        # Calculer la taille des cellules
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # Créer une image de debug
        debug_frame = frame.copy()
        
        # Dessiner la grille
        for i in range(self.grid_width + 1):
            x = i * cell_width
            cv2.line(debug_frame, (x, 0), (x, frame_height), (128, 128, 128), 1)
        
        for i in range(self.grid_height + 1):
            y = i * cell_height
            cv2.line(debug_frame, (0, y), (frame_width, y), (128, 128, 128), 1)
        
        # Méthode 1: HoughCircles avec paramètres optimisés
        hough_positions = self._detect_hough_circles(frame, debug_frame, cell_width, cell_height)
        
        # Méthode 2: Détection par contours si pas assez de cercles
        additional_positions = self._detect_by_contours(frame, debug_frame, cell_width, cell_height)
        
        # Combiner les deux méthodes (pas de consensus strict)
        all_positions = hough_positions + additional_positions
        
        # Supprimer les doublons
        detected_positions = list(set(all_positions))
        
        # Marquer les positions détectées
        for x, y in detected_positions:
            center_x = x * cell_width + cell_width // 2
            center_y = y * cell_height + cell_height // 2
            cv2.circle(debug_frame, (center_x, center_y), 4, (0, 255, 255), 2)  # CYAN
            cv2.putText(debug_frame, f"({x},{y})", 
                      (center_x-8, center_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # Statistiques
        cv2.putText(debug_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Hough: {len(hough_positions)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Contours: {len(additional_positions)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(debug_frame, f"Total: {len(detected_positions)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Sauvegarder l'image de debug
        #debug_filename = f"final_debug_{frame_count:04d}.jpg"
        #cv2.imwrite(debug_filename, debug_frame)
        
        return detected_positions
    
    def _detect_hough_circles(self, frame: np.ndarray, debug_frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """Détection HoughCircles optimisée"""
        positions = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Paramètres optimisés selon la taille de grille
        if self.grid_width <= 10:
            # Grille 10×10 - paramètres originaux
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(cell_width//3, 10),
                param1=30,
                param2=20,
                minRadius=max(3, self.point_size-3),
                maxRadius=self.point_size+6
            )
        else:
            # Grille 16×16 - paramètres ajustés
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(cell_width//2, 5),
                param1=20,
                param2=15,
                minRadius=max(2, self.point_size-2),
                maxRadius=self.point_size+3
            )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if self._is_red_circle(frame, x, y, r):
                    grid_x = x // cell_width
                    grid_y = y // cell_height
                    
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        positions.append((grid_x, grid_y))
                        cv2.circle(debug_frame, (x, y), r, (0, 255, 0), 1)
                        cv2.circle(debug_frame, (x, y), 2, (0, 255, 0), -1)
        
        return positions
    
    def _detect_by_contours(self, frame: np.ndarray, debug_frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """Détection par contours optimisée"""
        positions = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Seuils ajustés selon la grille
        if self.grid_width <= 10:
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            min_area, max_area = 15, 200
            circularity_threshold = 0.3
        else:
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 30, 30])
            upper_red2 = np.array([180, 255, 255])
            min_area, max_area = 8, 150
            circularity_threshold = 0.2
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > circularity_threshold:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            grid_x = cx // cell_width
                            grid_y = cy // cell_height
                            
                            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                                positions.append((grid_x, grid_y))
                                cv2.drawContours(debug_frame, [contour], -1, (255, 0, 0), 1)
                                cv2.circle(debug_frame, (cx, cy), 2, (255, 0, 0), -1)
        
        return positions
    
    def _is_red_circle(self, frame: np.ndarray, x: int, y: int, r: int) -> bool:
        """Vérifie si le cercle est rouge"""
        if x < r or y < r or x + r >= frame.shape[1] or y + r >= frame.shape[0]:
            return False
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        mean_color = cv2.mean(frame, mask=mask)
        b, g, r_val = mean_color[:3]
        
        # Seuils ajustés selon la grille
        if self.grid_width <= 10:
            return r_val > 80 and g < 120 and b < 120
        else:
            return r_val > 60 and g < 150 and b < 150
    
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
    
    def decode_video(self, video_path: str, output_file: str = None, max_frames: int = None):
        """Décode avec détection optimisée"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        decoded_subtitles = []
        
        print("Décodage optimisé basé sur decode_circles_optimized...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            current_time = frame_count / fps
            
            detected_positions = self.detect_circles_optimized(frame, frame_count)
            
            if detected_positions:
                binary_str = self.positions_to_binary(detected_positions)
                decoded_text = self.binary_to_text(binary_str)
                
                if decoded_text and len(decoded_text) > 2:
                    decoded_subtitles.append({
                        'time': current_time,
                        'frame': frame_count,
                        'text': decoded_text,
                        'positions': detected_positions
                    })
                    
                    print(f"Frame {frame_count} - Points: {len(detected_positions)} - Texte: '{decoded_text}'")
            
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
    parser = argparse.ArgumentParser(description='Décoder basé sur decode_circles_optimized')
    parser.add_argument('--video', required=True, help='Vidéo source')
    parser.add_argument('--mapping', required=True, help='Fichier de mapping')
    parser.add_argument('--output', help='Fichier de sortie')
    parser.add_argument('--max-frames', type=int, help='Nombre max de frames')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Erreur: {args.video} n'existe pas")
        return
    
    if not os.path.exists(args.mapping):
        print(f"Erreur: {args.mapping} n'existe pas")
        return
    
    decoder = SubtitleDecoderFinalBased(args.mapping)
    
    try:
        decoded_subtitles = decoder.decode_video(args.video, args.output, args.max_frames)
        
        print("\nExemples de sous-titres décodés:")
        for i, subtitle in enumerate(decoded_subtitles[:10]):
            print(f"{i+1}. {subtitle['time']:.2f}s: '{subtitle['text']}' ({len(subtitle['positions'])} points)")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
