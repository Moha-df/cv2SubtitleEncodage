#!/usr/bin/env python3
"""
Décodeur ULTRA-AMÉLIORÉ avec détection optimale
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple, Dict, Set

class SubtitleDecoderUltraImproved:
    def __init__(self, mapping_file: str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        
        print(f"Grille: {self.grid_width}×{self.grid_height}")
        print(f"Taille points: {self.point_size}")
        
    def detect_circles_ultra_improved(self, frame: np.ndarray, frame_count: int) -> List[Tuple[int, int]]:
        """
        Détection ULTRA-AMÉLIORÉE avec 3 méthodes + validation
        """
        frame_height, frame_width = frame.shape[:2]
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # Image de debug
        debug_frame = frame.copy()
        
        # Grille plus visible
        for i in range(self.grid_width + 1):
            x = i * cell_width
            cv2.line(debug_frame, (x, 0), (x, frame_height), (100, 100, 100), 1)
        
        for i in range(self.grid_height + 1):
            y = i * cell_height
            cv2.line(debug_frame, (0, y), (frame_width, y), (100, 100, 100), 1)
        
        # MÉTHODE 1: HoughCircles avec paramètres multiples
        hough_positions = self._detect_hough_multi_params(frame, debug_frame, cell_width, cell_height)
        
        # MÉTHODE 2: Détection par couleur rouge précise
        color_positions = self._detect_red_color_precise(frame, debug_frame, cell_width, cell_height)
        
        # MÉTHODE 3: Détection par contours optimisée
        contour_positions = self._detect_contours_optimized(frame, debug_frame, cell_width, cell_height)
        
        # MÉTHODE 4: Détection par template matching
        template_positions = self._detect_template_matching(frame, debug_frame, cell_width, cell_height)
        
        # CONSENSUS: Points détectés par AU MOINS 2 méthodes
        all_positions = hough_positions + color_positions + contour_positions + template_positions
        
        # Compter les votes pour chaque position
        position_votes = {}
        for pos in all_positions:
            position_votes[pos] = position_votes.get(pos, 0) + 1
        
        # Garder seulement les positions avec au moins 2 votes
        consensus_positions = [pos for pos, votes in position_votes.items() if votes >= 2]
        
        # Marquer les positions de consensus
        for x, y in consensus_positions:
            center_x = x * cell_width + cell_width // 2
            center_y = y * cell_height + cell_height // 2
            cv2.circle(debug_frame, (center_x, center_y), 6, (0, 255, 255), 2)  # CYAN
            cv2.putText(debug_frame, f"({x},{y})", 
                      (center_x-8, center_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # Statistiques détaillées
        cv2.putText(debug_frame, f"Frame: {frame_count}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"Hough: {len(hough_positions)}", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_frame, f"Color: {len(color_positions)}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_frame, f"Contour: {len(contour_positions)}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_frame, f"Template: {len(template_positions)}", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(debug_frame, f"Consensus: {len(consensus_positions)}", (10, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Sauvegarder debug
        debug_filename = f"ultra_debug_{frame_count:04d}.jpg"
        cv2.imwrite(debug_filename, debug_frame)
        
        return consensus_positions
    
    def _detect_hough_multi_params(self, frame: np.ndarray, debug_frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """HoughCircles avec plusieurs paramètres"""
        positions = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Essayer plusieurs configurations de paramètres
        param_sets = [
            {'dp': 1, 'minDist': max(cell_width//3, 5), 'param1': 30, 'param2': 20, 'minRadius': 2, 'maxRadius': 8},
            {'dp': 1, 'minDist': max(cell_width//4, 3), 'param1': 20, 'param2': 15, 'minRadius': 1, 'maxRadius': 10},
            {'dp': 2, 'minDist': max(cell_width//2, 8), 'param1': 50, 'param2': 30, 'minRadius': 3, 'maxRadius': 12}
        ]
        
        for params in param_sets:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **params)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    if self._is_red_circle_improved(frame, x, y, r):
                        grid_x = x // cell_width
                        grid_y = y // cell_height
                        
                        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                            positions.append((grid_x, grid_y))
                            cv2.circle(debug_frame, (x, y), r, (0, 255, 0), 1)
        
        return list(set(positions))  # Supprimer doublons
    
    def _detect_red_color_precise(self, frame: np.ndarray, debug_frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """Détection précise par couleur rouge"""
        positions = []
        
        # Convertir en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Masques rouges multiples
        red_masks = []
        
        # Rouge vif
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_masks.append(mask1)
        
        # Rouge foncé
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_masks.append(mask2)
        
        # Rouge plus permissif
        lower_red3 = np.array([0, 30, 30])
        upper_red3 = np.array([15, 255, 255])
        mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
        red_masks.append(mask3)
        
        # Combiner tous les masques
        combined_mask = np.zeros_like(mask1)
        for mask in red_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 200:  # Seuils larges
                # Vérifier la circularité
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.1:  # Seuil très permissif
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            grid_x = cx // cell_width
                            grid_y = cy // cell_height
                            
                            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                                positions.append((grid_x, grid_y))
                                cv2.drawContours(debug_frame, [contour], -1, (0, 0, 255), 1)
        
        return list(set(positions))
    
    def _detect_contours_optimized(self, frame: np.ndarray, debug_frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """Contours optimisés"""
        positions = []
        
        # Détection de contours sur image en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Filtre gaussien pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Détection de contours avec Canny adaptatif
        v = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        edges = cv2.Canny(blurred, lower, upper)
        
        # Dilatation pour connecter les contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 300:
                # Vérifier si c'est circulaire
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:
                        # Vérifier si c'est rouge
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            if self._is_red_circle_improved(frame, cx, cy, int(np.sqrt(area/np.pi))):
                                grid_x = cx // cell_width
                                grid_y = cy // cell_height
                                
                                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                                    positions.append((grid_x, grid_y))
                                    cv2.drawContours(debug_frame, [contour], -1, (255, 0, 0), 1)
        
        return list(set(positions))
    
    def _detect_template_matching(self, frame: np.ndarray, debug_frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """Détection par template matching"""
        positions = []
        
        # Créer un template de cercle rouge
        template_size = self.point_size * 2 + 4
        template = np.zeros((template_size, template_size, 3), dtype=np.uint8)
        center = template_size // 2
        cv2.circle(template, (center, center), self.point_size, (0, 0, 255), -1)
        
        # Convertir en niveaux de gris pour le matching
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Trouver les matches
        threshold = 0.6
        locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            # Vérifier si c'est rouge
            if self._is_red_circle_improved(frame, x + center, y + center, self.point_size):
                grid_x = (x + center) // cell_width
                grid_y = (y + center) // cell_height
                
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    positions.append((grid_x, grid_y))
                    cv2.rectangle(debug_frame, (x, y), (x + template_size, y + template_size), (255, 0, 255), 1)
        
        return list(set(positions))
    
    def _is_red_circle_improved(self, frame: np.ndarray, x: int, y: int, r: int) -> bool:
        """Vérification rouge améliorée"""
        if x < r or y < r or x + r >= frame.shape[1] or y + r >= frame.shape[0]:
            return False
        
        # Créer un masque circulaire
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Calculer la couleur moyenne dans le cercle
        mean_color = cv2.mean(frame, mask=mask)
        b, g, r_val = mean_color[:3]
        
        # Vérifier si c'est rouge (seuils ajustés)
        is_red = r_val > 50 and g < 200 and b < 200
        
        # Vérifier aussi la variance (couleur uniforme)
        if is_red:
            # Calculer la variance des couleurs
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            variance = np.var(masked_frame[mask > 0])
            is_red = is_red and variance < 5000  # Seuil de variance
        
        return is_red
    
    def positions_to_binary(self, positions: List[Tuple[int, int]]) -> str:
        """Convertit les positions en binaire"""
        binary_str = ['0'] * (self.grid_width * self.grid_height)
        
        for x, y in positions:
            index = y * self.grid_width + x
            if index < len(binary_str):
                binary_str[index] = '1'
        
        return ''.join(binary_str)
    
    def binary_to_text(self, binary_str: str) -> str:
        """Convertit binaire en texte avec correction d'erreurs"""
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
        """Décode avec détection ultra-améliorée"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        decoded_subtitles = []
        
        print("Décodage ULTRA-AMÉLIORÉ avec 4 méthodes...")
        print("Méthodes: HoughCircles + Couleur + Contours + Template")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            current_time = frame_count / fps
            
            detected_positions = self.detect_circles_ultra_improved(frame, frame_count)
            
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
                    
                    print(f"Frame {frame_count} - Consensus: {len(detected_positions)} - Texte: '{decoded_text}'")
            
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
    parser = argparse.ArgumentParser(description='Décoder ULTRA-AMÉLIORÉ')
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
    
    decoder = SubtitleDecoderUltraImproved(args.mapping)
    
    try:
        decoded_subtitles = decoder.decode_video(args.video, args.output, args.max_frames)
        
        print("\nExemples de sous-titres décodés:")
        for i, subtitle in enumerate(decoded_subtitles[:10]):
            print(f"{i+1}. {subtitle['time']:.2f}s: '{subtitle['text']}' ({len(subtitle['positions'])} consensus)")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
