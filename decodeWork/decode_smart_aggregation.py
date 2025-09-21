#!/usr/bin/env python3
"""
Décodeur intelligent avec agrégation de frames
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple, Dict
from collections import Counter
import re

class SubtitleDecoderSmartAggregation:
    def __init__(self, mapping_file: str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        
        print(f"Grille: {self.grid_width}×{self.grid_height}")
        print(f"Taille points: {self.point_size}")
        
        # Variables pour l'agrégation
        self.current_subtitle_frames = []
        self.current_subtitle_start_time = None
        self.current_subtitle_text = None
        self.aggregated_subtitles = []
        
    def detect_circles_optimized(self, frame: np.ndarray, frame_count: int) -> List[Tuple[int, int]]:
        """Détection optimisée (même que decode_final_based)"""
        detected_positions = []
        frame_height, frame_width = frame.shape[:2]
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # Image de debug
        debug_frame = frame.copy()
        
        # Grille
        for i in range(self.grid_width + 1):
            x = i * cell_width
            cv2.line(debug_frame, (x, 0), (x, frame_height), (128, 128, 128), 1)
        
        for i in range(self.grid_height + 1):
            y = i * cell_height
            cv2.line(debug_frame, (0, y), (frame_width, y), (128, 128, 128), 1)
        
        # HoughCircles
        hough_positions = self._detect_hough_circles(frame, debug_frame, cell_width, cell_height)
        
        # Contours
        additional_positions = self._detect_by_contours(frame, debug_frame, cell_width, cell_height)
        
        # Combiner
        all_positions = hough_positions + additional_positions
        detected_positions = list(set(all_positions))
        
        # Marquer les positions
        for x, y in detected_positions:
            center_x = x * cell_width + cell_width // 2
            center_y = y * cell_height + cell_height // 2
            cv2.circle(debug_frame, (center_x, center_y), 4, (0, 255, 255), 2)
        
        return detected_positions
    
    def _detect_hough_circles(self, frame: np.ndarray, debug_frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """HoughCircles optimisé"""
        positions = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.grid_width <= 10:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=max(cell_width//3, 10), 
                                     param1=30, param2=20, minRadius=max(3, self.point_size-3), maxRadius=self.point_size+6)
        else:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=max(cell_width//2, 5), 
                                     param1=20, param2=15, minRadius=max(2, self.point_size-2), maxRadius=self.point_size+3)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if self._is_red_circle(frame, x, y, r):
                    grid_x = x // cell_width
                    grid_y = y // cell_height
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        positions.append((grid_x, grid_y))
        
        return positions
    
    def _detect_by_contours(self, frame: np.ndarray, debug_frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """Contours optimisé"""
        positions = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if self.grid_width <= 10:
            lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
            min_area, max_area, circularity_threshold = 15, 200, 0.3
        else:
            lower_red1, upper_red1 = np.array([0, 30, 30]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([170, 30, 30]), np.array([180, 255, 255])
            min_area, max_area, circularity_threshold = 8, 150, 0.2
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
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
        
        return positions
    
    def _is_red_circle(self, frame: np.ndarray, x: int, y: int, r: int) -> bool:
        """Vérifie si le cercle est rouge"""
        if x < r or y < r or x + r >= frame.shape[1] or y + r >= frame.shape[0]:
            return False
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        mean_color = cv2.mean(frame, mask=mask)
        b, g, r_val = mean_color[:3]
        
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
    
    def aggregate_texts(self, texts: List[str]) -> str:
        """
        Agrège plusieurs textes pour trouver la phrase la plus probable
        """
        if not texts:
            return ""
        
        # Nettoyer les textes
        cleaned_texts = []
        for text in texts:
            # Supprimer les caractères de fin bizarres
            cleaned = re.sub(r'[^\w\s:.,!?*\'"-]', '', text)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if len(cleaned) > 3:  # Garder seulement les textes significatifs
                cleaned_texts.append(cleaned)
        
        if not cleaned_texts:
            return ""
        
        # Méthode 1: Vote par caractère position par position
        max_length = max(len(text) for text in cleaned_texts)
        char_votes = {}
        
        for i in range(max_length):
            char_votes[i] = Counter()
            for text in cleaned_texts:
                if i < len(text):
                    char_votes[i][text[i]] += 1
        
        # Reconstruire le texte caractère par caractère
        reconstructed = ""
        for i in range(max_length):
            if i in char_votes and char_votes[i]:
                most_common_char = char_votes[i].most_common(1)[0][0]
                reconstructed += most_common_char
        
        # Méthode 2: Vote par mots
        all_words = []
        for text in cleaned_texts:
            words = text.split()
            all_words.extend(words)
        
        word_counter = Counter(all_words)
        
        # Méthode 3: Trouver le texte le plus proche de la moyenne
        if len(cleaned_texts) > 1:
            # Calculer la distance de Levenshtein entre tous les textes
            best_text = cleaned_texts[0]
            min_total_distance = float('inf')
            
            for candidate in cleaned_texts:
                total_distance = 0
                for other in cleaned_texts:
                    if candidate != other:
                        total_distance += self._levenshtein_distance(candidate, other)
                
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    best_text = candidate
            
            # Combiner les deux méthodes
            if len(reconstructed) > len(best_text):
                return reconstructed
            else:
                return best_text
        
        return reconstructed if reconstructed else cleaned_texts[0]
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calcule la distance de Levenshtein entre deux chaînes"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def process_frame(self, frame: np.ndarray, frame_count: int, current_time: float) -> str:
        """Traite une frame et retourne le texte décodé"""
        detected_positions = self.detect_circles_optimized(frame, frame_count)
        
        if detected_positions:
            binary_str = self.positions_to_binary(detected_positions)
            decoded_text = self.binary_to_text(binary_str)
            
            if decoded_text and len(decoded_text) > 2:
                return decoded_text
        
        return ""
    
    def decode_video(self, video_path: str, output_file: str = None, max_frames: int = None):
        """Décode avec agrégation intelligente"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        decoded_subtitles = []
        
        print("Décodage intelligent avec agrégation de frames...")
        print("Analyse de plusieurs frames pour reconstituer les phrases correctes")
        
        # Variables pour l'agrégation
        current_subtitle_frames = []
        current_subtitle_start_time = None
        last_text = ""
        stable_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            current_time = frame_count / fps
            
            # Décoder la frame actuelle
            decoded_text = self.process_frame(frame, frame_count, current_time)
            
            if decoded_text:
                # Si c'est un nouveau sous-titre (texte différent)
                if decoded_text != last_text:
                    # Finaliser le sous-titre précédent s'il y en a un
                    if current_subtitle_frames and current_subtitle_start_time is not None:
                        aggregated_text = self.aggregate_texts(current_subtitle_frames)
                        if aggregated_text:
                            decoded_subtitles.append({
                                'start_time': current_subtitle_start_time,
                                'end_time': current_time,
                                'text': aggregated_text,
                                'frames_count': len(current_subtitle_frames)
                            })
                            print(f"Sous-titre finalisé: '{aggregated_text}' ({len(current_subtitle_frames)} frames)")
                    
                    # Commencer un nouveau sous-titre
                    current_subtitle_frames = [decoded_text]
                    current_subtitle_start_time = current_time
                    last_text = decoded_text
                    stable_frames = 1
                else:
                    # Même texte, l'ajouter à l'agrégation
                    current_subtitle_frames.append(decoded_text)
                    stable_frames += 1
            else:
                # Pas de texte détecté
                if current_subtitle_frames and stable_frames > 10:  # Attendre un peu avant de finaliser
                    # Finaliser le sous-titre actuel
                    aggregated_text = self.aggregate_texts(current_subtitle_frames)
                    if aggregated_text:
                        decoded_subtitles.append({
                            'start_time': current_subtitle_start_time,
                            'end_time': current_time,
                            'text': aggregated_text,
                            'frames_count': len(current_subtitle_frames)
                        })
                        print(f"Sous-titre finalisé: '{aggregated_text}' ({len(current_subtitle_frames)} frames)")
                    
                    # Reset
                    current_subtitle_frames = []
                    current_subtitle_start_time = None
                    last_text = ""
                    stable_frames = 0
            
            frame_count += 1
        
        # Finaliser le dernier sous-titre s'il y en a un
        if current_subtitle_frames and current_subtitle_start_time is not None:
            aggregated_text = self.aggregate_texts(current_subtitle_frames)
            if aggregated_text:
                decoded_subtitles.append({
                    'start_time': current_subtitle_start_time,
                    'end_time': current_time,
                    'text': aggregated_text,
                    'frames_count': len(current_subtitle_frames)
                })
                print(f"Sous-titre finalisé: '{aggregated_text}' ({len(current_subtitle_frames)} frames)")
        
        cap.release()
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for subtitle in decoded_subtitles:
                    f.write(f"{subtitle['start_time']:.2f}s - {subtitle['end_time']:.2f}s: {subtitle['text']}\n")
        
        print(f"\nDécodage terminé!")
        print(f"Trouvé {len(decoded_subtitles)} sous-titres agrégés")
        
        return decoded_subtitles

def main():
    parser = argparse.ArgumentParser(description='Décoder intelligent avec agrégation')
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
    
    decoder = SubtitleDecoderSmartAggregation(args.mapping)
    
    try:
        decoded_subtitles = decoder.decode_video(args.video, args.output, args.max_frames)
        
        print("\nSous-titres agrégés:")
        for i, subtitle in enumerate(decoded_subtitles):
            print(f"{i+1}. {subtitle['start_time']:.2f}s-{subtitle['end_time']:.2f}s: '{subtitle['text']}' ({subtitle['frames_count']} frames)")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
