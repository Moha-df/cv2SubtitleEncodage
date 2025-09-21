#!/usr/bin/env python3
"""
Décodeur avec amélioration progressive silencieuse - affiche seulement la phrase parfaite finale
"""

import cv2
import numpy as np
import json
import argparse
import os
from typing import List, Tuple, Dict
from collections import Counter
import re

class SubtitleDecoderSilentImprovement:
    def __init__(self, mapping_file: str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        
        print(f"Grille: {self.grid_width}×{self.grid_height}")
        print(f"Taille points: {self.point_size}")
        
        self.decoded_subtitles = []
        
    def detect_circles_optimized(self, frame: np.ndarray, frame_count: int) -> List[Tuple[int, int]]:
        """Détection optimisée des cercles rouges"""
        detected_positions = []
        frame_height, frame_width = frame.shape[:2]
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # HoughCircles
        hough_positions = self._detect_hough_circles(frame, cell_width, cell_height)
        
        # Contours
        additional_positions = self._detect_by_contours(frame, cell_width, cell_height)
        
        # Combiner et dédupliquer
        all_positions = hough_positions + additional_positions
        detected_positions = list(set(all_positions))
        
        return detected_positions
    
    def _detect_hough_circles(self, frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
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
    
    def _detect_by_contours(self, frame: np.ndarray, cell_width: int, cell_height: int) -> List[Tuple[int, int]]:
        """Détection par contours optimisée"""
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
    
    def is_same_phrase(self, text1: str, text2: str, similarity_threshold: float = 0.3) -> bool:
        """Détermine si deux textes sont probablement la même phrase - VERSION TRÈS PERMISSIVE"""
        if not text1 or not text2:
            return False
        
        # Nettoyer les textes de manière plus agressive
        clean1 = re.sub(r'[^\w\s]', ' ', text1.lower())  # Remplacer tous les symboles par des espaces
        clean2 = re.sub(r'[^\w\s]', ' ', text2.lower())
        
        clean1 = re.sub(r'\s+', ' ', clean1).strip()
        clean2 = re.sub(r'\s+', ' ', clean2).strip()
        
        if len(clean1) < 2 or len(clean2) < 2:
            return False
        
        # Méthode 1: Vérifier si les textes ont des mots communs significatifs
        words1 = set(word for word in clean1.split() if len(word) > 2)
        words2 = set(word for word in clean2.split() if len(word) > 2)
        
        if words1 and words2:
            common_words = words1.intersection(words2)
            # Si au moins 2 mots de plus de 2 caractères en commun, c'est probablement la même phrase
            if len(common_words) >= 2:
                return True
            
            # Ou si au moins 40% des mots sont communs
            total_unique_words = len(words1.union(words2))
            if total_unique_words > 0 and len(common_words) / total_unique_words >= 0.4:
                return True
        
        # Méthode 2: Similarité de caractères (très permissive)
        similarity = self._calculate_similarity(clean1, clean2)
        
        # Méthode 3: Vérifier les sous-séquences communes
        common_subseq = self._longest_common_subsequence(clean1, clean2)
        min_len = min(len(clean1), len(clean2))
        
        if min_len > 0 and len(common_subseq) / min_len >= 0.5:
            return True
        
        return similarity > similarity_threshold
    
    def _longest_common_subsequence(self, s1: str, s2: str) -> str:
        """Trouve la plus longue sous-séquence commune"""
        m, n = len(s1), len(s2)
        dp = [["" for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + s1[i-1]
                else:
                    dp[i][j] = dp[i-1][j] if len(dp[i-1][j]) > len(dp[i][j-1]) else dp[i][j-1]
        
        return dp[m][n]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité entre deux textes"""
        # Similarité de Jaccard sur les mots
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Distance de Levenshtein normalisée
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0.0
        
        levenshtein_distance = self._levenshtein_distance(text1, text2)
        levenshtein_similarity = 1 - (levenshtein_distance / max_len)
        
        # Combiner les deux méthodes
        return (jaccard_similarity * 0.6) + (levenshtein_similarity * 0.4)
    
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
    
    def create_perfect_sentence(self, texts: List[str]) -> str:
        """
        Crée la phrase parfaite en analysant toutes les détections
        MÉTHODE AMÉLIORÉE avec nettoyage plus intelligent
        """
        if not texts:
            return ""
        
        # Nettoyer tous les textes
        cleaned_texts = []
        for text in texts:
            # Nettoyer mais garder la structure des mots
            cleaned = re.sub(r'[^\w\s:.,!?\'"()-]', ' ', text)  # Remplacer caractères bizarres par espaces
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if len(cleaned) > 3:
                cleaned_texts.append(cleaned)
        
        if not cleaned_texts:
            return ""
        
        # Si on n'a qu'un seul texte, le retourner nettoyé
        if len(cleaned_texts) == 1:
            return cleaned_texts[0]
        
        # Trouver la longueur médiane pour éviter les textes trop courts ou trop longs
        lengths = [len(text) for text in cleaned_texts]
        median_length = sorted(lengths)[len(lengths)//2]
        
        # Filtrer les textes qui sont trop différents de la longueur médiane
        filtered_texts = []
        for text in cleaned_texts:
            if abs(len(text) - median_length) <= median_length * 0.5:  # ±50% de la longueur médiane
                filtered_texts.append(text)
        
        if not filtered_texts:
            filtered_texts = cleaned_texts
        
        # Méthode 1: Trouver le texte le plus représentatif (celui qui ressemble le plus aux autres)
        best_representative = ""
        max_total_similarity = 0
        
        for candidate in filtered_texts:
            total_similarity = 0
            for other in filtered_texts:
                if candidate != other:
                    total_similarity += self._calculate_similarity(candidate.lower(), other.lower())
            
            if total_similarity > max_total_similarity:
                max_total_similarity = total_similarity
                best_representative = candidate
        
        # Méthode 2: Amélioration caractère par caractère du texte le plus représentatif
        if len(filtered_texts) > 2:
            improved_text = self._improve_text_by_voting(best_representative, filtered_texts)
            if improved_text:
                best_representative = improved_text
        
        # Nettoyer le résultat final
        final_result = re.sub(r'\s+', ' ', best_representative).strip()
        
        # Correction de mots communs connus
        final_result = self._apply_common_corrections(final_result)
        
        return final_result
    
    def _improve_text_by_voting(self, base_text: str, all_texts: List[str]) -> str:
        """Améliore le texte de base en utilisant le vote des autres textes"""
        improved = list(base_text)
        
        for i in range(len(base_text)):
            char_votes = Counter()
            char_votes[base_text[i]] = 2  # Donner un avantage au caractère original
            
            # Collecter les votes des autres textes
            for text in all_texts:
                if i < len(text):
                    char_votes[text[i]] += 1
            
            # Prendre le caractère le plus voté
            if char_votes:
                most_common_char = char_votes.most_common(1)[0][0]
                improved[i] = most_common_char
        
        return ''.join(improved)
    
    def _apply_common_corrections(self, text: str) -> str:
        """Applique des corrections communes"""
        corrections = {
            r'\bnaruto\b': 'Naruto',
            r'\bscreaming\b': 'screaming',
            r'\bshock\b': 'shock',
            r'\bquiet\b': 'quiet',
            r'\s+': ' ',  # Espaces multiples
        }
        
        result = text
        for pattern, replacement in corrections.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result.strip()
    
    def process_frame(self, frame: np.ndarray) -> str:
        """Traite une frame et retourne le texte décodé"""
        detected_positions = self.detect_circles_optimized(frame, 0)
        
        if detected_positions:
            binary_str = self.positions_to_binary(detected_positions)
            decoded_text = self.binary_to_text(binary_str)
            
            if decoded_text and len(decoded_text) > 2:
                return decoded_text
        
        return ""
    
    def decode_video(self, video_path: str, output_file: str = None, max_frames: int = None):
        """Décode avec amélioration progressive silencieuse"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        
        print("Décodage silencieux avec amélioration progressive...")
        print("Analyse en cours... (pas d'affichage jusqu'à la phrase parfaite)")
        
        # Variables pour l'amélioration progressive
        current_phrase_texts = []
        current_phrase_start_time = None
        no_text_frames = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            current_time = frame_count / fps
            
            # Décoder la frame actuelle
            decoded_text = self.process_frame(frame)
            
            if decoded_text:
                no_text_frames = 0
                
                if not current_phrase_texts:
                    # Nouvelle phrase détectée
                    current_phrase_texts = [decoded_text]
                    current_phrase_start_time = current_time
                
                else:
                    # Vérifier si c'est la même phrase avec une logique plus intelligente
                    current_best = self.create_perfect_sentence(current_phrase_texts)
                    
                    # Plusieurs vérifications pour s'assurer que c'est une nouvelle phrase
                    is_new_phrase = True
                    
                    # 1. Vérifier contre le texte actuel
                    if self.is_same_phrase(current_best, decoded_text):
                        is_new_phrase = False
                    
                    # 2. Vérifier contre les derniers textes de la phrase actuelle
                    if is_new_phrase and len(current_phrase_texts) > 0:
                        recent_texts = current_phrase_texts[-3:]  # Vérifier les 3 derniers
                        for recent_text in recent_texts:
                            if self.is_same_phrase(recent_text, decoded_text):
                                is_new_phrase = False
                                break
                    
                    if not is_new_phrase:
                        # Même phrase, ajouter pour amélioration
                        current_phrase_texts.append(decoded_text)
                    else:
                        # Phrase différente, finaliser la précédente
                        if current_phrase_texts:
                            perfect_sentence = self.create_perfect_sentence(current_phrase_texts)
                            
                            if perfect_sentence:
                                self.decoded_subtitles.append({
                                    'start_time': current_phrase_start_time,
                                    'end_time': current_time,
                                    'text': perfect_sentence,
                                    'frames_count': len(current_phrase_texts),
                                    'confidence': len(current_phrase_texts) / max(1, frame_count - int(current_phrase_start_time * fps))
                                })
                                
                                # Afficher SEULEMENT la phrase parfaite finale
                                print(f"✓ PHRASE PARFAITE: '{perfect_sentence}' "
                                      f"({current_phrase_start_time:.1f}s-{current_time:.1f}s, "
                                      f"{len(current_phrase_texts)} détections)")
                        
                        # Commencer nouvelle phrase
                        current_phrase_texts = [decoded_text]
                        current_phrase_start_time = current_time
            
            else:
                # Pas de texte détecté
                no_text_frames += 1
                
                # Finaliser si pas de texte pendant un moment
                if no_text_frames > 30 and current_phrase_texts:  # Seuil encore plus élevé
                    perfect_sentence = self.create_perfect_sentence(current_phrase_texts)
                    
                    if perfect_sentence:
                        self.decoded_subtitles.append({
                            'start_time': current_phrase_start_time,
                            'end_time': current_time,
                            'text': perfect_sentence,
                            'frames_count': len(current_phrase_texts),
                            'confidence': len(current_phrase_texts) / max(1, frame_count - int(current_phrase_start_time * fps))
                        })
                        
                        # Afficher SEULEMENT la phrase parfaite finale
                        print(f"✓ PHRASE PARFAITE: '{perfect_sentence}' "
                              f"({current_phrase_start_time:.1f}s-{current_time:.1f}s, "
                              f"{len(current_phrase_texts)} détections)")
                    
                    # Reset
                    current_phrase_texts = []
                    current_phrase_start_time = None
                    no_text_frames = 0
            
            frame_count += 1
            processed_frames += 1
            
            # Affichage du progrès tous les 100 frames
            if processed_frames % 100 == 0:
                print(f"Traité {processed_frames} frames...")
        
        # Finaliser la dernière phrase s'il y en a une
        if current_phrase_texts:
            perfect_sentence = self.create_perfect_sentence(current_phrase_texts)
            
            if perfect_sentence:
                self.decoded_subtitles.append({
                    'start_time': current_phrase_start_time,
                    'end_time': current_time,
                    'text': perfect_sentence,
                    'frames_count': len(current_phrase_texts),
                    'confidence': len(current_phrase_texts) / max(1, frame_count - int(current_phrase_start_time * fps))
                })
                
                print(f"✓ PHRASE PARFAITE: '{perfect_sentence}' "
                      f"({current_phrase_start_time:.1f}s-{current_time:.1f}s, "
                      f"{len(current_phrase_texts)} détections)")
        
        cap.release()
        
        # Sauvegarder les résultats
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("SOUS-TITRES PARFAITS (Amélioration Progressive)\n")
                f.write("="*50 + "\n\n")
                
                for i, subtitle in enumerate(self.decoded_subtitles):
                    f.write(f"{i+1}. [{subtitle['start_time']:.2f}s - {subtitle['end_time']:.2f}s]\n")
                    f.write(f"   {subtitle['text']}\n")
                    f.write(f"   (Confiance: {subtitle['confidence']:.2f}, {subtitle['frames_count']} détections)\n\n")
        
        print(f"\n🎉 DÉCODAGE TERMINÉ!")
        print(f"📝 {len(self.decoded_subtitles)} phrases parfaites extraites")
        print(f"🎯 Chaque phrase a été optimisée à partir de multiples détections")
        
        return self.decoded_subtitles

def main():
    parser = argparse.ArgumentParser(description='Décodeur avec amélioration progressive silencieuse')
    parser.add_argument('--video', required=True, help='Vidéo source')
    parser.add_argument('--mapping', required=True, help='Fichier de mapping')
    parser.add_argument('--output', help='Fichier de sortie')
    parser.add_argument('--max-frames', type=int, help='Nombre max de frames')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Erreur: {args.video} n'existe pas")
        return
    
    if not os.path.exists(args.mapping):
        print(f"❌ Erreur: {args.mapping} n'existe pas")
        return
    
    decoder = SubtitleDecoderSilentImprovement(args.mapping)
    
    try:
        decoded_subtitles = decoder.decode_video(args.video, args.output, args.max_frames)
        
        print("\n" + "="*60)
        print("RÉSUMÉ FINAL - PHRASES PARFAITES:")
        print("="*60)
        
        for i, subtitle in enumerate(decoded_subtitles):
            confidence_stars = "⭐" * min(5, int(subtitle['confidence'] * 5))
            print(f"{i+1:2d}. [{subtitle['start_time']:6.1f}s-{subtitle['end_time']:6.1f}s] {confidence_stars}")
            print(f"     '{subtitle['text']}'")
            print(f"     ({subtitle['frames_count']} détections)")
            print()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()