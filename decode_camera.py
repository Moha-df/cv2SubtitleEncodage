#!/usr/bin/env python3
"""
Décodeur de sous-titres en temps réel pour smartphone - VERSION COMPLÈTE
- Utilise la caméra en direct
- Affiche les sous-titres décodés en overlay
- Système de redressement automatique
- Optimisé pour la performance temps réel
"""

import cv2
import numpy as np
import json
import argparse
import time
import os
from typing import List, Tuple, Optional
from collections import deque, Counter
import threading
import queue
import re

class RealTimeSubtitleDecoder:
    def __init__(self, mapping_file: str):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        self.grid_width, self.grid_height = self.mapping_data['grid_size']
        self.point_size = self.mapping_data['point_size']
        
        # Buffer circulaire pour stocker les dernières détections
        self.detection_buffer = deque(maxlen=10)
        self.current_subtitle = ""
        self.subtitle_confidence = 0.0
        self.last_detection_time = 0
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=10)
        
        # Threading pour le traitement
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=1)
        self.processing_thread = None
        self.is_running = False
        
        # Système de redressement
        self.perspective_matrix = None
        self.screen_corners = None
        self.corner_detection_enabled = True
        self.manual_corners = None
        self.corner_buffer = deque(maxlen=5)
        self.debug_mode = False
        
        # Système de debug visuel
        self.debug_circles = False
        self.debug_frame = None
        self.last_detected_positions = []
        self.last_red_mask = None
        self.last_contours = []
        
        print(f"🎯 Décodeur temps réel initialisé")
        print(f"📱 Grille: {self.grid_width}×{self.grid_height}")
        print(f"🔴 Taille points: {self.point_size}")
        print(f"🔧 Redressement automatique activé")
    
    def detect_screen_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Détecte automatiquement les coins de l'écran pour le redressement"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection de contours pour trouver le rectangle de l'écran
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Dilatation pour connecter les lignes
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Trouver les contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chercher le plus grand contour rectangulaire
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            # Approximation polygonale
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Si c'est un quadrilatère et assez grand
            if len(approx) == 4 and cv2.contourArea(approx) > frame.shape[0] * frame.shape[1] * 0.1:
                # Ordonner les points
                corners = self.order_corners(approx.reshape(4, 2))
                return corners
        
        return None
    
    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Ordonne les coins dans le bon ordre pour la transformation perspective"""
        # Calculer les sommes et différences
        sums = corners.sum(axis=1)
        diffs = np.diff(corners, axis=1)
        
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(sums)]      # top-left
        ordered[2] = corners[np.argmax(sums)]      # bottom-right
        ordered[1] = corners[np.argmin(diffs)]     # top-right
        ordered[3] = corners[np.argmax(diffs)]     # bottom-left
        
        return ordered
    
    def calculate_perspective_matrix(self, corners: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Calcule la matrice de transformation perspective"""
        dst_corners = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(corners, dst_corners)
        return matrix
    
    def apply_perspective_correction(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Applique la correction de perspective si possible"""
        if not self.corner_detection_enabled:
            return frame
        
        # Détecter les coins
        corners = self.detect_screen_corners(frame)
        
        if corners is not None:
            # Ajouter au buffer pour stabiliser
            self.corner_buffer.append(corners)
            
            # Utiliser la moyenne des dernières détections pour stabiliser
            if len(self.corner_buffer) >= 3:
                # Calculer les coins moyens
                avg_corners = np.mean(list(self.corner_buffer), axis=0)
                
                # Calculer les dimensions cibles
                width1 = np.linalg.norm(avg_corners[1] - avg_corners[0])
                width2 = np.linalg.norm(avg_corners[2] - avg_corners[3])
                target_width = int((width1 + width2) / 2)
                
                height1 = np.linalg.norm(avg_corners[3] - avg_corners[0])
                height2 = np.linalg.norm(avg_corners[2] - avg_corners[1])
                target_height = int((height1 + height2) / 2)
                
                # Limiter les dimensions
                max_size = 1200
                if target_width > max_size or target_height > max_size:
                    scale = min(max_size / target_width, max_size / target_height)
                    target_width = int(target_width * scale)
                    target_height = int(target_height * scale)
                
                # Calculer et appliquer la transformation
                self.perspective_matrix = self.calculate_perspective_matrix(
                    avg_corners.astype(np.float32), target_width, target_height
                )
                
                corrected_frame = cv2.warpPerspective(
                    frame, self.perspective_matrix, (target_width, target_height)
                )
                
                self.screen_corners = avg_corners
                return corrected_frame
        
        # Si on a déjà une matrice de transformation stable, l'utiliser
        elif self.perspective_matrix is not None:
            try:
                # Obtenir les dimensions de la dernière transformation
                h, w = frame.shape[:2]
                corners_3d = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)
                transformed_corners = cv2.perspectiveTransform(corners_3d, self.perspective_matrix)
                
                target_width = int(np.max(transformed_corners[0, :, 0]))
                target_height = int(np.max(transformed_corners[0, :, 1]))
                
                corrected_frame = cv2.warpPerspective(
                    frame, self.perspective_matrix, (target_width, target_height)
                )
                return corrected_frame
            except:
                # Reset si erreur
                self.perspective_matrix = None
                self.screen_corners = None
        
        return frame
    
    def draw_corner_detection(self, frame: np.ndarray) -> np.ndarray:
        """Dessine les coins détectés sur la frame pour debug"""
        display_frame = frame.copy()
        
        if self.screen_corners is not None:
            corners = self.screen_corners.astype(int)
            
            # Dessiner les coins
            colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0)]
            for i, corner in enumerate(corners):
                cv2.circle(display_frame, tuple(corner), 8, colors[i], -1)
                cv2.putText(display_frame, str(i), tuple(corner + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
            
            # Dessiner les lignes du rectangle
            cv2.polylines(display_frame, [corners], True, (255, 255, 255), 2)
        
        return display_frame
    
    def detect_circles_fast(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Détection ultra-rapide optimisée pour le temps réel avec redressement et debug"""
        start_time = time.time()
        
        # Appliquer le redressement si activé
        corrected_frame = frame
        if self.corner_detection_enabled:
            corrected_frame = self.apply_perspective_correction(frame)
            if corrected_frame is None:
                corrected_frame = frame
        
        detected_positions = []
        frame_height, frame_width = corrected_frame.shape[:2]
        
        # Éviter division par zéro
        if frame_width == 0 or frame_height == 0:
            return detected_positions
            
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        # Créer frame de debug si activé
        if self.debug_circles:
            self.debug_frame = corrected_frame.copy()
            
            # Dessiner la grille
            for i in range(self.grid_width + 1):
                x = i * cell_width
                cv2.line(self.debug_frame, (x, 0), (x, frame_height), (128, 128, 128), 1)
            
            for i in range(self.grid_height + 1):
                y = i * cell_height
                cv2.line(self.debug_frame, (0, y), (frame_width, y), (128, 128, 128), 1)
        
        # Réduire la résolution pour accélérer le traitement
        scale_factor = 1.0
        if frame_width > 1280:
            scale_factor = 0.5
            small_frame = cv2.resize(corrected_frame, None, fx=scale_factor, fy=scale_factor)
        else:
            small_frame = corrected_frame
        
        # Détection par couleur rouge
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        
        # Masques rouges optimisés
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        
        # Sauvegarder le masque pour debug
        if self.debug_circles:
            self.last_red_mask = cv2.resize(red_mask, (frame_width, frame_height)) if scale_factor != 1.0 else red_mask.copy()
        
        # Morphologie rapide
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sauvegarder les contours pour debug
        if self.debug_circles:
            self.last_contours = []
        
        # Ajuster les tailles selon l'échelle
        min_area = 10 / (scale_factor * scale_factor)
        max_area = 200 / (scale_factor * scale_factor)
        
        valid_contours = 0
        rejected_contours = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Calculer le centre pour debug
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"] / scale_factor)
                cy = int(M["m01"] / M["m00"] / scale_factor)
                
                # Debug: sauvegarder tous les contours avec leur statut
                if self.debug_circles:
                    self.last_contours.append({
                        'center': (cx, cy),
                        'area': area,
                        'valid': min_area < area < max_area,
                        'contour': (contour / scale_factor).astype(int) if scale_factor != 1.0 else contour
                    })
                
                if min_area < area < max_area:
                    valid_contours += 1
                    
                    # Convertir en position de grille
                    if cell_width > 0 and cell_height > 0:
                        grid_x = cx // cell_width
                        grid_y = cy // cell_height
                        
                        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                            detected_positions.append((grid_x, grid_y))
                            
                            # Debug: marquer les positions valides
                            if self.debug_circles:
                                # Centre de la cellule
                                center_x = grid_x * cell_width + cell_width // 2
                                center_y = grid_y * cell_height + cell_height // 2
                                
                                # Cercle vert pour position valide
                                cv2.circle(self.debug_frame, (center_x, center_y), 8, (0, 255, 0), 2)
                                # Point détecté en rouge
                                cv2.circle(self.debug_frame, (cx, cy), 4, (0, 0, 255), -1)
                                # Ligne de connexion
                                cv2.line(self.debug_frame, (cx, cy), (center_x, center_y), (255, 255, 0), 1)
                                # Texte avec coordonnées de grille
                                cv2.putText(self.debug_frame, f"({grid_x},{grid_y})", 
                                           (center_x + 10, center_y - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    rejected_contours += 1
                    
                    # Debug: marquer les contours rejetés
                    if self.debug_circles:
                        color = (0, 0, 255) if area < min_area else (255, 0, 0)  # Rouge si trop petit, bleu si trop grand
                        cv2.circle(self.debug_frame, (cx, cy), 6, color, 1)
                        cv2.putText(self.debug_frame, f"A:{int(area)}", (cx + 8, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Sauvegarder les positions pour debug
        if self.debug_circles:
            self.last_detected_positions = detected_positions.copy()
            
            # Ajouter des statistiques sur la frame
            stats_text = [
                f"Detectes: {len(detected_positions)}",
                f"Valides: {valid_contours}",
                f"Rejetes: {rejected_contours}",
                f"Grille: {cell_width}x{cell_height}",
                f"Aires: {min_area:.0f}-{max_area:.0f}"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(self.debug_frame, text, (10, 30 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Enregistrer le temps de traitement
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
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
                    if 32 <= char_code <= 126:
                        text += chr(char_code)
                    elif char_code == 0:
                        text += ' '
                except:
                    pass
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte décodé"""
        if not text:
            return ""
        
        # Supprimer les caractères bizarres
        cleaned = re.sub(r'[^\w\s:.,!?\'"()-]', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Corrections communes
        corrections = {
            r'\bnaruto\b': 'Naruto',
            r'\bscreaming\b': 'screaming',
            r'\bshock\b': 'shock',
            r'\bquiet\b': 'quiet',
        }
        
        for pattern, replacement in corrections.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def is_similar_text(self, text1: str, text2: str) -> bool:
        """Vérifie si deux textes sont similaires (version rapide)"""
        if not text1 or not text2:
            return False
        
        # Nettoyage rapide
        clean1 = re.sub(r'[^\w\s]', ' ', text1.lower()).strip()
        clean2 = re.sub(r'[^\w\s]', ' ', text2.lower()).strip()
        
        if len(clean1) < 3 or len(clean2) < 3:
            return False
        
        # Vérification rapide par mots
        words1 = set(word for word in clean1.split() if len(word) > 2)
        words2 = set(word for word in clean2.split() if len(word) > 2)
        
        if not words1 or not words2:
            return False
        
        # Si au moins 50% des mots sont communs
        common = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return (common / total) > 0.5 if total > 0 else False
    
    def update_subtitle_buffer(self, detected_text: str):
        """Met à jour le buffer avec smoothing intelligent"""
        current_time = time.time()
        
        if detected_text and len(detected_text) > 2:
            # Nettoyer le texte
            clean_text = self.clean_text(detected_text)
            
            if clean_text:
                # Ajouter au buffer avec timestamp
                self.detection_buffer.append({
                    'text': clean_text,
                    'time': current_time,
                    'length': len(clean_text)
                })
                
                # Analyser le buffer pour déterminer le sous-titre stable
                self.analyze_buffer()
        
        self.last_detection_time = current_time
    
    def analyze_buffer(self):
        """Analyse le buffer pour extraire le sous-titre le plus stable"""
        if not self.detection_buffer:
            return
        
        # Grouper les textes similaires
        text_groups = {}
        
        for detection in self.detection_buffer:
            text = detection['text']
            found_group = False
            
            # Chercher un groupe existant
            for group_key, group_texts in text_groups.items():
                if self.is_similar_text(text, group_key):
                    group_texts.append(detection)
                    found_group = True
                    break
            
            # Créer un nouveau groupe
            if not found_group:
                text_groups[text] = [detection]
        
        # Trouver le groupe le plus fréquent et récent
        best_group = None
        best_score = 0
        
        for group_key, group_detections in text_groups.items():
            if len(group_detections) >= 2:  # Au moins 2 détections
                # Score basé sur fréquence et récence
                frequency_score = len(group_detections)
                recency_score = max(d['time'] for d in group_detections)
                total_score = frequency_score * 10 + recency_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_group = group_detections
        
        # Mettre à jour le sous-titre affiché
        if best_group:
            # Prendre le texte le plus long du groupe
            best_text = max(best_group, key=lambda x: x['length'])['text']
            confidence = len(best_group) / len(self.detection_buffer)
            
            if best_text != self.current_subtitle:
                self.current_subtitle = best_text
                self.subtitle_confidence = confidence
                print(f"📝 Nouveau sous-titre: '{best_text}' (confiance: {confidence:.2f})")
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Dessine l'overlay avec les informations"""
        # Si mode debug cercles activé, utiliser la frame de debug
        if self.debug_circles and self.debug_frame is not None:
            overlay_frame = self.debug_frame.copy()
        else:
            overlay_frame = frame.copy()
            
        height, width = overlay_frame.shape[:2]
        
        # Calculer FPS
        current_time = time.time()
        self.fps_counter.append(current_time)
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
        else:
            fps = 0
        
        # Zone de sous-titre (en bas)
        subtitle_height = 120 if self.debug_circles else 80  # Plus de place en mode debug
        subtitle_y = height - subtitle_height
        
        # Fond semi-transparent pour les sous-titres
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (0, subtitle_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay_frame, 0.7, overlay, 0.3, 0, overlay_frame)
        
        # Afficher le sous-titre
        if self.current_subtitle:
            # Calculer la taille du texte
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(width / 800, 1.2)
            thickness = max(1, int(font_scale * 2))
            
            # Mesurer le texte
            (text_width, text_height), baseline = cv2.getTextSize(
                self.current_subtitle, font, font_scale, thickness
            )
            
            # Centrer le texte
            text_x = (width - text_width) // 2
            text_y = subtitle_y + 30  # Un peu plus haut si debug actif
            
            # Couleur selon la confiance
            if self.subtitle_confidence > 0.7:
                color = (0, 255, 0)  # Vert
            elif self.subtitle_confidence > 0.4:
                color = (0, 255, 255)  # Jaune
            else:
                color = (0, 0, 255)  # Rouge
            
            # Dessiner le texte avec contour
            cv2.putText(overlay_frame, self.current_subtitle, (text_x, text_y),
                       font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(overlay_frame, self.current_subtitle, (text_x, text_y),
                       font, font_scale, color, thickness)
        
        # Info debug en mode debug cercles
        if self.debug_circles:
            debug_info = f"Debug: {len(self.last_detected_positions)} cercles | "
            if self.last_red_mask is not None:
                red_pixels = cv2.countNonZero(self.last_red_mask)
                debug_info += f"Pixels rouges: {red_pixels} | "
            debug_info += f"Confiance: {self.subtitle_confidence:.2f}"
            
            cv2.putText(overlay_frame, debug_info, (10, subtitle_y + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Afficher le masque rouge dans un coin (miniature)
            if self.last_red_mask is not None:
                mask_small = cv2.resize(self.last_red_mask, (150, 100))
                mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
                
                # Placer dans le coin supérieur droit
                mask_x = width - 160
                mask_y = 10
                
                # Fond noir pour la miniature
                cv2.rectangle(overlay_frame, (mask_x-5, mask_y-5), (mask_x+155, mask_y+105), (0, 0, 0), -1)
                overlay_frame[mask_y:mask_y+100, mask_x:mask_x+150] = mask_colored
                
                # Titre de la miniature
                cv2.putText(overlay_frame, "Masque Rouge", (mask_x, mask_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Info en haut à gauche (déplacé car miniature à droite)
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Proc: {sum(self.processing_times)/len(self.processing_times)*1000:.1f}ms" if self.processing_times else "Proc: 0ms"
        ]
        
        # Statut du redressement
        if self.corner_detection_enabled:
            if self.screen_corners is not None:
                info_lines.append("🔧 Redressé")
            else:
                info_lines.append("🔍 Recherche écran...")
        else:
            info_lines.append("🔧 Off")
        
        # Modes debug
        debug_modes = []
        if self.debug_circles:
            debug_modes.append("🔴 Cercles")
        if self.debug_mode:
            debug_modes.append("🔧 Coins")
        
        if debug_modes:
            info_lines.append("Debug: " + " ".join(debug_modes))
        
        for i, line in enumerate(info_lines):
            cv2.putText(overlay_frame, line, (10, 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Indicateur de détection (plus gros en mode debug)
        indicator_size = 15 if self.debug_circles else 10
        current_time = time.time()
        if current_time - self.last_detection_time < 0.5:
            cv2.circle(overlay_frame, (width - 30, 30), indicator_size, (0, 255, 0), -1)
        else:
            cv2.circle(overlay_frame, (width - 30, 30), indicator_size, (0, 0, 255), -1)
        
        # Mode debug : dessiner les coins détectés
        if self.debug_mode:
            overlay_frame = self.draw_corner_detection(overlay_frame)
        
        return overlay_frame
    
    def process_frame_worker(self):
        """Worker thread pour le traitement des frames"""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Détecter et décoder
                positions = self.detect_circles_fast(frame)
                
                if positions:
                    binary_str = self.positions_to_binary(positions)
                    decoded_text = self.binary_to_text(binary_str)
                    
                    if decoded_text:
                        # Mettre à jour le buffer
                        self.update_subtitle_buffer(decoded_text)
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Erreur traitement: {e}")
                import traceback
                traceback.print_exc()
    
    def start_camera(self, camera_id: int = 0):
        """Démarre la capture caméra en temps réel"""
        print(f"📱 Démarrage caméra {camera_id}...")
        
        # Fixer les problèmes d'affichage Qt
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        
        cap = cv2.VideoCapture(camera_id)
        
        # Configuration optimale pour le temps réel
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la caméra {camera_id}")
        
        # Tester la capture
        ret, test_frame = cap.read()
        if not ret:
            raise ValueError(f"Impossible de lire depuis la caméra {camera_id}")
        
        print(f"✅ Caméra initialisée: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # Démarrer le thread de traitement
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_frame_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("🚀 Décodeur temps réel actif!")
        print("📝 Les sous-titres apparaîtront en bas de l'écran")
        print("🔧 Redressement automatique activé")
        print("⌨️  Commandes:")
        print("   'q' : Quitter")
        print("   'c' : Clear buffer")
        print("   's' : Screenshot")
        print("   'r' : Toggle redressement")
        print("   'd' : Toggle debug coins")
        print("   'v' : Toggle debug cercles (VISIBILITÉ)")
        print("   'h' : Aide")
        
        # Créer une seule fenêtre
        window_name = 'Decodeur Sous-titres Temps Reel'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Impossible de lire la frame")
                    break
                
                frame_count += 1
                
                # Ajouter la frame à la queue de traitement (non-bloquant)
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait(frame.copy())
                except:
                    pass
                
                # Dessiner l'overlay
                display_frame = self.draw_overlay(frame)
                
                # Afficher dans UNE SEULE fenêtre
                cv2.imshow(window_name, display_frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.current_subtitle = ""
                    self.detection_buffer.clear()
                    print("🧹 Buffer effacé")
                elif key == ord('s'):
                    screenshot_name = f"subtitle_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, display_frame)
                    print(f"📸 Screenshot sauvé: {screenshot_name}")
                elif key == ord('r'):
                    self.corner_detection_enabled = not self.corner_detection_enabled
                    if not self.corner_detection_enabled:
                        self.perspective_matrix = None
                        self.screen_corners = None
                        self.corner_buffer.clear()
                    status = "ON" if self.corner_detection_enabled else "OFF"
                    print(f"🔧 Redressement: {status}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    status = "ON" if self.debug_mode else "OFF"
                    print(f"🐛 Debug coins: {status}")
                elif key == ord('v'):
                    self.debug_circles = not self.debug_circles
                    status = "ON" if self.debug_circles else "OFF"
                    print(f"🔴 Debug cercles: {status}")
                    if self.debug_circles:
                        print("   🎯 Mode debug cercles activé:")
                        print("      • Grille visible")
                        print("      • Cercles détectés en VERT")
                        print("      • Cercles rejetés en ROUGE/BLEU")
                        print("      • Statistiques affichées")
                        print("      • Masque rouge en miniature")
                elif key == ord('h'):
                    print("\n" + "="*50)
                    print("🎯 AIDE - DÉCODEUR TEMPS RÉEL")
                    print("="*50)
                    print("⌨️  CONTRÔLES:")
                    print("   'q' : Quitter le programme")
                    print("   'c' : Clear buffer (effacer sous-titres)")
                    print("   's' : Screenshot avec overlay")
                    print("   'r' : Toggle redressement perspective")
                    print("   'd' : Toggle debug coins de l'écran")
                    print("   'v' : Toggle debug cercles rouges")
                    print("   'h' : Afficher cette aide")
                    print("\n🔴 DEBUG CERCLES:")
                    print("   • VERT : Cercles valides détectés")
                    print("   • ROUGE : Cercles trop petits")
                    print("   • BLEU : Cercles trop grands")
                    print("   • Grille : Montre la subdivision")
                    print("   • Miniature : Masque de détection rouge")
                    print("\n🔧 REDRESSEMENT:")
                    print("   • Auto-détecte les bords de l'écran")
                    print("   • Corrige la perspective automatiquement")
                    print("   • Peut être désactivé si problématique")
                    print("="*50)
                
                # Limiter à 30 FPS maximum
                if frame_count % 30 == 0:
                    time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n⚡ Arrêt par Ctrl+C")
        
        finally:
            # Nettoyage
            print("🔄 Nettoyage...")
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2)
            cap.release()
            cv2.destroyAllWindows()
            time.sleep(0.5)
            print("👋 Décodeur arrêté")

    def start_video(self, video_path: str):
        """Démarre la lecture d'une vidéo avec overlay"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo {video_path}")

        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_frame_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        window_name = 'Decodeur Sous-titres Temps Reel'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame.copy())
            except:
                pass

            display_frame = self.draw_overlay(frame)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(30) & 0xFF  # 30ms entre frames pour vidéo
            if key == ord('q'):
                break

        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(description='Décodeur temps réel pour smartphone')
    parser.add_argument('--mapping', required=True, help='Fichier de mapping JSON')
    parser.add_argument('--camera', type=int, default=0, help='ID de la caméra (défaut: 0)')
    
    args = parser.parse_args()
    
    try:
        decoder = RealTimeSubtitleDecoder(args.mapping)
        decoder.start_video("video_16x16.mp4")

        #decoder.start_camera(args.camera)
        
    except KeyboardInterrupt:
        print("\n⚡ Arrêt par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()