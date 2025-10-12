#!/usr/bin/env python3
"""
Décodeur de sous-titres SIMPLE - Juste ce qu'il faut !
- Détecte les cercles blancs avec HoughCircles
- Vote majoritaire entre 4 grilles
- Affiche les sous-titres en overlay
- PAS DE REDRESSEMENT DE MERDE !
"""

import cv2
import numpy as np
import json
import argparse
import time
import os
from typing import List, Tuple
from collections import deque

class SimpleSubtitleDecoder:
    def __init__(self, grid_width: int = 16, grid_height: int = 16, point_size: int = 6):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.point_size = point_size
        self.num_grids = 4
        
        # Offsets pour chaque grille (identiques à l'encodeur)
        self.grid_offsets = [0, 5, 10, 15]  # Grille 0: pas d'offset, 1: +5, 2: +10, 3: +15
        
        # Positions des 4 grilles (identiques à l'encodeur)
        self.grid_positions = [
            (0.002, 0.02),    # Haut gauche
            (0.52, 0.02),    # Haut droite  
            (0.002, 0.52),    # Bas gauche
            (0.52, 0.52)     # Bas droite
        ]
        
        # Buffer pour lisser les détections
        self.detection_buffer = deque(maxlen=10)
        self.current_subtitle = ""
        
        # Stockage des sous-titres décodés
        self.decoded_subtitles = []
        
        # Debug
        self.debug_mode = False
        
        # Système de redressement pour caméra
        self.perspective_correction = False
        self.debug_perspective = False
        self.perspective_matrix = None
        self.screen_corners = None
        self.corner_buffer = deque(maxlen=5)
        
        #print(f"🎯 Décodeur SIMPLE initialisé")
        #print(f"📱 Grille: {self.grid_width}×{self.grid_height}")
        #print(f"🔴 Taille points: {self.point_size}")
        #print(f"⚙️ Offsets: {self.grid_offsets}")
    
    def load_mapping_config(self, mapping_file: str, override_point_size: bool = True):
        """Charge la configuration depuis un fichier de mapping JSON"""
        try:
            import json
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            # Charger les offsets si disponibles
            if 'grid_offsets' in mapping_data:
                self.grid_offsets = mapping_data['grid_offsets']
                #print(f"✅ Offsets chargés depuis {mapping_file}: {self.grid_offsets}")
            
            # Charger les autres paramètres si disponibles
            if 'grid_positions' in mapping_data:
                self.grid_positions = mapping_data['grid_positions']
                #print(f"✅ Positions grilles chargées: {len(self.grid_positions)} grilles")
            
            # Ne charger point_size que si autorisé
            if 'point_size' in mapping_data and override_point_size:
                self.point_size = mapping_data['point_size']
                #print(f"✅ Taille points chargée: {self.point_size}px")
                
        except Exception as e:
            print(f"⚠️ Impossible de charger {mapping_file}: {e}")
            print(f"⚠️ Utilisation des paramètres par défaut")
    
    def export_subtitles_to_json(self, filename="decoded_subtitles_camera.json"):
        """Exporte tous les sous-titres décodés vers un fichier JSON"""
        output_data = {"subtitles": []}
        
        for i, text in enumerate(self.decoded_subtitles):
            output_data["subtitles"].append({
                "text": text,
                "timestamp": time.time(),
                "confidence": 1.0,  # Placeholder pour compatibilité
                "index": i
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        #print(f"📁 Exporté {len(self.decoded_subtitles)} sous-titres vers {filename}")
    
    def detect_white_circles(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Détection de cercles par forme (peu importe la couleur)"""
        
        # AJOUTER DES CONTOURS ROUGES AUX COINS pour aider la détection
        frame_with_borders = self.add_red_borders(frame)
        
        # Redressement si activé (pour caméra)
        working_frame = frame_with_borders
        if self.perspective_correction:
            corrected = self.apply_perspective_correction(frame_with_borders)
            if corrected is not None:
                working_frame = corrected
        
        frame_height, frame_width = working_frame.shape[:2]
        
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
        
        # Floutage gaussien pour réduire le bruit et améliorer la détection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Détection de contours avec Canny pour aider HoughCircles
        edges = cv2.Canny(blurred, 30, 100)
        
        # Analyser chaque grille
        all_grid_detections = []
        
        for grid_id, (rel_x, rel_y) in enumerate(self.grid_positions):
            # Zone de la grille (48% de l'écran)
            grid_pixel_width = int(frame_width * 0.48)
            grid_pixel_height = int(frame_height * 0.48)
            
            grid_x_offset = int(rel_x * frame_width)
            grid_y_offset = int(rel_y * frame_height)
            
            # Extraire la région
            if (grid_y_offset + grid_pixel_height <= frame_height and 
                grid_x_offset + grid_pixel_width <= frame_width):
                
                grid_blurred = blurred[grid_y_offset:grid_y_offset + grid_pixel_height,
                                      grid_x_offset:grid_x_offset + grid_pixel_width]
            else:
                all_grid_detections.append([])
                continue
            
            # Paramètres HoughCircles adaptatifs avec cas spéciaux
            if self.point_size == 1:
                # Paramètres ultra-extrêmes pour les cercles pixels (1px)
                min_radius = 1
                max_radius = 3
                min_dist = 1
                param1 = 10  # Sensibilité absolue
                param2 = 2   # Seuil critique minimal
                #print(f"🔥🔥🔥 Paramètres ULTRA-EXTRÊMES pour point_size=1 (limite absolue!)")
            elif self.point_size == 2:
                # Paramètres ultra-spéciaux pour les cercles minuscules (2px)
                min_radius = 1
                max_radius = 4
                min_dist = 1
                param1 = 15  # Extrêmement sensible
                param2 = 3   # Seuil très très bas
                #print(f"🔥🔥 Paramètres ULTRA-spéciaux pour point_size=2")
            elif self.point_size == 3:
                # Paramètres spéciaux pour les très petits cercles (3px)
                min_radius = 1
                max_radius = 6
                min_dist = 2
                param1 = 20  # Très sensible
                param2 = 5   # Très sensible
                #print(f"🔥 Paramètres spéciaux pour point_size=3")
            elif self.point_size == 7:
                # Paramètres spéciaux pour les cercles moyens-gros (7px)
                min_radius = 6
                max_radius = 8
                min_dist = 6
                param1 = 45  # Moins sensible pour éviter le bruit
                param2 = 13  # Seuil plus élevé
                #print(f"⭐ Paramètres spéciaux pour point_size=7")
            elif self.point_size == 6:
                # Paramètres spéciaux pour les cercles moyens (6px)
                min_radius = 5
                max_radius = 8
                min_dist = 5
                param1 = 100  # Sensibilité modérée
                param2 = 10  # Seuil modéré
                #print(f"🌟 Paramètres spéciaux pour point_size=6")
            else:
                # Paramètres normaux pour autres tailles
                min_radius = max(1, self.point_size - 3)
                max_radius = self.point_size + 5
                min_dist = max(4, self.point_size - 1)
                param1 = 30
                param2 = 8
            
            # DEBUG: Afficher les paramètres utilisés
            if self.debug_mode:
                print(f"🔍 Grid {grid_id}: point_size={self.point_size}, min_r={min_radius}, max_r={max_radius}, dist={min_dist}, p1={param1}, p2={param2}")
            
            # DÉTECTION DE CERCLES PAR FORME (sans filtrage de couleur)
            circles = cv2.HoughCircles(
                grid_blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min_dist,
                param1=param1,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            # DEBUG: Afficher le résultat de la détection
            if self.debug_mode:
                circles_count = len(circles[0]) if circles is not None else 0
                #print(f"🎯 Grid {grid_id}: {circles_count} cercles détectés")
                if circles_count == 0:
                    print(f"❌ Grid {grid_id}: Aucun cercle détecté avec ces paramètres")
            
            grid_detections = []
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Taille des cellules
                cell_width = grid_pixel_width / self.grid_width
                cell_height = grid_pixel_height / self.grid_height
                
                for (x, y, r) in circles:
                    # Convertir en position de grille
                    grid_x = int(x // cell_width)
                    grid_y = int(y // cell_height)
                    
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        grid_detections.append((grid_x, grid_y))
                        
                        # Debug visuel
                        if self.debug_mode:
                            abs_x = x + grid_x_offset
                            abs_y = y + grid_y_offset
                            cv2.circle(working_frame, (abs_x, abs_y), r, (0, 255, 0), 2)
                            cv2.putText(working_frame, f"G{grid_id}({grid_x},{grid_y})", 
                                       (abs_x + 8, abs_y - 8), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            all_grid_detections.append(grid_detections)
            
            # Debug grille avec paramètres adaptatifs
            if self.debug_mode:
                grid_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][grid_id]
                cv2.rectangle(working_frame, 
                            (grid_x_offset, grid_y_offset),
                            (grid_x_offset + grid_pixel_width, grid_y_offset + grid_pixel_height),
                            grid_color, 2)
                cv2.putText(working_frame, f"Grid {grid_id} (r:{min_radius}-{max_radius}, d:{min_dist}, p:{param1},{param2})", 
                           (grid_x_offset + 5, grid_y_offset + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, grid_color, 1)
        
        # VOTE MAJORITAIRE (2/4 grilles minimum)
        voted_positions = self.vote_majority(all_grid_detections)
        
        if self.debug_mode:
            total_detections = sum(len(detections) for detections in all_grid_detections)
            #print(f"📊 TOTAL: {total_detections} détections brutes → {len(voted_positions)} positions validées par vote")
            
            # Debug détaillé par grille
            for i, detections in enumerate(all_grid_detections):
                print(f"   Grid {i}: {len(detections)} détections")

            if total_detections == 0:
                print("❌ AUCUN CERCLE DÉTECTÉ DANS AUCUNE GRILLE")
        
        # Stocker la frame de travail pour l'affichage
        self.debug_frame = working_frame if self.debug_mode else None
        
        return voted_positions
    
    def vote_majority(self, all_grid_detections: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """Vote majoritaire avec correction des offsets - au moins 2 grilles d'accord"""
        position_votes = {}
        positions_per_grid = self.grid_width * self.grid_height
        
        for grid_id, grid_detections in enumerate(all_grid_detections):
            offset = self.grid_offsets[grid_id]
            
            for pos in grid_detections:
                x, y = pos
                # Convertir en index linéaire
                detected_index = y * self.grid_width + x
                # Appliquer la dé-offset (inverse de l'encodeur)
                original_index = (detected_index - offset) % positions_per_grid
                # Reconvertir en coordonnées
                original_x = original_index % self.grid_width
                original_y = original_index // self.grid_width
                original_pos = (original_x, original_y)
                
                position_votes[original_pos] = position_votes.get(original_pos, 0) + 1
        
        # Garder les positions avec au moins 2 votes
        voted_positions = []
        for pos, votes in position_votes.items():
            if votes >= 2:
                voted_positions.append(pos)
        
        return voted_positions
    
    def positions_to_binary(self, positions: List[Tuple[int, int]]) -> str:
        """Convertit les positions en binaire"""
        binary_str = ['0'] * (self.grid_width * self.grid_height)
        for x, y in positions:
            index = y * self.grid_width + x
            if index < len(binary_str):
                binary_str[index] = '1'
        return ''.join(binary_str)
    
    def binary_to_text(self, binary_str: str) -> str:
        """Convertit le binaire en texte"""
        try:
            # Découper en chunks de 8 bits
            chars = []
            for i in range(0, len(binary_str), 8):
                byte = binary_str[i:i+8]
                if len(byte) == 8:
                    char_code = int(byte, 2)
                    if 32 <= char_code <= 126:  # ASCII imprimable
                        chars.append(chr(char_code))
            
            text = ''.join(chars).strip()
            return text if text else ""
        except:
            return ""
    
    def detect_screen_corners(self, frame: np.ndarray) -> np.ndarray:
        """Détecte les coins de l'écran pour le redressement"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Pré-traitement pour améliorer la détection
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Détection de contours multi-seuils
        edges1 = cv2.Canny(gray, 20, 60, apertureSize=3)
        edges2 = cv2.Canny(gray, 40, 120, apertureSize=3)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphologie pour connecter les contours
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges = cv2.dilate(edges, kernel_dilate, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        
        # Trouver les contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filtrer les contours par taille
        frame_area = frame.shape[0] * frame.shape[1]
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.1 * frame_area < area < 0.9 * frame_area:
                # Vérifier le ratio d'aspect
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if 1.0 < aspect_ratio < 3.0:  # Ratio d'écran raisonnable
                        valid_contours.append((contour, area))
        
        # Trier par aire décroissante
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        
        # Essayer les 3 plus grands contours
        for contour, _ in valid_contours[:3]:
            perimeter = cv2.arcLength(contour, True)
            for epsilon_factor in [0.015, 0.02, 0.025, 0.03, 0.04]:
                epsilon = epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    corners = approx.reshape(4, 2).astype(np.float32)
                    
                    # Vérifier que c'est un quadrilatère valide
                    if self.is_valid_quadrilateral(corners):
                        ordered_corners = self.order_corners(corners)
                        return ordered_corners
        
        return None
    
    def is_valid_quadrilateral(self, corners: np.ndarray) -> bool:
        """Vérifie si le quadrilatère est valide pour un écran"""
        # Calculer les longueurs des côtés
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            side_length = np.linalg.norm(p2 - p1)
            sides.append(side_length)
        
        # Ratio des côtés pas trop déséquilibré
        min_side = min(sides)
        max_side = max(sides)
        if max_side / min_side > 3.0:
            return False
        
        # Vérifier les angles (approximativement droits)
        angles = []
        for i in range(4):
            p1 = corners[(i - 1) % 4]
            p2 = corners[i]
            p3 = corners[(i + 1) % 4]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
        
        # Angles entre 70° et 110°
        for angle in angles:
            if angle < 70 or angle > 110:
                return False
        
        return True
    
    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Ordonne les coins : top-left, top-right, bottom-right, bottom-left"""
        sums = corners.sum(axis=1)
        diffs = np.diff(corners, axis=1)
        
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(sums)]      # top-left
        ordered[2] = corners[np.argmax(sums)]      # bottom-right
        ordered[1] = corners[np.argmin(diffs)]     # top-right
        ordered[3] = corners[np.argmax(diffs)]     # bottom-left
        
        return ordered
    
    def apply_perspective_correction(self, frame: np.ndarray) -> np.ndarray:
        """Applique la correction de perspective"""
        # Détecter les coins
        corners = self.detect_screen_corners(frame)
        
        if corners is not None:
            # Ajouter au buffer pour stabiliser
            self.corner_buffer.append(corners)
            
            if len(self.corner_buffer) >= 2:
                # Moyenne des dernières détections
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
                
                # Matrice de transformation
                dst_corners = np.array([
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1]
                ], dtype=np.float32)
                
                self.perspective_matrix = cv2.getPerspectiveTransform(avg_corners.astype(np.float32), dst_corners)
                
                try:
                    corrected_frame = cv2.warpPerspective(frame, self.perspective_matrix, (target_width, target_height))
                    self.screen_corners = avg_corners
                    
                    # Debug du redressement
                    if self.debug_perspective:
                        debug_frame = frame.copy()
                        # Dessiner les coins détectés
                        for i, corner in enumerate(avg_corners.astype(int)):
                            cv2.circle(debug_frame, tuple(corner), 8, (0, 255, 0), -1)
                            cv2.putText(debug_frame, str(i), tuple(corner + 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Dessiner le contour
                        cv2.polylines(debug_frame, [avg_corners.astype(int)], True, (255, 255, 0), 3)
                        
                        # Afficher dans une fenêtre séparée
                        cv2.imshow('Perspective Debug', debug_frame)
                    
                    return corrected_frame
                except Exception as e:
                    if self.debug_perspective:
                        print(f"Erreur transformation: {e}")
        
        # Utiliser la dernière matrice stable si disponible
        elif self.perspective_matrix is not None:
            try:
                h, w = frame.shape[:2]
                corners_3d = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)
                transformed_corners = cv2.perspectiveTransform(corners_3d, self.perspective_matrix)
                
                target_width = int(np.max(transformed_corners[0, :, 0]))
                target_height = int(np.max(transformed_corners[0, :, 1]))
                
                corrected_frame = cv2.warpPerspective(frame, self.perspective_matrix, (target_width, target_height))
                return corrected_frame
            except:
                self.perspective_matrix = None
        
        return None
    
    def add_red_borders(self, frame: np.ndarray) -> np.ndarray:
        """Ajoute des contours rouges aux coins pour aider la détection"""
        bordered_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Épaisseur des contours
        thickness = 8
        corner_size = min(width, height) // 8  # Taille des coins (1/8 de l'écran)
        
        # Couleur rouge vif
        red_color = (0, 0, 255)  # BGR
        
        # COIN HAUT-GAUCHE
        cv2.rectangle(bordered_frame, (0, 0), (corner_size, thickness), red_color, -1)  # Horizontal haut
        cv2.rectangle(bordered_frame, (0, 0), (thickness, corner_size), red_color, -1)  # Vertical gauche
        
        # COIN HAUT-DROITE  
        cv2.rectangle(bordered_frame, (width - corner_size, 0), (width, thickness), red_color, -1)  # Horizontal haut
        cv2.rectangle(bordered_frame, (width - thickness, 0), (width, corner_size), red_color, -1)  # Vertical droite
        
        # COIN BAS-GAUCHE
        cv2.rectangle(bordered_frame, (0, height - thickness), (corner_size, height), red_color, -1)  # Horizontal bas
        cv2.rectangle(bordered_frame, (0, height - corner_size), (thickness, height), red_color, -1)  # Vertical gauche
        
        # COIN BAS-DROITE
        cv2.rectangle(bordered_frame, (width - corner_size, height - thickness), (width, height), red_color, -1)  # Horizontal bas
        cv2.rectangle(bordered_frame, (width - thickness, height - corner_size), (width, height), red_color, -1)  # Vertical droite
        
        return bordered_frame
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Dessine l'overlay avec les sous-titres"""
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Zone de sous-titres en bas
        if self.current_subtitle:
            # Fond noir semi-transparent
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, height - 100), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Texte des sous-titres
            font_scale = min(width / 800, 2.0)
            cv2.putText(display_frame, self.current_subtitle, (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # TOUJOURS afficher les statuts (pas seulement en debug)
        height, width = display_frame.shape[:2]
        
        # Fond semi-transparent pour les statuts
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Statut redressement
        redress_status = "ON" if self.perspective_correction else "OFF"
        redress_color = (0, 255, 0) if self.perspective_correction else (0, 0, 255)
        cv2.putText(display_frame, f"Redressement: {redress_status}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, redress_color, 2)
        
        # Statut debug redressement
        debug_redress_status = "ON" if self.debug_perspective else "OFF"
        debug_redress_color = (0, 255, 255) if self.debug_perspective else (128, 128, 128)
        cv2.putText(display_frame, f"Debug redresse: {debug_redress_status}", (15, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, debug_redress_color, 1)
        
        # Statut détection des coins
        if self.perspective_correction:
            if self.screen_corners is not None:
                corners_status = "COINS DETECTES"
                corners_color = (0, 255, 0)
            else:
                corners_status = "RECHERCHE COINS..."
                corners_color = (0, 255, 255)
            cv2.putText(display_frame, corners_status, (15, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, corners_color, 1)
        
        # Debug cercles info
        if self.debug_mode:
            cv2.putText(display_frame, f"Subtitle: {self.current_subtitle}", (15, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Utiliser la frame de debug si disponible (avec cercles détectés)
        if hasattr(self, 'debug_frame') and self.debug_frame is not None and self.debug_mode:
            # Redimensionner si nécessaire
            if self.debug_frame.shape != display_frame.shape:
                debug_resized = cv2.resize(self.debug_frame, (display_frame.shape[1], display_frame.shape[0]))
                return debug_resized
            else:
                return self.debug_frame
        
        return display_frame
    
    def process_video(self, video_path: str):
        """Traite une vidéo"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            #print(f"❌ Impossible d'ouvrir: {video_path}")
            return
        
        #print(f"🎥 Traitement de: {video_path}")
        
        cv2.namedWindow('Subtitles Decoder', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0  # Compteur de frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Ne traiter qu'une frame sur 5
            if frame_count % 5 == 0:
                # DÉTECTION
                positions = self.detect_white_circles(frame)
                
                if positions:
                    # Convertir en texte
                    binary_str = self.positions_to_binary(positions)
                    text = self.binary_to_text(binary_str)
                    
                    if text:
                        self.detection_buffer.append(text)
                        
                        # Prendre le texte le plus fréquent
                        if len(self.detection_buffer) >= 3:
                            from collections import Counter
                            most_common = Counter(self.detection_buffer).most_common(1)
                            if most_common:
                                new_subtitle = most_common[0][0]
                                if new_subtitle != self.current_subtitle:
                                    self.current_subtitle = new_subtitle
                                    self.decoded_subtitles.append(new_subtitle)
                                    #print(f"💾 Stocké: '{new_subtitle[:50]}...'")
            
            # AFFICHAGE (toujours afficher pour fluidité visuelle)
            display_frame = self.draw_overlay(frame)
            cv2.imshow('Subtitles Decoder', display_frame)
            
            # Contrôles
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                #print(f"🐛 Debug: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('c'):
                self.current_subtitle = ""
                self.detection_buffer.clear()
                #print("🧹 Cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        # S'assurer que toutes les fenêtres de debug sont fermées
        cv2.destroyWindow('Perspective Debug')
        
        # Exporter les sous-titres décodés
        if self.decoded_subtitles:
            self.export_subtitles_to_json()
        else:
            print("📝 Aucun sous-titre décodé à exporter")
    
    def process_camera(self, camera_id: int = 0):
        """Traite la caméra"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            #print(f"❌ Impossible d'ouvrir caméra: {camera_id}")
            return
        
        #print(f"📷 Caméra: {camera_id}")
        #print("🎮 Contrôles actifs:")
        #print("  'r': Activer/désactiver redressement perspective")
        #print("  'f': Debug redressement (fenêtre séparée avec coins)")
        #print("  'd': Debug cercles, 'c': clear, 'q': quit")
        #print("")
        #print("🔴 ASTUCE: Les contours rouges aux coins aident la détection !")
        #print("   Placez l'écran bien visible avec ses 4 coins dans la caméra")
        
        cv2.namedWindow('Subtitles Decoder', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0  # Compteur de frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Ne traiter qu'une frame sur 10
            if frame_count % 5 == 0:
                # DÉTECTION
                positions = self.detect_white_circles(frame)
                
                if positions:
                    binary_str = self.positions_to_binary(positions)
                    text = self.binary_to_text(binary_str)
                    
                    if text:
                        self.detection_buffer.append(text)
                        
                        if len(self.detection_buffer) >= 3:
                            from collections import Counter
                            most_common = Counter(self.detection_buffer).most_common(1)
                            if most_common:
                                new_subtitle = most_common[0][0]
                                if new_subtitle != self.current_subtitle:
                                    self.current_subtitle = new_subtitle
                                    self.decoded_subtitles.append(new_subtitle)
                                    #print(f"💾 Stocké: '{new_subtitle[:50]}...'")
            
            # AFFICHAGE (toujours afficher pour fluidité visuelle)
            display_frame = self.draw_overlay(frame)
            cv2.imshow('Subtitles Decoder', display_frame)
            
            # Contrôles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                #print(f"🐛 Debug cercles: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('c'):
                self.current_subtitle = ""
                self.detection_buffer.clear()
                #print("🧹 Cleared")
            elif key == ord('r'):
                self.perspective_correction = not self.perspective_correction
                if not self.perspective_correction:
                    self.perspective_matrix = None
                    self.screen_corners = None
                    self.corner_buffer.clear()
                    cv2.destroyWindow('Perspective Debug')
                status = "ACTIVÉ" if self.perspective_correction else "DÉSACTIVÉ"
                #print(f"🔄 Redressement: {status}")
                if self.perspective_correction:
                    print("   → Cherche les contours rouges aux coins de l'écran...")
            elif key == ord('f'):
                if self.perspective_correction:
                    self.debug_perspective = not self.debug_perspective
                    if not self.debug_perspective:
                        cv2.destroyWindow('Perspective Debug')
                    status = "ACTIVÉ" if self.debug_perspective else "DÉSACTIVÉ"
                    #print(f"🔍 Debug redressement: {status}")
                    if self.debug_perspective:
                        print("   → Fenêtre 'Perspective Debug' ouverte")
                else:
                    print("⚠️ Activez d'abord le redressement avec 'r'")
                    print("   → Les contours rouges aident à détecter l'écran")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Exporter les sous-titres décodés
        if self.decoded_subtitles:
            self.export_subtitles_to_json()
        else:
            print("📝 Aucun sous-titre décodé à exporter")

def main():
    """Fonction principale avec support caméra/vidéo"""
    parser = argparse.ArgumentParser(description='🎯 Décodeur de sous-titres avec redressement')
    parser.add_argument('source', nargs='?', default='0', 
                       help='Source: 0=caméra 0, 1=caméra 1, etc. ou fichier vidéo')
    parser.add_argument('--debug', action='store_true', help='Mode debug')
    parser.add_argument('--point-size', type=int, default=6, help='Taille des cercles en pixels')
    
    args = parser.parse_args()
    
    try:
        decoder = SimpleSubtitleDecoder(point_size=args.point_size)
        decoder.debug_mode = args.debug
        
        # Déterminer si c'est une caméra ou vidéo
        source = args.source
        
        if source.isdigit():
            # C'est un numéro de caméra
            camera_id = int(source)
            #print(f"📷 Mode caméra {camera_id}")
            #print("🎮 Contrôles:")
            #print("  'q': Quitter")
            #print("  'd': Debug cercles ON/OFF")
            #print("  'r': Redressement ON/OFF (détecte les coins)")
            #print("  'f': Debug redressement ON/OFF (fenêtre debug)")
            #print("  'c': Clear sous-titres")
            #print("")
            #print("🔴 Les contours ROUGES aux coins aident la détection !")
            decoder.process_camera(camera_id)
        else:
            # C'est un fichier vidéo
            #print(f"🎥 Mode vidéo: {source}")
            
            # Essayer de charger le fichier de mapping correspondant
            video_name = source.replace('.mp4', '')
            possible_mapping_files = [
                f"{video_name}_mapping.json",
                f"{video_name}_16x16_4grids_white_with_borders_mapping.json",
                "video_16x16_4grids_white_with_borders_mapping.json"
            ]
            
            mapping_loaded = False
            # Sauvegarder le point_size passé en argument pour qu'il ait la priorité
            original_point_size = decoder.point_size
            
            for mapping_file in possible_mapping_files:
                if os.path.exists(mapping_file):
                    #print(f"📂 Chargement des paramètres depuis: {mapping_file}")
                    decoder.load_mapping_config(mapping_file, override_point_size=False)
                    #print(f"🔧 Point size conservé: {original_point_size}px (argument)")
                    mapping_loaded = True
                    break
            
            if not mapping_loaded:
                print("⚠️ Aucun fichier de mapping trouvé, utilisation des paramètres par défaut")
            
            #print("🎮 Contrôles: 'q' quitter, 'd' debug, 'c' clear")
            #print("📺 Mode vidéo = PAS de redressement (inutile)")
            decoder.process_video(source)
            
    except Exception as e:
        #print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()