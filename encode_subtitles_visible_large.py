#!/usr/bin/env python3
"""
Version avec 4 grilles configurables avec disques blancs
"""

import cv2
import numpy as np
import re
import json
from typing import List, Tuple, Dict
import argparse
import os

class SubtitleEncoderVisibleLarge:
    def __init__(self, grid_size: Tuple[int, int] = (16, 16), 
                 camouflage_level: int = 0, 
                 local_radius: int = 0):
        """
        Encodeur avec 4 grilles dans les coins (grilles 16x16 chacune)
        
        Args:
            grid_size: Taille de chaque grille (width, height)
            camouflage_level: 0-100, 0=blanc pur, 100=couleur locale
            local_radius: Rayon de la zone locale (0=moyenne de toute la frame)
        """
        self.grid_width, self.grid_height = grid_size
        self.grid_size = grid_size
        self.point_size = 6  # Disques adaptés pour grilles 16x16
        self.point_intensity = 255  # Blanc pur
        self.num_grids = 4  # 4 grilles dans les coins
        
        # Nouveaux paramètres de camouflage
        self.camouflage_level = max(0, min(100, camouflage_level))  # Clamp entre 0-100
        self.local_radius = max(0, local_radius)
        
        # Offsets pour rendre chaque grille différente visuellement
        self.grid_offsets = [0, 5, 10, 15]  # Grille 0: pas d'offset, 1: +5, 2: +10, 3: +15
        # Positions relatives des 4 grilles (top-left, top-right, bottom-left, bottom-right)
        self.grid_positions = [
            (0.002, 0.02),    # Haut gauche
            (0.52, 0.02),    # Haut droite  
            (0.002, 0.52),    # Bas gauche
            (0.52, 0.52)     # Bas droite
        ]
        
    def parse_srt(self, srt_file: str) -> List[Dict]:
        """Parse un fichier SRT"""
        subtitles = []
        
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parser simple pour SRT
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                index = int(lines[0])
                time_line = lines[1]
                text = '\n'.join(lines[2:])
                
                # Parser le timing
                start_time, end_time = time_line.split(' --> ')
                start_seconds = self._time_to_seconds(start_time)
                end_seconds = self._time_to_seconds(end_time)
                
                subtitles.append({
                    'index': index,
                    'start_time': start_seconds,
                    'end_time': end_seconds,
                    'text': text
                })
        
        return subtitles
        
    def _time_to_seconds(self, time_str: str) -> float:
        """Convertit timestamp en secondes"""
        time_str = time_str.replace(',', '.')
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def text_to_binary(self, text: str) -> str:
        """Convertit texte en binaire"""
        binary_str = ""
        for char in text:
            binary_char = format(ord(char), '08b')
            binary_str += binary_char
        return binary_str
    
    def binary_to_grid_positions(self, binary_str: str) -> List[Tuple[int, int, int]]:
        """Convertit binaire en positions avec OFFSETS différents pour chaque grille"""
        positions = []
        positions_per_grid = self.grid_width * self.grid_height
        
        # Limiter la longueur binaire à UNE SEULE grille (même contenu logique sur les 4)
        limited_binary = binary_str[:positions_per_grid]
        
        # Générer les positions pour chacune des 4 grilles avec offset
        for grid_id in range(self.num_grids):
            offset = self.grid_offsets[grid_id]
            
            for i, bit in enumerate(limited_binary):
                if bit == '1':
                    # Appliquer l'offset avec bouclage circulaire
                    shifted_index = (i + offset) % positions_per_grid
                    x = shifted_index % self.grid_width
                    y = shifted_index // self.grid_width
                    positions.append((x, y, grid_id))
        
        return positions
    
    def get_local_color(self, frame: np.ndarray, center_x: int, center_y: int, 
                       frame_avg_color: np.ndarray) -> Tuple[int, int, int]:
        """
        Calcule la couleur adaptative basée sur le camouflage_level
        
        Args:
            frame: Image source
            center_x, center_y: Position du centre du cercle
            frame_avg_color: Couleur moyenne de toute la frame (fallback)
            
        Returns:
            Tuple (B, G, R) de la couleur à utiliser
        """
        if self.camouflage_level == 0:
            # Pas de camouflage : blanc pur
            return (255, 255, 255)
        
        height, width = frame.shape[:2]
        
        if self.local_radius == 0:
            # Utiliser la moyenne de toute la frame
            local_color = frame_avg_color
        else:
            # Extraire la région locale
            y1 = max(0, center_y - self.local_radius)
            y2 = min(height, center_y + self.local_radius)
            x1 = max(0, center_x - self.local_radius)
            x2 = min(width, center_x + self.local_radius)
            
            local_region = frame[y1:y2, x1:x2]
            
            if local_region.size == 0:
                local_color = frame_avg_color
            else:
                local_color = np.mean(local_region, axis=(0, 1))
        
        # Interpolation linéaire entre blanc (255, 255, 255) et couleur locale
        white = np.array([255, 255, 255], dtype=np.float32)
        blend_factor = self.camouflage_level / 100.0
        
        blended_color = white * (1 - blend_factor) + local_color * blend_factor
        
        return tuple(int(c) for c in blended_color)
    
    def add_visible_points(self, frame: np.ndarray, positions: List[Tuple[int, int, int]], 
                          frame_width: int, frame_height: int) -> np.ndarray:
        """Ajoute des disques adaptatifs sur 4 grilles"""
        modified_frame = frame.copy()
        
        # Précalculer la couleur moyenne de toute la frame si besoin
        frame_avg_color = None
        if self.camouflage_level > 0:
            frame_avg_color = np.mean(frame, axis=(0, 1))
        
        # Taille de chaque grille (40% de l'écran pour éviter les chevauchements)
        grid_pixel_width = int(frame_width * 0.48)
        grid_pixel_height = int(frame_height * 0.48)
        
        cell_width = grid_pixel_width // self.grid_width
        cell_height = grid_pixel_height // self.grid_height
        
        for x, y, grid_id in positions:
            # Calculer la position de base de la grille
            grid_x_offset = int(self.grid_positions[grid_id][0] * frame_width)
            grid_y_offset = int(self.grid_positions[grid_id][1] * frame_height)
            
            # Position du centre du disque
            center_x = grid_x_offset + x * cell_width + cell_width // 2
            center_y = grid_y_offset + y * cell_height + cell_height // 2
            
            # Déterminer la couleur du point
            point_color = self.get_local_color(frame, center_x, center_y, frame_avg_color)
            
            # Disque avec anti-aliasing
            cv2.circle(modified_frame, (center_x, center_y), self.point_size, 
                      point_color, -1, cv2.LINE_AA)
        
        return modified_frame
    
    def add_red_borders(self, frame: np.ndarray) -> np.ndarray:
        """Ajoute des contours rouges aux coins pour aider la détection du décodeur"""
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
    
    def encode_video(self, video_path: str, srt_path: str, output_path: str):
        """Encode la vidéo"""
        print(f"Parsing des sous-titres avec {self.num_grids} grilles de {self.grid_width}×{self.grid_height}...")
        print(f"Camouflage: {self.camouflage_level}% (rayon local: {self.local_radius}px)")
        
        subtitles = self.parse_srt(srt_path)
        print(f"Trouvé {len(subtitles)} sous-titres")
        
        # Analyser la capacité (même contenu sur les 4 grilles)
        max_chars = (self.grid_width * self.grid_height) // 8
        print(f"Capacité par grille (×{self.num_grids} pour redondance): {max_chars} caractères max")
        
        for subtitle in subtitles:
            if len(subtitle['text']) > max_chars:
                print(f"ATTENTION: Sous-titre trop long: '{subtitle['text'][:50]}...' ({len(subtitle['text'])} chars > {max_chars})")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Propriétés vidéo: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        mapping_data = {
            'grid_size': self.grid_size,
            'num_grids': self.num_grids,
            'grid_positions': self.grid_positions,
            'grid_offsets': self.grid_offsets,
            'point_size': self.point_size,
            'point_intensity': self.point_intensity,
            'camouflage_level': self.camouflage_level,
            'local_radius': self.local_radius,
            'encoding_type': 'adaptive_color_circles_4_grids',
            'video_properties': {
                'fps': fps,
                'width': width,
                'height': height
            },
            'subtitles': []
        }
        
        frame_count = 0
        
        print("Encodage en cours...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            subtitle_positions = []
            current_subtitle_text = ""
            
            for subtitle in subtitles:
                if subtitle['start_time'] <= current_time <= subtitle['end_time']:
                    binary_text = self.text_to_binary(subtitle['text'])
                    positions = self.binary_to_grid_positions(binary_text)
                    subtitle_positions.extend(positions)
                    current_subtitle_text = subtitle['text']
                    
                    if subtitle['index'] not in [s['index'] for s in mapping_data['subtitles']]:
                        mapping_data['subtitles'].append({
                            'index': subtitle['index'],
                            'start_time': subtitle['start_time'],
                            'end_time': subtitle['end_time'],
                            'text': subtitle['text'],
                            'binary': binary_text,
                            'positions': positions
                        })
                    break
            
            # TOUJOURS ajouter les contours rouges pour la détection de perspective
            frame = self.add_red_borders(frame)
            
            if subtitle_positions:
                frame = self.add_visible_points(frame, subtitle_positions, width, height)
                if frame_count % (fps * 2) == 0:
                    print(f"Frame {frame_count}/{total_frames} - Sous-titre: '{current_subtitle_text}' - {len(subtitle_positions)} points + contours rouges")
            elif frame_count % (fps * 2) == 0:
                print(f"Frame {frame_count}/{total_frames} - Pas de sous-titres + contours rouges")
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        mapping_path = output_path.replace('.mp4', '_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"Encodage terminé!")
        print(f"Vidéo: {output_path}")
        print(f"Mapping: {mapping_path}")

def main():
    parser = argparse.ArgumentParser(description='Encoder avec 4 grilles configurables')
    parser.add_argument('--video', required=True, help='Vidéo source')
    parser.add_argument('--srt', required=True, help='Fichier SRT')
    parser.add_argument('--output', required=True, help='Vidéo de sortie')
    parser.add_argument('--grid-width', type=int, default=16, help='Largeur de chaque grille (4 grilles au total)')
    parser.add_argument('--grid-height', type=int, default=16, help='Hauteur de chaque grille (4 grilles au total)')
    parser.add_argument('--point-size', type=int, default=6, help='Taille des cercles en pixels')
    parser.add_argument('--camouflage', type=int, default=0, help='Niveau de camouflage 0-100 (0=blanc, 100=couleur locale)')
    parser.add_argument('--local-radius', type=int, default=0, help='Rayon zone locale (0=moyenne frame entière)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Erreur: {args.video} n'existe pas")
        return
    
    if not os.path.exists(args.srt):
        print(f"Erreur: {args.srt} n'existe pas")
        return
    
    encoder = SubtitleEncoderVisibleLarge(
        grid_size=(args.grid_width, args.grid_height),
        camouflage_level=args.camouflage,
        local_radius=args.local_radius
    )
    
    # Configurer la taille des points si spécifiée
    if hasattr(args, 'point_size') and args.point_size:
        encoder.point_size = args.point_size
        print(f"Taille des cercles: {args.point_size}px")
    
    try:
        encoder.encode_video(args.video, args.srt, args.output)
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()