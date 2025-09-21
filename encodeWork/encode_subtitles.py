#!/usr/bin/env python3
"""
Script d'encodage des sous-titres dans une vidéo
Partie 1 du projet IHM - Encodage des sous-titres sous forme de points invisibles
"""

import cv2
import numpy as np
import re
import json
from typing import List, Tuple, Dict
import argparse
import os

class SubtitleEncoder:
    def __init__(self, grid_size: Tuple[int, int] = (10, 10)):
        """
        Initialise l'encodeur de sous-titres
        
        Args:
            grid_size: Taille de la grille virtuelle (largeur, hauteur)
        """
        self.grid_width, self.grid_height = grid_size
        self.grid_size = grid_size
        self.point_size = 2  # Taille des points en pixels
        self.point_intensity = 5  # Variation d'intensité des points
        
    def parse_srt(self, srt_file: str) -> List[Dict]:
        """
        Parse un fichier SRT et retourne une liste de sous-titres
        
        Args:
            srt_file: Chemin vers le fichier SRT
            
        Returns:
            Liste de dictionnaires contenant les sous-titres
        """
        subtitles = []
        
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern pour matcher les blocs de sous-titres
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            index, start_time, end_time, text = match
            text = text.strip().replace('\n', ' ')
            
            # Convertir les timestamps en secondes
            start_seconds = self._time_to_seconds(start_time)
            end_seconds = self._time_to_seconds(end_time)
            
            subtitles.append({
                'index': int(index),
                'start_time': start_seconds,
                'end_time': end_seconds,
                'text': text
            })
        
        return subtitles
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convertit un timestamp SRT en secondes"""
        time_str = time_str.replace(',', '.')
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def text_to_binary(self, text: str) -> str:
        """
        Convertit un texte en représentation binaire
        
        Args:
            text: Texte à convertir
            
        Returns:
            Chaîne binaire
        """
        binary_str = ""
        for char in text:
            # Convertir chaque caractère en binaire (8 bits)
            binary_char = format(ord(char), '08b')
            binary_str += binary_char
        return binary_str
    
    def binary_to_grid_positions(self, binary_str: str) -> List[Tuple[int, int]]:
        """
        Convertit une chaîne binaire en positions de grille
        
        Args:
            binary_str: Chaîne binaire
            
        Returns:
            Liste des positions (x, y) dans la grille
        """
        positions = []
        total_positions = self.grid_width * self.grid_height
        
        # Limiter la longueur binaire à la taille de la grille
        limited_binary = binary_str[:total_positions]
        
        for i, bit in enumerate(limited_binary):
            if bit == '1':
                x = i % self.grid_width
                y = i // self.grid_width
                positions.append((x, y))
        
        return positions
    
    def add_invisible_points(self, frame: np.ndarray, positions: List[Tuple[int, int]], 
                           frame_width: int, frame_height: int) -> np.ndarray:
        """
        Ajoute des points invisibles sur une frame
        
        Args:
            frame: Frame de la vidéo
            positions: Positions des points dans la grille
            frame_width: Largeur de la frame
            frame_height: Hauteur de la frame
            
        Returns:
            Frame modifiée avec les points
        """
        modified_frame = frame.copy()
        
        # Calculer la taille des cellules de la grille
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        for x, y in positions:
            # Calculer la position réelle du point
            center_x = x * cell_width + cell_width // 2
            center_y = y * cell_height + cell_height // 2
            
            # Ajouter un point subtil (variation de couleur très légère)
            for dy in range(-self.point_size, self.point_size + 1):
                for dx in range(-self.point_size, self.point_size + 1):
                    px = center_x + dx
                    py = center_y + dy
                    
                    if 0 <= px < frame_width and 0 <= py < frame_height:
                        # Modifier légèrement la couleur (ajouter une petite variation)
                        for channel in range(3):  # BGR
                            current_value = modified_frame[py, px, channel]
                            # Ajouter une variation très subtile
                            variation = self.point_intensity if (dx*dx + dy*dy) <= self.point_size*self.point_size else 0
                            new_value = min(255, max(0, current_value + variation))
                            modified_frame[py, px, channel] = new_value
        
        return modified_frame
    
    def encode_video(self, video_path: str, srt_path: str, output_path: str):
        """
        Encode les sous-titres dans la vidéo
        
        Args:
            video_path: Chemin vers la vidéo source
            srt_path: Chemin vers le fichier SRT
            output_path: Chemin vers la vidéo de sortie
        """
        print("Parsing des sous-titres...")
        subtitles = self.parse_srt(srt_path)
        print(f"Trouvé {len(subtitles)} sous-titres")
        
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        # Obtenir les propriétés de la vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Propriétés vidéo: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Configurer le writer vidéo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Créer le mapping de référence pour le décodage
        mapping_data = {
            'grid_size': self.grid_size,
            'point_size': self.point_size,
            'point_intensity': self.point_intensity,
            'video_properties': {
                'fps': fps,
                'width': width,
                'height': height
            },
            'subtitles': []
        }
        
        frame_count = 0
        current_subtitle_index = 0
        
        print("Encodage en cours...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Vérifier si on est dans une période de sous-titre
            subtitle_positions = []
            current_subtitle_text = ""
            
            for subtitle in subtitles:
                if subtitle['start_time'] <= current_time <= subtitle['end_time']:
                    # Convertir le texte en positions de grille
                    binary_text = self.text_to_binary(subtitle['text'])
                    positions = self.binary_to_grid_positions(binary_text)
                    subtitle_positions.extend(positions)
                    current_subtitle_text = subtitle['text']
                    
                    # Sauvegarder les données pour le mapping
                    if subtitle not in [s for s in mapping_data['subtitles'] if s['index'] == subtitle['index']]:
                        mapping_data['subtitles'].append({
                            'index': subtitle['index'],
                            'start_time': subtitle['start_time'],
                            'end_time': subtitle['end_time'],
                            'text': subtitle['text'],
                            'binary': binary_text,
                            'positions': positions
                        })
                    break
            
            # Ajouter les points invisibles si nécessaire
            if subtitle_positions:
                frame = self.add_invisible_points(frame, subtitle_positions, width, height)
                if frame_count % (fps * 2) == 0:  # Afficher le progrès toutes les 2 secondes
                    print(f"Frame {frame_count}/{total_frames} - Sous-titre: '{current_subtitle_text}'")
            
            out.write(frame)
            frame_count += 1
        
        # Nettoyer
        cap.release()
        out.release()
        
        # Sauvegarder le mapping de référence
        mapping_path = output_path.replace('.mp4', '_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"Encodage terminé!")
        print(f"Vidéo encodée: {output_path}")
        print(f"Mapping de référence: {mapping_path}")

def main():
    parser = argparse.ArgumentParser(description='Encoder des sous-titres dans une vidéo')
    parser.add_argument('--video', required=True, help='Chemin vers la vidéo source')
    parser.add_argument('--srt', required=True, help='Chemin vers le fichier SRT')
    parser.add_argument('--output', required=True, help='Chemin vers la vidéo de sortie')
    parser.add_argument('--grid-width', type=int, default=10, help='Largeur de la grille')
    parser.add_argument('--grid-height', type=int, default=10, help='Hauteur de la grille')
    
    args = parser.parse_args()
    
    # Vérifier que les fichiers existent
    if not os.path.exists(args.video):
        print(f"Erreur: La vidéo {args.video} n'existe pas")
        return
    
    if not os.path.exists(args.srt):
        print(f"Erreur: Le fichier SRT {args.srt} n'existe pas")
        return
    
    # Créer l'encodeur
    encoder = SubtitleEncoder(grid_size=(args.grid_width, args.grid_height))
    
    try:
        # Encoder la vidéo
        encoder.encode_video(args.video, args.srt, args.output)
    except Exception as e:
        print(f"Erreur lors de l'encodage: {e}")

if __name__ == "__main__":
    main()
