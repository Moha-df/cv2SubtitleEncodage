#!/usr/bin/env python3
"""
Version avec grille configurable pour les points visibles
"""

import cv2
import numpy as np
import re
import json
from typing import List, Tuple, Dict
import argparse
import os

class SubtitleEncoderVisibleLarge:
    def __init__(self, grid_size: Tuple[int, int] = (16, 16)):
        """
        Encodeur avec grille configurable
        """
        self.grid_width, self.grid_height = grid_size
        self.grid_size = grid_size
        self.point_size = 6 
        self.point_intensity = 50
        
    def parse_srt(self, srt_file: str) -> List[Dict]:
        """Parse un fichier SRT"""
        subtitles = []
        
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        

        # pour lire le fichier srt
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            index, start_time, end_time, text = match
            text = text.strip().replace('\n', ' ')
            
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
    
    def binary_to_grid_positions(self, binary_str: str) -> List[Tuple[int, int]]:
        """Convertit binaire en positions de grille"""
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
    
    def add_visible_points(self, frame: np.ndarray, positions: List[Tuple[int, int]], 
                          frame_width: int, frame_height: int) -> np.ndarray:
        """Ajoute des points ROUGES visibles"""
        modified_frame = frame.copy()
        
        cell_width = frame_width // self.grid_width
        cell_height = frame_height // self.grid_height
        
        for x, y in positions:
            center_x = x * cell_width + cell_width // 2
            center_y = y * cell_height + cell_height // 2
            
            # Cercle rouge
            cv2.circle(modified_frame, (center_x, center_y), self.point_size, (0, 0, 255), -1)
            # Contour blanc
            cv2.circle(modified_frame, (center_x, center_y), self.point_size + 2, (255, 255, 255), 2)
        
        return modified_frame
    
    def encode_video(self, video_path: str, srt_path: str, output_path: str):
        """Encode la vidéo"""
        print(f"Parsing des sous-titres avec grille {self.grid_width}×{self.grid_height}...")
        subtitles = self.parse_srt(srt_path)
        print(f"Trouvé {len(subtitles)} sous-titres")
        
        # Analyser la capacité de la grille
        max_chars = (self.grid_width * self.grid_height) // 8
        print(f"Capacité de la grille: {max_chars} caractères max")
        
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
            
            if subtitle_positions:
                frame = self.add_visible_points(frame, subtitle_positions, width, height)
                if frame_count % (fps * 2) == 0:
                    print(f"Frame {frame_count}/{total_frames} - Sous-titre: '{current_subtitle_text}' - {len(subtitle_positions)} points")
            
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
    parser = argparse.ArgumentParser(description='Encoder avec grille configurable')
    parser.add_argument('--video', required=True, help='Vidéo source')
    parser.add_argument('--srt', required=True, help='Fichier SRT')
    parser.add_argument('--output', required=True, help='Vidéo de sortie')
    parser.add_argument('--grid-width', type=int, default=16, help='Largeur de la grille')
    parser.add_argument('--grid-height', type=int, default=16, help='Hauteur de la grille')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Erreur: {args.video} n'existe pas")
        return
    
    if not os.path.exists(args.srt):
        print(f"Erreur: {args.srt} n'existe pas")
        return
    
    encoder = SubtitleEncoderVisibleLarge(grid_size=(args.grid_width, args.grid_height))
    
    try:
        encoder.encode_video(args.video, args.srt, args.output)
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
