#!/usr/bin/env python3
"""
Script de comparaison des statistiques de décodage
Compare le mapping original avec les sous-titres décodés
"""

import json
import argparse
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import difflib

def load_json(filename: str) -> Dict:
    """Charge un fichier JSON"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """Nettoie le texte pour la comparaison"""
    if not text:
        return ""
    # Supprimer caractères spéciaux mais garder ponctuation utile
    cleaned = re.sub(r'[^\w\s:.,!?*\'"-]', '', text.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calcule la similarité entre deux textes (premiers 32 caractères seulement)"""
    # Ne comparer que les 32 premiers caractères
    clean1 = clean_text(text1[:32])
    clean2 = clean_text(text2[:32])
    
    if not clean1 or not clean2:
        return 0.0
    
    # Utiliser SequenceMatcher pour une meilleure similarité
    return difflib.SequenceMatcher(None, clean1, clean2).ratio()

def find_best_match(decoded_text: str, original_subtitles: List[Dict]) -> Tuple[Dict, float]:
    """Trouve le meilleur match pour un texte décodé"""
    best_match = None
    best_score = 0.0
    
    for original in original_subtitles:
        score = calculate_text_similarity(decoded_text, original['text'])
        if score > best_score:
            best_score = score
            best_match = original
    
    return best_match, best_score

def calculate_stats(original_mapping: Dict, decoded_json: Dict) -> Dict:
    """Calcule les statistiques de décodage avec analyse détaillée par sous-titre"""
    original_subtitles = original_mapping.get('subtitles', [])
    decoded_subtitles = decoded_json.get('subtitles', [])
    
    stats = {
        'total_original': len(original_subtitles),
        'total_decoded': len(decoded_subtitles),
        'original_best_scores': {},  # index -> meilleur score obtenu
        'original_matches': {},      # index -> liste des décodages qui matchent
        'noise_decodings': [],       # décodages qui ne matchent rien
        'all_matches': [],           # tous les matches avec détails
        'noise_percentage': 0.0,
        'average_best_score': 0.0,
        'coverage_percentage': 0.0,  # % d'originaux qui ont au moins un match
    }
    
    # Initialiser les dictionnaires pour chaque original
    for original in original_subtitles:
        idx = original.get('index', 0)
        stats['original_best_scores'][idx] = 0.0
        stats['original_matches'][idx] = []
    
    # Analyser chaque décodage
    for decoded in decoded_subtitles:
        best_original, score = find_best_match(decoded['text'], original_subtitles)
        
        if best_original and score > 0.3:  # Seuil minimum pour considérer comme match
            original_idx = best_original.get('index', 0)
            
            # Enregistrer le match
            match_info = {
                'decoded_text': decoded['text'],
                'original_text': best_original['text'],
                'similarity': score,
                'decoded_confidence': decoded.get('confidence', 0),
                'original_index': original_idx,
                'group_size': decoded.get('group_size', 1)
            }
            stats['all_matches'].append(match_info)
            
            # Mettre à jour le meilleur score pour cet original
            if score > stats['original_best_scores'][original_idx]:
                stats['original_best_scores'][original_idx] = score
            
            # Ajouter à la liste des matches pour cet original
            stats['original_matches'][original_idx].append(match_info)
        else:
            # C'est du bruit
            stats['noise_decodings'].append({
                'text': decoded['text'],
                'confidence': decoded.get('confidence', 0),
                'group_size': decoded.get('group_size', 1)
            })
    
    # Calculer les statistiques finales
    if stats['original_best_scores']:
        best_scores = [score for score in stats['original_best_scores'].values() if score > 0]
        stats['average_best_score'] = sum(best_scores) / len(best_scores) if best_scores else 0.0
        
        # % d'originaux qui ont été décodés (au moins un match)
        covered_originals = sum(1 for score in stats['original_best_scores'].values() if score > 0)
        stats['coverage_percentage'] = covered_originals / stats['total_original'] if stats['total_original'] > 0 else 0
    
    # % de bruit
    stats['noise_percentage'] = len(stats['noise_decodings']) / stats['total_decoded'] if stats['total_decoded'] > 0 else 0
    
    return stats

def print_stats(stats: Dict, original_mapping: Dict = None):
    """Affiche les statistiques détaillées"""
    print("=== STATISTIQUES DE DÉCODAGE DÉTAILLÉES ===\n")
    
    print(f"📊 Sous-titres originaux: {stats['total_original']}")
    print(f"🔍 Sous-titres décodés: {stats['total_decoded']}")
    print(f"✅ Matches trouvés: {len(stats['all_matches'])}")
    print(f"🗑️  Bruit détecté: {len(stats['noise_decodings'])}\n")
    
    print("📈 Couverture des originaux:")
    print(f"  % d'originaux décodés: {stats['coverage_percentage']*100:.1f}%")
    print(f"  Score moyen des meilleurs décodages: {stats['average_best_score']*100:.1f}%")
    print(f"  % de bruit dans les décodages: {stats['noise_percentage']*100:.1f}%\n")
    

    
    print("\n📊 Analyse du bruit:")
    if stats['noise_decodings']:
        print(f"  Total bruit: {len(stats['noise_decodings'])} décodages")
        print("  Exemples de bruit:")
        for noise in stats['noise_decodings'][:5]:
            print(f"    '{noise['text'][:50]}...' (conf: {noise['confidence']:.2f}, groupe: {noise.get('group_size', 1)})")
    
    print("\n🔍 Détails par sous-titre original:")
    for idx in sorted(stats['original_best_scores'].keys()):
        score = stats['original_best_scores'][idx]
        matches = stats['original_matches'][idx]
        status = "✅ Décodé" if score > 0 else "❌ Non trouvé"
        print(f"  #{idx}: {status} - Meilleur: {score*100:.1f}% - Matches: {len(matches)}")
    
    print("\n💡 Résumé:")
    print(f"  • {stats['coverage_percentage']*100:.1f}% des sous-titres originaux ont été décodés")
    print(f"  • Score moyen des décodages réussis: {stats['average_best_score']*100:.1f}%")
    print(f"  • {stats['noise_percentage']*100:.1f}% des décodages sont du bruit")

def main():
    parser = argparse.ArgumentParser(description='Compare les statistiques de décodage')
    parser.add_argument('--original', required=True, help='Fichier mapping original JSON')
    parser.add_argument('--decoded', required=True, help='Fichier décodé JSON')
    parser.add_argument('--output', help='Fichier de sortie pour les stats détaillées')
    
    args = parser.parse_args()
    
    try:
        print("📂 Chargement des fichiers...")
        original_mapping = load_json(args.original)
        decoded_json = load_json(args.decoded)
        
        print("🔬 Calcul des statistiques...")
        stats = calculate_stats(original_mapping, decoded_json)
        
        print_stats(stats, original_mapping)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Stats détaillées sauvegardées dans {args.output}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()