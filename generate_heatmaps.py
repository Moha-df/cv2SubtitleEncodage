import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Data from experiments
data = {
    'Camouflage': [0, 70, 100, 100, 100, 100],
    'LocalRadius': [0, 20, 0, 10, 30, 300],
    'OriginsDecodes': [100.0, 100.0, 91.7, 0.0, 16.7, 100.0],
    'ScoreMoyenGlobal': [99.1, 90.1, 80.5, 0.0, 5.2, 74.6],
    'Bruit': [0.0, 10.0, 5.9, 100.0, 94.1, 9.2]
}

# Define parameter ranges
camouflage_values = [0, 70, 100]
localradius_values = [0, 10, 20, 30, 300]

# Initialize matrices with NaN
origins_matrix = np.full((len(camouflage_values), len(localradius_values)), np.nan)
score_matrix = np.full((len(camouflage_values), len(localradius_values)), np.nan)
bruit_matrix = np.full((len(camouflage_values), len(localradius_values)), np.nan)

# Fill matrices with data
for i in range(len(data['Camouflage'])):
    cam = data['Camouflage'][i]
    rad = data['LocalRadius'][i]
    
    cam_idx = camouflage_values.index(cam)
    rad_idx = localradius_values.index(rad)
    
    origins_matrix[cam_idx, rad_idx] = data['OriginsDecodes'][i]
    score_matrix[cam_idx, rad_idx] = data['ScoreMoyenGlobal'][i]
    bruit_matrix[cam_idx, rad_idx] = data['Bruit'][i]

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Impact of Camouflage and LocalRadius Parameters on Subtitle Decoding Performance', 
             fontsize=14, fontweight='bold', y=1.02)

# Custom colormaps
cmap_green_red = LinearSegmentedColormap.from_list('green_red', ['#8B0000', '#FF6B6B', '#FFD93D', '#90EE90', '#006400'])
cmap_blue_yellow = LinearSegmentedColormap.from_list('blue_yellow', ['#8B0000', '#FF6B6B', '#FFD93D', '#6BB6FF', '#00008B'])
cmap_red_green = LinearSegmentedColormap.from_list('red_green', ['#006400', '#90EE90', '#FFD93D', '#FF6B6B', '#8B0000'])

# Heatmap 1: Origins Decoded
sns.heatmap(origins_matrix, 
            ax=axes[0],
            cmap=cmap_green_red,
            annot=True, 
            fmt='.1f',
            cbar_kws={'label': 'Percentage (%)'},
            xticklabels=localradius_values,
            yticklabels=camouflage_values,
            vmin=0,
            vmax=100,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'fontsize': 10, 'fontweight': 'bold'})
axes[0].set_title('Subtitle Decoding Success Rate', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('LocalRadius (pixels)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Camouflage (%)', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='both', which='major', labelsize=10)

# Heatmap 2: Score Moyen Global
sns.heatmap(score_matrix, 
            ax=axes[1],
            cmap=cmap_blue_yellow,
            annot=True, 
            fmt='.1f',
            cbar_kws={'label': 'Score (%)'},
            xticklabels=localradius_values,
            yticklabels=camouflage_values,
            vmin=0,
            vmax=100,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'fontsize': 10, 'fontweight': 'bold'})
axes[1].set_title('Average Global Decoding Quality', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('LocalRadius (pixels)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Camouflage (%)', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='both', which='major', labelsize=10)

# Heatmap 3: Bruit
sns.heatmap(bruit_matrix, 
            ax=axes[2],
            cmap=cmap_red_green,
            annot=True, 
            fmt='.1f',
            cbar_kws={'label': 'Percentage (%)'},
            xticklabels=localradius_values,
            yticklabels=camouflage_values,
            vmin=0,
            vmax=100,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'fontsize': 10, 'fontweight': 'bold'})
axes[2].set_title('False Positive Rate (Noise)', fontsize=12, fontweight='bold', pad=10)
axes[2].set_xlabel('LocalRadius (pixels)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Camouflage (%)', fontsize=11, fontweight='bold')
axes[2].tick_params(axis='both', which='major', labelsize=10)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('subtitle_encoding_heatmaps.png', dpi=300, bbox_inches='tight')
print("[v0] Heatmaps generated successfully: subtitle_encoding_heatmaps.png")

# Display the plot
plt.show()
