import sys
import matplotlib.pyplot as plt
import numpy as np

# Try to dynamically load from the codebase to prevent drift (Acceptance Criteria 3)
try:
    sys.path.append('../../')  # Adjust relative path if needed
    FLOWER_EEF_MIX = [
    ("eef_droid", 0.35),
    ("bridge_dataset", 4.0),
    ("fractal20220817_data", 2.0),
    ("dobbe", 2.0),
    # ("fmb", 1.0),
    ("bc_z", 0.8),
    ("cmu_play_fusion", 6.0),
    ("libero_10_no_noops", 10.0),
    ("libero_goal_no_noops", 18.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 6.0),
]
    raw_mix = FLOWER_EEF_MIX
    print("Loaded data mix from flower_vla/dataset/oxe/mixes.py")
except ImportError:
    print("Could not import FLOWER_EEF_MIX. Falling back to hardcoded values.")
    raw_mix = [
        ('libero_goal_no_noops', 18.0),
        ('libero_10_no_noops', 10.0),
        ('cmu_play_fusion', 6.0),
        ('stanford_hydra_dataset_converted_externally_to_rlds', 6.0),
        ('eef_droid', 0.35),
        ('bridge_dataset', 4.0),
        ('fractal20220817_data', 2.0),
        ('dobbe', 2.0),
        ('bc_z', 0.8)
    ]

# Define the taxonomy and color mapping (Okabe-Ito Palette)
# Grouped as: (Embodiment, Color, List of Datasets)
taxonomy = [
    ("Franka", "#E69F00", [
        'libero_goal_no_noops', 
        'libero_10_no_noops', 
        'cmu_play_fusion', 
        'stanford_hydra_dataset_converted_externally_to_rlds', 
        'eef_droid'
    ]),
    ("WidowX", "#56B4E9", ['bridge_dataset']),
    ("Google Robot", "#009E73", ['fractal20220817_data']),
    ("Stretch", "#0072B2", ['dobbe']),
    ("Everyday Robots", "#CC79A7", ['bc_z'])
]


# Convert raw_mix to a dictionary for easy lookup
mix_dict = dict(raw_mix)
total_weight = sum(mix_dict.values())

# Flatten data for plotting
labels = []
weights = []
pcts = []
colors = []

# Process active training datasets
for embodiment, color, datasets in taxonomy:
    for ds in datasets:
        if ds in mix_dict:
            w = mix_dict[ds]
            labels.append(ds)
            weights.append(w)
            pcts.append((w / total_weight) * 100)
            colors.append(color)

# Setup the plot
plt.rcParams.update({
    "font.family": "serif",
    "axes.axisbelow": True, # Ensure grid is behind bars
})

fig, ax = plt.subplots(figsize=(10, 6))

# Plot active bars
y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, pcts, color=colors, height=0.7)

# Add absolute weight annotations to the right of each bar
for i, (bar, w) in enumerate(zip(bars, weights)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{w:.2f}", va='center', ha='left', fontsize=10, color='black')

# Formatting axes
ax.set_yticks(np.arange(len(labels)))
# Escape underscores so they render cleanly in the text, but keep the raw code names
ax.set_yticklabels(labels, fontsize=9)

ax.set_xlabel('Normalized Sampling Weight (%)', fontsize=11)
ax.set_title('Pretraining Data Mix ($\mathtt{flowereef}$)', loc='left', fontsize=14, fontweight='bold', pad=15)

# Invert y-axis so the largest/first items are at the top
ax.invert_yaxis()

# Vertical grid lines only
ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
ax.yaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add the sum annotation in the bottom right
ax.text(0.95, 0.05, f"$\Sigma = {total_weight:.2f}$\n(Weights re-normalized by dataloader)", 
        transform=ax.transAxes, ha='right', va='bottom', 
        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="lightgray"))

# # Add embodiment group labels on the left axis (optional visual grouping)
# y_offset = 0
# for embodiment, color, datasets in taxonomy:
#     count = len([d for d in datasets if d in mix_dict])
#     if count > 0:
#         # Place text at the vertical center of the group
#         center_y = y_offset + (count - 1) / 2
#         # Use an annotation pointing to the axis
#         ax.annotate(embodiment, xy=(-0.02, center_y), xycoords=('axes fraction', 'data'),
#                     xytext=(-0.35, center_y), textcoords=('axes fraction', 'data'),
#                     ha='right', va='center', fontsize=11, fontweight='bold', color=color,
#                     arrowprops=dict(arrowstyle='-[', color=color, lw=2, 
#                                     shrinkA=0, shrinkB=0,
#                                     connectionstyle="angle,angleA=0,angleB=90,rad=0"))
#         y_offset += count

plt.tight_layout()
plt.savefig("datamix.pdf", format='pdf', bbox_inches='tight')
print("Successfully saved datamix.pdf")