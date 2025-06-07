import matplotlib.pyplot as plt
import numpy as np

# Geological periods data
periods = [
    {"name": "Cambrian", "start": 541, "end": 485},
    {"name": "Ordovician", "start": 485, "end": 444},
    {"name": "Silurian", "start": 444, "end": 419},
    {"name": "Devonian", "start": 419, "end": 359},
    {"name": "Carboniferous", "start": 359, "end": 299},
    {"name": "Permian", "start": 299, "end": 252},
    {"name": "Triassic", "start": 252, "end": 201},
    {"name": "Jurassic", "start": 201, "end": 145},
    {"name": "Cretaceous", "start": 145, "end": 66},
    {"name": "Paleogene", "start": 66, "end": 23},
    {"name": "Neogene", "start": 23, "end": 2.58},
    {"name": "Quaternary", "start": 2.58, "end": 0}
]

# Deep sky objects with distances (in Ma, approximated from light-years)
dso_data = [
    {"name": "ngc3595", "distance_ma": 60},
    {"name": "ngc5033", "distance_ma": 40},
    {"name": "ngc4921", "distance_ma": 320},
    {"name": "ngc4244", "distance_ma": 14},
    {"name": "m81", "distance_ma": 12},
    {"name": "abell2151", "distance_ma": 500},
    {"name": "m13", "distance_ma": 0.022},
    {"name": "ic4617", "distance_ma": 25},
    {"name": "m101", "distance_ma": 21},
    {"name": "m51", "distance_ma": 23},
    {"name": "m63", "distance_ma": 27},
    {"name": "m106", "distance_ma": 23},
    {"name": "ngc925", "distance_ma": 9},
    {"name": "ngc891", "distance_ma": 27},
    {"name": "m33", "distance_ma": 2.7},
    {"name": "ic727", "distance_ma": 50},
    {"name": "ngc5395", "distance_ma": 60},
    {"name": "m66", "distance_ma": 35}
]

# Prepare geological periods for plotting
period_names = [p["name"] for p in periods]
starts = [p["start"] for p in periods]
ends = [p["end"] for p in periods]
durations = [starts[i] - ends[i] for i in range(len(starts))]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(period_names))
ax.barh(y_pos, durations, left=ends, color='skyblue', edgecolor='black')

# Customize the plot
ax.set_yticks(y_pos)
ax.set_yticklabels(period_names)
ax.invert_yaxis()  # Oldest at top
ax.set_xlabel('Millions of Years Ago (Ma)')
ax.set_title('Major Geological Periods with Deep Sky Objects (Last 600 Million Years)')
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Add period boundary labels
for i, period in enumerate(periods):
    ax.text(period["end"], i, f'{period["end"]}', va='center', ha='right', fontsize=8)
    if period["start"] != 541:
        ax.text(period["start"], i, f'{period["start"]}', va='center', ha='left', fontsize=8)

# Add DSO annotations
within_timeline = []
outside_timeline = []
for dso in dso_data:
    if dso["distance_ma"] <= 541:
        within_timeline.append(dso)
    else:
        outside_timeline.append(dso)

# Place DSOs within timeline
for dso in within_timeline:
    # Find the period containing the DSO's distance
    for i, period in enumerate(periods):
        if period["end"] <= dso["distance_ma"] <= period["start"]:
            # Place text at the DSO's distance, slightly offset vertically
            ax.text(dso["distance_ma"], i + 0.2, dso["name"], fontsize=8, ha='center', color='darkred')
            break
    else:
        # If DSO is in Quaternary (0–2.58 Ma), place near 0
        if dso["distance_ma"] <= 2.58:
            ax.text(dso["distance_ma"], len(periods) - 1 + 0.2, dso["name"], fontsize=8, ha='center', color='darkred')

# List DSOs outside timeline in a text box
if outside_timeline:
    outside_text = "DSOs beyond 541 Ma:\n" + "\n".join(
        f"{dso['name']}: ~{dso['distance_ma']} Ma" for dso in outside_timeline
    )
    ax.text(0.95, 0.05, outside_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('chart.jpg', format='jpg', dpi=300)
plt.close()