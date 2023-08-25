import matplotlib.pyplot as plt
import numpy as np

# Set custom font style for the entire plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

fig, ax = plt.subplots(figsize=(8, 6))

size = 0.3
vals = np.array([[122,68], [9,3], [30, 10]])
labels = ['Avg Total Passes(190)', 'Avg Total Shoots(12)', 'Avg Total Dribbles(40)']

sub_labels = [['intercepted', 'Completed'], ['missed', 'inGoalie'], ['missed', 'succesful']]

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap([1, 2, 5, 6, 9, 10])

total_outer = vals.sum(axis=1).sum()
total_inner = vals.sum()

outer_percentages = [f'{(val/total_outer)*100:.1f}%' for val in vals.sum(axis=1)]
inner_percentages = [f'{(val/total_inner)*100:.1f}%' for val in vals.flatten()]
inner_numbers = vals.flatten()

wedges, outer_texts, outer_autopct = ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
                                 wedgeprops=dict(width=size, edgecolor='w'),
                                 labels=labels, startangle=90,
                                 autopct='')

for text, pct in zip(outer_texts, outer_percentages):
    text.set_text(f"{text.get_text()}\n{pct}")

for wedge in wedges:
    wedge.set_linewidth(2)
    wedge.set_edgecolor('white')

wedges, inner_texts, inner_autopct = ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
                                             wedgeprops=dict(width=size, edgecolor='w'),
                                             labels=None, startangle=90,
                                             autopct='')

for wedge, number, sub_label in zip(wedges, inner_numbers, np.array(sub_labels).flatten()):
    angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
    x = (1-size/2) * np.cos(np.deg2rad(angle))
    y = (1-size/2) * np.sin(np.deg2rad(angle))

    ax.text(x, y, f"{sub_label}\n{int(number)}", ha='center', va='center', fontsize=12, color='white')

# Enhance text and title aesthetics
title_font = {'fontsize': 20, 'fontweight': 'bold', 'fontname': 'Arial'}
plt.title('Statistics of 100 Games', **title_font)

ax.set(aspect="equal")
ax.legend(wedges, labels, title="Categories", loc="lower left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout()
plt.savefig('./beforeCircle')
plt.show()
