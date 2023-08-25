import matplotlib.pyplot as plt
import numpy as np

species = ("Base-Agent", "Yushan2022", "Cyrus2022","Helios2022","ItAndroids2022","The8")
penguin_means = {
    'Win':(86,9,3,1,40,43),
    'Draw': (12,32,7,7,41,31),
    'Lost':(2,59,90,92,19,26),
}

x = np.arange(len(species))  # the label locations
width = 0.2 # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Matches')
ax.set_title('R3CESBU team result in games against the prominent teams')
ax.set_xticks(x + 2*width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 120)
plt.savefig("./After")
plt.show()
