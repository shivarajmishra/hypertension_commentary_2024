import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, TextArea, HPacker, OffsetImage
from matplotlib.patches import Circle

# Define the color map
color_map = {
    'High income': '#1f77b4',  # Blue
    'Upper middle income': '#ff7f0e',  # Orange
    'Lower middle income': '#2ca02c',  # Green
    'Low income': '#d62728',  # Red
}

class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)  # Calculate radius based on area

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r  # Set bubble radii
        self.bubbles[:, 3] = area  # Set bubble areas
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.any(distance < 0)

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.where(distance < 0)[0]

    def collapse(self, n_iterations=50):
        for _ in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                dir_vec = self.com - self.bubbles[i, :2]

                if np.linalg.norm(dir_vec) > 0:
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)

                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    colliding_indices = self.collides_with(new_bubble, rest_bub)
                    for colliding in colliding_indices:
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        if np.linalg.norm(dir_vec) > 0:
                            dir_vec = dir_vec / np.linalg.norm(dir_vec)
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        new_point1 = (self.bubbles[i, :2] + orth * self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth * self.step_dist)
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2
                if self.step_dist < 0.01:
                    break

    def plot(self, ax, labels, colors, text_sizes=None):
        if text_sizes is None:
            text_sizes = [8] * len(labels)  # Default font size if not provided

        for i in range(len(self.bubbles)):
            circ = plt.Circle(self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=text_sizes[i], color='white')

        # Add a legend
        unique_colors = list(set(colors))
        legend_labels = [labels[colors.index(color)] for color in unique_colors]
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                              markersize=10, markerfacecolor=color) for label, color in zip(legend_labels, unique_colors)]
        ax.legend(handles=handles, loc='best')

def add_bubble_scale(ax, sizes, colors, labels, position=(0.9, 0.1)):
    for i, size in enumerate(sizes):
        da = DrawingArea(20, 20, 0, 0)
        p = Circle((10, 10), size, color=colors[i])
        da.add_artist(p)
        text = TextArea(labels[i], textprops=dict(color=colors[i], size=10, fontweight='bold'))
        vbox = HPacker(children=[da, text], align="center", pad=0, sep=3)
        ab = AnnotationBbox(vbox, position,
                            xybox=(1.005, position[1] + 0.05 * i),
                            xycoords='axes fraction',
                            boxcoords=("axes fraction", "axes fraction"),
                            box_alignment=(0.2, 0.5),
                            bboxprops=dict(alpha=0))
        ax.add_artist(ab)

def add_image(ax, image_path, position=(0.9, 0.1), zoom=0.2):
    img = plt.imread(image_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, position,
                        xycoords='axes fraction',
                        boxcoords="axes fraction",
                        box_alignment=(0.5, 0.5),
                        bboxprops=dict(edgecolor='none', alpha=0))
    ax.add_artist(ab)

# Load data
df = pd.read_excel("datafile_pop_uncontrolled.xlsx", index_col=None) #replace this with the right datasets here

# Initialize BubbleChart
bubble_chart = BubbleChart(area=df['pop_unc'],
                           bubble_spacing=1)

# Collapse bubbles to fit the chart
bubble_chart.collapse()

# Set text sizes based on conditions
df.loc[df['pop_unc'] <= 1500000, 'text_sizes'] = 4
df.loc[(df['pop_unc'] > 1500000) & (df['pop_unc'] < 6500000), 'text_sizes'] = 10
df.loc[df['pop_unc'] > 6500000, 'text_sizes'] = 14

df.loc[df['pop_unc'] <= 1500000, 'Country_code'] = df['Country_code']
df.loc[(df['pop_unc'] > 1500000) & (df['pop_unc'] < 6500000), 'Country_code'] = df['Country']
df.loc[df['pop_unc'] > 6500000, 'Country_code'] = df['Country']

# Ensure Country_code column exists for use in labels
df['Country_code'] = df['Country_code'].astype(str)

# Add color column based on income category
Income = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
df['color'] = df['Income'].map(color_map)

# Prepare text sizes and colors
text_sizes = df['text_sizes'].tolist()
colors = df['color'].tolist()
labels = df['Country_code'].tolist()

# Plot
fig, ax = plt.subplots(figsize=(20, 12), subplot_kw=dict(aspect="equal"))
bubble_chart.plot(ax, labels, colors, text_sizes=text_sizes)

# Add color legend based on income categories
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markersize=10, markerfacecolor=color_map[label]) for label in color_map]
ax.legend(handles=handles, loc='best')

# # Add bubble scale
# bubble_sizes = [5, 10, 15]  # Example sizes for the scale
# bubble_colors = ['#e6e600', '#999900', '#666600']  # Example colors for the scale
# bubble_labels = ['<30K', '30-100K', '>100K']  # Example labels for the scale
# add_bubble_scale(ax, bubble_sizes, bubble_colors, bubble_labels, position=(0.9, 0.1))

# Add image scale
add_image(ax, 'scale.png', position=(0.98, 0.1), zoom=0.6)

ax.axis("off")
ax.relim()
ax.autoscale_view()
ax.set_title('')
plt.tight_layout()
plt.savefig('HTNphy.png', format='png', bbox_inches='tight', pad_inches=0.1)
plt.show()
