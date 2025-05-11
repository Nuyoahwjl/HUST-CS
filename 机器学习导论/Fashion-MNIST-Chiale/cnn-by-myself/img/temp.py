import matplotlib.patches as patches
import matplotlib.pyplot as plt
def draw_architecture():
    fig, ax = plt.subplots(figsize=(16, 6))
    layers = [
        ("Input\n(1, 28, 28)", 0.5, 2, 0.8, 1.8, "lightgray"),
        ("Conv1\n(6, 28, 28)", 1.8, 2.2, 0.8, 1.4, "skyblue"),
        ("BN+ReLU", 2.7, 2.2, 0.6, 1.4, "lightgreen"),
        ("Pool1\n(6, 14, 14)", 3.5, 2.4, 0.8, 1.0, "violet"),
        ("Conv2\n(16, 10, 10)", 4.8, 2.2, 0.8, 1.4, "skyblue"),
        ("BN+ReLU", 5.7, 2.2, 0.6, 1.4, "lightgreen"),
        ("Pool2\n(16, 5, 5)", 6.5, 2.4, 0.8, 1.0, "violet"),
        ("Flatten", 7.8, 2.4, 0.8, 1.0, "lightgray"),
        ("FC1\n(120)", 8.9, 2.4, 0.8, 1.0, "salmon"),
        ("ReLU", 9.8, 2.4, 0.6, 1.0, "lightgreen"),
        ("FC2\n(10)", 10.7, 2.4, 0.8, 1.0, "salmon"),
    ]
    for label, x, y, width, height, color in layers:
        rect = patches.FancyBboxPatch((x, y), width, height,
                                      boxstyle="round,pad=0.02",
                                      linewidth=1,
                                      edgecolor='black',
                                      facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, label,
                ha='center', va='center', fontsize=8)
    ax.set_xlim(0, 12)
    ax.set_ylim(1.5, 4.5)
    ax.axis('off')
    plt.show()

draw_architecture()