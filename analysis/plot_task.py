import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import os

# List of image file paths
image_paths = [
    '../papers/figures/empty_slots.png',
    '../papers/figures/empty_slots.png',
    '../papers/figures/empty_slots.png',
    # Add more image paths as needed
]

if __name__ == '__main__':

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.5, 1, 0.1, 0.7])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0],
                                               width_ratios=[1, 1, 1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1])
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[3],
                                               width_ratios=[0.6, 1, 1.5, 0.6])


        # Load the image
    img = mpimg.imread('../papers/figures/empty_slots.png')
    ax1 = fig.add_subplot(gs_row1[1])

    ax1.imshow(img)
    ax1.axis('off')

    img2 = mpimg.imread('../papers/figures/task_up.png')

    ax2 = fig.add_subplot(gs_row2[0])
    ax2.imshow(img2)
    ax2.axis('off')

    img3 = mpimg.imread('../papers/figures/empty_slots_utargets.png')
    ax3 = fig.add_subplot(gs_row3[1])
    ax3.imshow(img3)
    ax3.axis('off')

    img4 = mpimg.imread('../papers/figures/task_conditions.png')
    ax4 = fig.add_subplot(gs_row3[2])
    ax4.imshow(img4)
    ax4.axis('off')

    # Add labels
    fig.text(0.16, 0.9, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.16, 0.34, 'B', fontsize=13, fontweight='bold', va='center')
    fig.text(0.54, 0.34, 'C', fontsize=13, fontweight='bold', va='center')

    # Adjust layout
    plt.tight_layout()
    plt.show()
