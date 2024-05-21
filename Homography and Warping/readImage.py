import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor
import matplotlib.image as mpimg


def readImage(img1, img2):
    # Create a 1x2 grid of subplots
    fig, ax = plt.subplots(1, 2)

    # Plot the images
    ax[0].imshow(img1)
    ax[0].set_title('Image 1')

    ax[1].imshow(img2)
    ax[1].set_title('Image 2')

    #Use cursors
    cursor1 = Cursor(ax[0], useblit=True, color='red', linewidth=1)
    cursor2 = Cursor(ax[1], useblit=True, color='red', linewidth=1)

    points_1, points_2 = [], []
    # # Function to handle mouse clicks
    def onclick(event):
        if event.inaxes == ax[0]:
            print(f"View 1: Clicked at ({event.xdata:.2f}, {event.ydata:.2f})")
            points_1.append([event.xdata, event.ydata])
        elif event.inaxes == ax[1]:
            print(f"View 2: Clicked at ({event.xdata:.2f}, {event.ydata:.2f})")
            points_2.append([event.xdata, event.ydata])
    # # Connect the click event to the function
    fig.canvas.mpl_connect('button_press_event', onclick)

    # # Show the plot
    plt.tight_layout()
    plt.show()

    assert (len(points_1) > 3) and (len(points_2) > 3), \
        f"Have to choose a minimum of 4 points. Your len(points_1) = {len(points_1)}, len(points_2) = {len(points_2)}" 
    assert len(points_1) == len(points_2), \
        f"Have to choose equal number of points on both sides. Your len(points_1) = {len(points_1)} != len(points_2) = {len(points_2)}" 
    return np.array(points_1).T, np.array(points_2).T

