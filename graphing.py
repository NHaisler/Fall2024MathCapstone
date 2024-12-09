import numpy as np
import matplotlib.pyplot as plt

# Function to plot 3D data from a list of points
def plot_3d_points(points, points2=None):
    """
    Plots a 3D graph of points passed as a list of [x, y, z] coordinates.
    Optionally, plots a second set of points for comparison.
    
    Args:
        points (list or np.array): A list or array of points, each being [x, y, z].
        points2 (list or np.array, optional): A second set of points to plot (default is None).
    """
    points = np.array(points)  # Ensure points are in a numpy array format
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the first set of points (directly from the given points)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], label='Points 1', color='b')

    # If a second set of points is provided, plot it as well
    if points2 is not None:
        points2 = np.array(points2)  # Ensure points2 is in a numpy array format
        ax.plot(points2[:, 0], points2[:, 1], points2[:, 2], label='Points 2', color='r')

    # Set titles and labels
    ax.set_title('3D Plot of Given Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add legend to distinguish between the two sets
    ax.legend()
    
    # Show the plot
    plt.show()
