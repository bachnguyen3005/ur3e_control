import csv
import numpy as np
# Example function to save waypoints to a CSV file
def save_waypoints_to_csv(waypoints, filename='ur3e_waypoints.csv'):
    """
    Save a list of 3D points to a CSV file.
    
    Args:
        waypoints: List of (x, y, z) tuples or lists
        filename: Name of the CSV file to save
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['x', 'y', 'z'])
        # Write waypoints
        for point in waypoints:
            writer.writerow(point)
    
    print(f"Saved {len(waypoints)} waypoints to {filename}")

# Example usage:
waypoints = []
for i in range(50):
    # This is just an example of generating points in a spiral
    # Replace this with your actual 50 waypoints
    t = i / 5
    x = 0.3 + 0.05 * t * np.cos(t)
    y = 0.0 + 0.05 * t * np.sin(t)
    z = 0.4 - 0.002 * i
    waypoints.append([x, y, z])

save_waypoints_to_csv(waypoints)