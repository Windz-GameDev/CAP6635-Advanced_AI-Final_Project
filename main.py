from ThirdPartyLibraryAlgorithms.informed_rrt_star import InformedRRTStar
from Algorithms.bug2 import Bug2
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import time

def distance_squared_point_to_segment(v, w, p):
    # Check if the segment start and end points are the same
    if np.array_equal(v, w):
        # If the segment is actually a point, return the squared distance from p to this point
        return (p - v).dot(p - v)
    
    # Calculate the squared length of the segment vw
    l2 = (w - v).dot(w - v)
    
    # Project point p onto the line defined by v and w
    t = max(0, min(1, (p - v).dot(w - v) / l2))
    
    # Calculate the projection as a point on the segment
    projection = v + t * (w - v)
    
    # Return the squared distance from p to the projection point on the segment
    return (p - projection).dot(p - projection)

def find_collision_point(path, obstacles):
    # Iterate over each segment in the path
    for i, (segment_start, segment_end) in enumerate(zip(path[:-1], path[1:])):
        x1, y1 = segment_start
        x2, y2 = segment_end
        
        # Check each obstacle for a collision with the current segment
        for (ox, oy, size) in obstacles:
            # Calculate the squared distance from the obstacle center to the segment
            dd = distance_squared_point_to_segment(
                np.array([x1, y1]), np.array([x2, y2]), np.array([ox, oy]))
            
            # Check if the distance is less than or equal to the obstacle's radius squared
            if dd <= size ** 2:
                # Calculate the vector along the segment
                vec = np.array([x2 - x1, y2 - y1])
                vec_len = np.linalg.norm(vec)
                vec_unit = vec / vec_len
                
                # Calculate the scalar projection of the obstacle center onto the segment
                t = ((ox - x1) * vec_unit[0] + (oy - y1) * vec_unit[1])
                
                # Calculate the actual collision point
                collision_point = np.array([x1, y1]) + t * vec_unit
                
                # Return the index of the segment and the collision point
                return i, (collision_point[0], collision_point[1])
    
    # Return None if no collision is found
    return None, None

def generate_random_obstacles(num_obstacles, rand_area, min_radius, max_radius, start, goal):
    obstacles = []
    # Continue generating obstacles until the desired number is reached
    while len(obstacles) < num_obstacles:
        # Generate random coordinates and radius for the obstacle
        x = random.uniform(rand_area[0], rand_area[1])
        y = random.uniform(rand_area[0], rand_area[1])
        radius = random.uniform(min_radius, max_radius)
        
        # Check if the new obstacle overlaps with the start or goal
        if not (is_in_obstacle(x, y, radius, start) or is_in_obstacle(x, y, radius, goal)):
            # If no overlap, add the obstacle to the list
            obstacles.append((x, y, radius))
    
    return obstacles

def is_in_obstacle(x, y, radius, point):
    # Calculate the squared distance from the point to the center of the obstacle
    # and check if it is less than the squared radius of the obstacle
    return (x - point[0]) ** 2 + (y - point[1]) ** 2 < radius ** 2

def plot_results(results):
    # Unpack results into separate lists
    # 'results' is expected to be a list of tuples, where each tuple contains:
    # (path_length, runtime, num_nodes, bug_run)
    path_lengths, runtimes, num_nodes, bug_runs = zip(*results)

    # Create a figure with three subplots arranged horizontally (1 row, 3 columns)
    # 'figsize' sets the width and height of the figure in inches
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    labels_added = set()  # Initialize an empty set to track which labels have been added to the legend

    # Iterate over each result using enumerate to get both the index (i) and the values
    # 'zip' is used again to iterate over elements of path_lengths, runtimes, num_nodes, and bug_runs simultaneously
    for i, (length, runtime, nodes, bug_run) in enumerate(zip(path_lengths, runtimes, num_nodes, bug_runs)):
        # Choose the color based on whether the Bug algorithm was used
        color = 'purple' if bug_run else 'gray'
        # Set the label based on whether the Bug algorithm was used
        label = 'Online (Bug used)' if bug_run else 'Offline'
        
        # Check if the label has already been added to avoid duplicates in the legend
        if label not in labels_added:
            # Plot bars with labels only the first time the label is used
            axs[0].bar(i + 1, length, color=color, label=label)
            axs[1].bar(i + 1, runtime, color=color, label=label)
            axs[2].bar(i + 1, nodes, color=color, label=label)
            labels_added.add(label)  # Add the label to the set
        else:
            # Plot bars without labels for subsequent data points with the same label
            axs[0].bar(i + 1, length, color=color)
            axs[1].bar(i + 1, runtime, color=color)
            axs[2].bar(i + 1, nodes, color=color)

    # Set titles and labels for each subplot
    axs[0].set_title('Path Lengths')
    axs[0].set_xlabel('Run')
    axs[0].set_ylabel('Length')
    axs[1].set_title('Runtimes')
    axs[1].set_xlabel('Run')
    axs[1].set_ylabel('Time (s)')
    axs[2].set_title('Number of Nodes')
    axs[2].set_xlabel('Run')
    axs[2].set_ylabel('Nodes')

    # Add a legend to the first subplot. The legend will contain unique labels
    # 'loc' specifies the location of the legend
    axs[0].legend(loc='upper right')
    plt.tight_layout()  # Adjust the layout to make sure there is no content overlap
    plt.show()  # Display the plot

def run_path_planning(args, ax):
    # Record the start time to calculate the total runtime later
    start_time = time.time()

    # Flag to indicate whether the Bug2 algorithm was used
    bug_run = False

    # Generate random start and goal positions within the specified range
    start_x = random.uniform(args.rand_min, args.rand_max)
    start_y = random.uniform(args.rand_min, args.rand_max)
    goal_x = random.uniform(args.rand_min, args.rand_max)
    goal_y = random.uniform(args.rand_min, args.rand_max)

    # Store start and goal positions as lists for easier handling
    start = [start_x, start_y]
    goal = [goal_x, goal_y]

    # Define the area in which random obstacles will be placed
    rand_area = [args.rand_min, args.rand_max]

    # Ensure that plotted circles (representing obstacles) appear as circles and not ellipses
    ax.set_aspect('equal')

    # Announce the start of the path planning process using Informed RRT*
    print("Starting main program with Informed RRT*")

    # Generate a list of obstacles that do not overlap with the start or goal positions
    obstacle_list = generate_random_obstacles(args.num_obstacles, rand_area, args.min_radius, args.max_radius, start, goal)

    # Create an instance of the InformedRRTStar class with the generated parameters
    rrt_star = InformedRRTStar(start=start, goal=goal, rand_area=rand_area, obstacle_list=obstacle_list, ax=ax)

    # Perform the Informed RRT* search to find a path from start to goal
    path = rrt_star.informed_rrt_star_search(animation=False)

    if path:
        # If a path is found, reverse it to start from the initial start node
        path.reverse()

        # Draw the graph showing nodes and edges, and plot the found path
        rrt_star.draw_graph()
        ax.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        # Generate additional obstacles to simulate a dynamic environment
        new_obstacle_list = generate_random_obstacles(args.num_new_obstacles, rand_area, args.min_radius, args.max_radius, start, goal)

        # Plot these new obstacles with a distinct color
        for obs in new_obstacle_list:
            ax.plot(obs[0], obs[1], 'o', ms=30 * obs[2], color='blue')

        # Check for any collisions with the new obstacles along the found path
        print("Checking for collisions...")
        collision_index, collision_point = find_collision_point(path, new_obstacle_list)

        if collision_index is not None:
            # If a collision is detected, set the flag indicating the Bug2 algorithm will be used
            bug_run = True
            print(f"Collision detected at index: {collision_index} and point: {collision_point}. Initiating Bug Algorithm.")

            # Extract the portion of the path before the collision to use as the starting path for Bug2
            path_before_collision = path[:collision_index+1]

            # Combine the original and new obstacles for the Bug2 algorithm
            combined_obstacle_list = obstacle_list + new_obstacle_list

            # Initialize the Bug2 algorithm with the collision point as the new start position
            bug_2 = Bug2(start_x=collision_point[0], start_y=collision_point[1], goal_x=goal[0], goal_y=goal[1], obstacles=combined_obstacle_list)
            bug_path = bug_2.run()

            if bug_path is None:
                path = None
            else:
                # If a complete path was found
                if not (bug_path[-1][0] != goal[0] or bug_path[-1][1] != goal[1]):
                    # If Bug2 finds a path, merge it with the portion of the path before the collision
                    path = path_before_collision + bug_path[1:]  # Avoid duplicating the collision point
                else:
                    path = None

                # Plot the portion of the new path created by Bug2
                x_path, y_path = zip(*bug_path)
                ax.plot(x_path, y_path, '-', color='purple')
        else:
            print("No collision detected with new obstacles, no need to run Bug Algorithm.")
    else:
        print("No path found.")

    # Calculate the total runtime of the path planning process
    end_time = time.time()
    runtime = end_time - start_time

    # Calculate the length of the path and the number of nodes used in the RRT* algorithm
    if path:
        path_length = sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(path[:-1], path[1:]))
        num_nodes = len(rrt_star.node_list)
    else:
        path_length = 0
        num_nodes = len(rrt_star.node_list)

    # Return the path length, runtime, number of nodes, and whether the Bug algorithm was used
    return path_length, runtime, num_nodes, bug_run

def main():
    # Initialize the argument parser with a description of the program's purpose
    parser = argparse.ArgumentParser(description="Path planning with dynamic obstacles.")
    
    # Define command-line arguments to configure the path planning environment
    parser.add_argument("--rand_min", type=float, default=-2, help="Minimum random area range")
    parser.add_argument("--rand_max", type=float, default=15, help="Maximum random area range")
    parser.add_argument("--num_obstacles", type=int, default=10, help="Number of static obstacles")
    parser.add_argument("--num_new_obstacles", type=int, default=5, help="Number of new dynamic obstacles introduced after initial path planning")
    parser.add_argument("--min_radius", type=float, default=1, help="Minimum radius of obstacles")
    parser.add_argument("--max_radius", type=float, default=1, help="Maximum radius of obstacles")
    parser.add_argument("--num_tests", type=int, default=25, help="Number of test cases to run")
    
    # Parse the arguments provided by the user or use the default values
    args = parser.parse_args()

    # Calculate the number of rows and columns needed for subplot visualization based on the number of tests
    num_cols = 5  # Define the number of columns for subplots
    num_rows = (args.num_tests + num_cols - 1) // num_cols  # Calculate the number of rows needed

    # Create a figure with subplots arranged in a grid defined by the number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axs = axs.flatten()  # Flatten the 2D array of axes into a 1D array for easier access

    # Initialize a list to store the results from each test case
    results = []

    # Loop through each test case, running the path planning algorithm and collecting results
    for i in range(args.num_tests):
        print(f"Running test case {i + 1}/{args.num_tests}")  # Inform the user about the progress
        
        # Check if there are enough subplots for the current test case
        if i < len(axs):
            # Run the path planning function with the current settings and subplot axis
            result = run_path_planning(args, axs[i])
            
            # If a result is returned, add it to the results list
            if result:
                results.append(result)
        else:
            # If there are no more subplots available, stop the loop
            break

    # Once all test cases have been run, plot the aggregated results
    plot_results(results)

# Check if this script is being run as the main program and not as a module
if __name__ == '__main__':
    main()