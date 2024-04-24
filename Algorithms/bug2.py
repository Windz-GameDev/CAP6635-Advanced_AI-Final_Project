import matplotlib.pyplot as plt
import numpy as np

class Bug2:
    def __init__(self, start_x, start_y, goal_x, goal_y, obstacles, ax=None):
        # Initialize the starting coordinates of the robot
        self.start_x = start_x
        self.start_y = start_y
        
        # Initialize the goal coordinates the robot aims to reach
        self.goal_x = goal_x
        self.goal_y = goal_y
        
        # List of obstacles in the environment, each defined by (x, y, radius)
        self.obstacles = obstacles
        
        # Set the current position of the robot to the starting position
        self.current_x = start_x
        self.current_y = start_y
        
        # Initialize the path with the starting position
        self.path = [(start_x, start_y)]
        
        # Matplotlib axis object for plotting, if provided
        self.ax = ax
        
        # If no axis is provided, create a new figure and axis for plotting
        # if self.ax is None:
        #    self.fig, self.ax = plt.subplots()
        
        # Optionally plot initial conditions (uncomment to use)
        # self.plot_initial_conditions()
        
        # Optionally print initialization details (uncomment to use)
        # print(f"Bug 2 initialized with start: ({start_x}, {start_y}), goal: ({goal_x}, {goal_y}), and {len(obstacles)} obstacles")

    def plot_initial_conditions(self):
        # Plot the start position on the provided matplotlib axis with a green circle
        self.ax.plot(self.start_x, self.start_y, 'go', label='Start')
        
        # Plot the goal position on the provided matplotlib axis with a red circle
        self.ax.plot(self.goal_x, self.goal_y, 'ro', label='Goal')
        
        # Iterate through each obstacle in the list and plot it
        for obs in self.obstacles:
            # Create a circle at the obstacle's position with the specified radius
            # The color is gray and partially transparent (alpha=0.3)
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.3)
            
            # Add the circle to the plot
            self.ax.add_artist(circle)
        
        # Set the plot to have equal scaling on both axes, which ensures that circles appear as circles
        self.ax.axis('equal')
        
        # Update the plot legend to include the labels for start and goal positions
        self.update_legend()
        
        # Optionally print a message indicating that the initial conditions have been plotted
        # print("Initial conditions plotted")

    def is_obstacle_hit(self, x, y):
        # Iterate through each obstacle in the list to check for a collision
        for obs in self.obstacles:
            # Calculate the Euclidean distance from the point (x, y) to the center of the obstacle (obs[0], obs[1])
            if np.linalg.norm(np.array([x - obs[0], y - obs[1]])) < obs[2]:
                # If the distance is less than the radius of the obstacle, a collision is detected
                return True
        # If no obstacles are hit, return False
        return False

    def is_goal_reached(self, x, y):
        # Calculate the Euclidean distance from the current position (x, y) to the goal position
        return np.linalg.norm(np.array([x - self.goal_x, y - self.goal_y])) < 0.1

    def follow_boundary(self, obstacle, direction):
        # Extract the center coordinates and radius of the obstacle
        x, y, radius = obstacle
        
        # Initialize a list to store the boundary points
        boundary_points = []
        
        # Define the step size for boundary traversal, smaller step size for finer resolution
        step_size = 0.05
        
        # Calculate the number of steps needed to make a full circle around the obstacle
        num_steps = int(2 * np.pi * radius / step_size)
        
        # Iterate over each step to calculate boundary points
        for i in range(num_steps):
            # Calculate the angle for the current step
            angle = i * step_size / radius
            
            # Adjust the angle for clockwise direction if specified
            if direction == 'clockwise':
                angle = -angle
            
            # Calculate the new x and y coordinates on the boundary
            new_x = x + radius * np.cos(angle)
            new_y = y + radius * np.sin(angle)
            
            # Check if the new boundary point does not collide with any obstacle
            if not self.is_obstacle_hit(new_x, new_y):
                # Add the boundary point to the list if it's valid
                boundary_points.append((new_x, new_y))
        
        # Return the list of boundary points
        return boundary_points
    
    def leave_condition(self, x, y, d_min):
        # Calculate the current distance from the point (x, y) to the goal
        d_current = np.linalg.norm(np.array([x - self.goal_x, y - self.goal_y]))
        
        # Check if the current distance is less than the minimum distance encountered so far
        if d_current < d_min:
            # Iterate through each obstacle to check if a direct line to the goal intersects any obstacle
            for obs in self.obstacles:
                if self.is_line_segment_intersecting_obstacle((x, y), (self.goal_x, self.goal_y), obs):
                    # If an intersection is found, the leave condition is not satisfied
                    return False
            # If no intersections are found, the leave condition is satisfied
            return True
        # If the current distance is not less than the minimum distance, the leave condition is not satisfied
        return False

    def is_line_segment_intersecting_obstacle(self, start, end, obstacle):
        # Extract the coordinates of the start and end points of the line segment
        x1, y1 = start
        x2, y2 = end
        
        # Extract the center coordinates and radius of the obstacle
        ox, oy, radius = obstacle

        # Calculate the components of the vector from start to end of the line segment
        dx = x2 - x1
        dy = y2 - y1
        
        # Compute coefficients for the quadratic formula (derived from the line-circle intersection equation)
        a = dx ** 2 + dy ** 2
        b = 2 * (dx * (x1 - ox) + dy * (y1 - oy))
        c = (x1 - ox) ** 2 + (y1 - oy) ** 2 - radius ** 2

        # Calculate the discriminant of the quadratic equation
        discriminant = b ** 2 - 4 * a * c
        
        # If the discriminant is negative, there are no real roots, and the line does not intersect the obstacle
        if discriminant < 0:
            return False

        # Calculate the real roots of the quadratic equation
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)

        # Check if either of the roots is within the segment's range (0 <= t <= 1)
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True

        # If neither root is within the range, the line does not intersect the obstacle within the segment
        return False

    def move_to_goal(self):
        # Initialize the minimum distance to infinity to start tracking the closest approach to the goal
        d_min = np.inf
        # hit_point = None
        hit_obstacle = None

        # Continue moving towards the goal until the goal is reached
        while not self.is_goal_reached(self.current_x, self.current_y):
            # Calculate the direction vector from the current position to the goal
            direction = np.array([self.goal_x - self.current_x, self.goal_y - self.current_y])
            direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
            step_size = 0.1  # Define the step size for each movement towards the goal

            # Calculate the new position by moving along the direction vector
            new_x = self.current_x + direction[0] * step_size
            new_y = self.current_y + direction[1] * step_size

            # Check if the new position collides with any obstacle
            if self.is_obstacle_hit(new_x, new_y):
                # Record the point of collision and the corresponding obstacle
                hit_point = (self.current_x, self.current_y)
                hit_obstacle = self.get_hit_obstacle(new_x, new_y)
                break  # Stop moving directly towards the goal and prepare to follow the obstacle boundary

            # Update the current position to the new position
            self.current_x = new_x
            self.current_y = new_y
            # Add the new position to the path
            self.path.append((self.current_x, self.current_y))
            # Optionally plot the path (uncomment to use)
            # self.plot_path()

        # If an obstacle was hit, follow its boundary
        if hit_obstacle is not None:
            # Follow the boundary of the obstacle in a clockwise direction
            boundary_points = self.follow_boundary(hit_obstacle, 'clockwise')
            # Extend the path with the boundary points
            self.path.extend(boundary_points)
            # Optionally plot the path (uncomment to use)
            # self.plot_path()

            # After following the boundary, check each point to see if it's a good point to leave the boundary
            for x, y in boundary_points:
                if self.leave_condition(x, y, d_min):
                    # If leaving the boundary is favorable, update the current position and recursively move to the goal
                    self.current_x = x
                    self.current_y = y
                    self.move_to_goal()
                    break

                # Update the minimum distance to the goal if the current point is closer than previously recorded
                d_current = np.linalg.norm(np.array([x - self.goal_x, y - self.goal_y]))
                if d_current < d_min:
                    d_min = d_current

        # Return the path taken to reach the goal or to follow the obstacle
        return self.path

    def get_hit_obstacle(self, x, y):
        # Iterate through each obstacle in the list to find which one was hit
        for obs in self.obstacles:
            # Calculate the distance from the current point (x, y) to the center of the obstacle (obs[0], obs[1])
            if np.linalg.norm(np.array([x - obs[0], y - obs[1]])) < obs[2]:
                # If the distance is less than the radius of the obstacle, then this is the obstacle that was hit
                return obs
        # Return None if no obstacle was hit (though this should not happen in the context where this method is called)
        return None

    def plot_path(self):
        # Extract x and y coordinates from the path list
        x, y = zip(*self.path)
        
        # Plot the path as a blue line on the provided matplotlib axis
        self.ax.plot(x, y, 'b-', label='Path')
        
        # Mark the current position with a blue 'x'
        self.ax.plot(x[-1], y[-1], 'bx', label='Current Position')
        
        # Update the plot legend to include new labels
        self.update_legend()
        
        # Pause the plot update briefly to allow for dynamic visualization
        plt.pause(0.1)
        
        # Optionally print a message indicating that the path has been plotted (uncomment to use)
        # print("Path plotted")

    def update_legend(self):
        # Retrieve the current handles and labels from the plot
        handles, labels = self.ax.get_legend_handles_labels()
        
        # Create a dictionary to eliminate any duplicate labels
        by_label = dict(zip(labels, handles))
        
        # Update the legend on the plot with the unique handles and labels
        self.ax.legend(by_label.values(), by_label.keys())
        
        # Redraw the plot to reflect the updated legend
        plt.draw()
        
        # Optionally print a message indicating that the legend has been updated (uncomment to use)
        # print("Legend updated")

    def run(self):
        # Print a message indicating the start of the Bug 2 algorithm
        print("Running Bug 2 algorithm")
        
        # Execute the move_to_goal method to attempt to reach the goal while avoiding obstacles
        path = self.move_to_goal()

        # Optionally print the final status based on whether the goal was reached
        if self.is_goal_reached(self.current_x, self.current_y):
            
            # If the goal is reached, append the goal coordinates to the path
            path.append((self.goal_x, self.goal_y))
            print("Goal reached with Bug2!")
        else:
            print("No path found to the goal with Bug2")

        # Print a message indicating the completion of the Bug 2 algorithm
        print(f"Bug 2 algorithm finished.")
        
        # Return the path taken by the robot
        return path