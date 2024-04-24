"""
Informed RRT* path planning

author: Karan Chawla
        Atsushi Sakai(@Atsushi_twi)

Reference: Informed RRT*: Optimal Sampling-based Path planning Focused via
Direct Sampling of an Admissible Ellipsoidal Heuristic
https://arxiv.org/pdf/1404.2334.pdf

Modified by Aaron Goldstein for University of North Florida - CAP6635 - Advanced AI Term Project 
"""
import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from .utils.angle import rot_mat_2d

show_animation = True


class InformedRRTStar:

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=0.5,
             goal_sample_rate=10, max_iter=200, ax=None):
        """
        Initialize the Informed RRT* algorithm with the specified parameters.

        :param start: Starting point of the path as a list [x, y].
        :param goal: Goal point of the path as a list [x, y].
        :param obstacle_list: List of obstacles, each defined as a tuple (x, y, radius).
        :param rand_area: Random sampling area [min, max] for x and y coordinates.
        :param expand_dis: Expansion distance for each new node.
        :param goal_sample_rate: Percentage chance of sampling the goal point.
        :param max_iter: Maximum number of iterations to run the algorithm.
        :param ax: Matplotlib axis object for plotting. If None, a new figure is created.
        """

        # Create start and goal nodes using the Node class, which stores coordinates and tree information
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])

        # Define the random sampling area for the x and y coordinates
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]

        # Set the expansion distance for each step towards a randomly sampled point
        self.expand_dis = expand_dis

        # Set the rate at which the goal is sampled instead of a random point
        self.goal_sample_rate = goal_sample_rate

        # Set the maximum number of iterations the algorithm will run
        self.max_iter = max_iter

        # Store the list of obstacles, each defined by its center coordinates and radius
        self.obstacle_list = obstacle_list

        # Initialize the list that will store all nodes created during the search
        self.node_list = None

        # Setup the plotting axis; if none is provided, create a new figure and axis
        self.ax = ax
        if self.ax is None:
            self.fig, self.ax = plt.subplots()  # Create a new matplotlib figure and axis
        else:
            self.fig = ax.figure  # Use the existing figure associated with the provided axis

    def informed_rrt_star_search(self, animation=True):
        """
        Perform the Informed RRT* search to find an optimal path from start to goal.

        :param animation: Boolean flag to indicate whether to animate the search process.
        """

        # Initialize the node list with the start node
        self.node_list = [self.start]

        # Initialize the best cost as infinity; this will store the cost of the best path found
        c_best = float('inf')

        # Set to store nodes that are part of the solution path
        solution_set = set()

        # Initialize the path variable; this will store the best path once found
        path = None

        # Calculate the minimum cost (straight line distance) from start to goal
        c_min = math.hypot(self.start.x - self.goal.x, self.start.y - self.goal.y)

        # Calculate the center of the search space based on start and goal positions
        x_center = np.array([[(self.start.x + self.goal.x) / 2.0],
                            [(self.start.y + self.goal.y) / 2.0], [0]])

        # Calculate the direction vector from start to goal normalized by c_min
        a1 = np.array([[(self.goal.x - self.start.x) / c_min],
                    [(self.goal.y - self.start.y) / c_min], [0]])

        # Calculate the rotation angle of the search space
        e_theta = math.atan2(a1[1, 0], a1[0, 0])

        # Compute the rotation matrix for the search space
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)  # First column of identity matrix transposed
        m = a1 @ id1_t
        u, s, vh = np.linalg.svd(m, True, True)
        c = u @ np.diag([1.0, 1.0, np.linalg.det(u) * np.linalg.det(np.transpose(vh))]) @ vh

        # Main loop for the RRT* search
        for i in range(self.max_iter):
            # Sample a random point in the informed search space
            rnd = self.informed_sample(c_best, c_min, x_center, c)

            # Find the nearest node in the tree to the sampled point
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[n_ind]

            # Steer from the nearest node towards the sampled point
            theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
            new_node = self.get_new_node(theta, n_ind, nearest_node)
            d = self.line_cost(nearest_node, new_node)

            # Check if the path to the new node is collision-free
            no_collision = self.check_collision(nearest_node, theta, d)

            if no_collision:
                # Find nearby nodes for potential connection
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)

                # Add the new node to the tree
                self.node_list.append(new_node)
                self.rewire(new_node, near_inds)

                # Check if the new node is close to the goal and the path is collision-free
                if self.is_near_goal(new_node):
                    if self.check_segment_collision(new_node.x, new_node.y, self.goal.x, self.goal.y):
                        solution_set.add(new_node)
                        last_index = len(self.node_list) - 1
                        temp_path = self.get_final_course(last_index)
                        temp_path_len = self.get_path_len(temp_path)
                        if temp_path_len < c_best:
                            path = temp_path
                            c_best = temp_path_len

            # Optionally animate the search process
            if animation:
                self.draw_graph(x_center=x_center, c_best=c_best, c_min=c_min, e_theta=e_theta, rnd=rnd)

        return path

    def choose_parent(self, new_node, near_inds):
        """
        Choose the best parent for the new node from the nearby nodes based on the cost.

        :param new_node: The new node for which the parent is being chosen.
        :param near_inds: Indices of nearby nodes that could potentially be parents.
        :return: The new node with updated parent and cost if a better parent is found.
        """
        # If there are no nearby nodes, return the new node as is
        if len(near_inds) == 0:
            return new_node

        # List to store the cost of reaching the new node from each nearby node
        d_list = []
        for i in near_inds:
            # Calculate the Euclidean distance from a nearby node to the new node
            dx = new_node.x - self.node_list[i].x
            dy = new_node.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)

            # Check if the path from the nearby node to the new node is collision-free
            if self.check_collision(self.node_list[i], theta, d):
                # If collision-free, add the cost to reach the new node to the list
                d_list.append(self.node_list[i].cost + d)
            else:
                # If there is a collision, set the cost to infinity
                d_list.append(float('inf'))

        # Find the minimum cost from the list
        min_cost = min(d_list)
        min_ind = near_inds[d_list.index(min_cost)]

        # If the minimum cost is infinity, no valid parent was found, return the new node as is
        if min_cost == float('inf'):
            print("min cost is inf")
            return new_node

        # Update the new node's cost and parent to the one with the minimum cost
        new_node.cost = min_cost
        new_node.parent = min_ind

        return new_node

    def find_near_nodes(self, new_node):
        """
        Find nodes in the vicinity of the new node based on a calculated radius.

        :param new_node: The new node around which nearby nodes are to be found.
        :return: Indices of nearby nodes within the calculated radius.
        """
        # Calculate the number of nodes currently in the node list
        n_node = len(self.node_list)

        # Calculate the radius within which to look for nearby nodes
        # The radius depends on the number of nodes and is designed to shrink
        # as the number of nodes increases, focusing the search as the tree densifies
        r = 50.0 * math.sqrt(math.log(n_node) / n_node)

        # Compute the squared distances from all nodes to the new node
        d_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]

        # Find indices of nodes whose distances are within the square of the radius
        # This avoids computing a square root, enhancing performance
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]

        # Return the list of indices of nearby nodes
        return near_inds

    def informed_sample(self, c_max, c_min, x_center, c):
        if c_max < float('inf'):
            if c_max >= c_min:
                r = [c_max / 2.0, math.sqrt(c_max**2 - c_min**2) / 2.0, math.sqrt(c_max**2 - c_min**2) / 2.0]
                rl = np.diag(r)
                x_ball = self.sample_unit_ball()
                rnd = np.dot(np.dot(c, rl), x_ball) + x_center
                rnd = [rnd[(0, 0)], rnd[(1, 0)]]
            else:
                # Handle the case when c_max is smaller than c_min
                # You can choose to raise an exception, assign a default value, or take other appropriate action
                raise ValueError("c_max must be greater than or equal to c_min")
        else:
            rnd = self.sample_free_space()

        return rnd

    @staticmethod
    def sample_unit_ball():
        """
        Sample a point uniformly from within a unit ball (3D sphere with radius 1).

        :return: A numpy array representing a point [x, y, z] in the unit ball.
        """
        # Generate two random numbers between 0 and 1
        a = random.random()
        b = random.random()

        # Ensure that 'a' is the smaller number to maintain uniformity in distribution
        if b < a:
            a, b = b, a

        # Calculate the coordinates using spherical to Cartesian coordinate transformation
        sample = (b * math.cos(2 * math.pi * a / b),
                b * math.sin(2 * math.pi * a / b))

        # Return the sample as a numpy array with a z-coordinate of 0 (since we only need 2D)
        return np.array([[sample[0]], [sample[1]], [0]])

    def sample_free_space(self):
        """
        Sample a random point from the free configuration space defined by the random area bounds.

        :return: A list [x, y] representing a randomly sampled point in the search space.
        """
        # Decide whether to sample the goal point based on the goal sampling rate
        if random.randint(0, 100) > self.goal_sample_rate:
            # Sample a random point within the defined bounds of the search area
            rnd = [random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand)]
        else:
            # Sample the goal point to guide the search towards the goal
            rnd = [self.goal.x, self.goal.y]

        return rnd

    @staticmethod
    def get_path_len(path):
        """
        Calculate the total length of a given path.

        :param path: A list of points [x, y] that make up the path.
        :return: The total length of the path as a float.
        """
        path_len = 0
        # Iterate through the path points to sum up the distances between consecutive points
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            # Calculate the Euclidean distance between two consecutive points
            path_len += math.hypot(node1_x - node2_x, node1_y - node2_y)

        return path_len

    @staticmethod
    def line_cost(node1, node2):
        """
        Calculate the Euclidean distance between two nodes.

        :param node1: The first node as an instance of the Node class.
        :param node2: The second node as an instance of the Node class.
        :return: The Euclidean distance between node1 and node2 as a float.
        """
        # Compute the Euclidean distance between the two nodes using their x and y coordinates
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        """
        Find the index of the node in the list that is closest to the given random point.

        :param nodes: A list of Node objects representing the current nodes in the RRT* tree.
        :param rnd: A list [x, y] representing the random point to which the nearest node is sought.
        :return: The index of the nearest node in the nodes list.
        """
        # Calculate the squared Euclidean distance from each node to the random point
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in nodes]

        # Find the index of the node with the minimum distance to the random point
        min_index = d_list.index(min(d_list))

        # Return the index of the nearest node
        return min_index

    def get_new_node(self, theta, n_ind, nearest_node):
        """
        Generate a new node in the direction of the sampled point from the nearest node.

        :param theta: The angle in radians pointing from the nearest node towards the sampled point.
        :param n_ind: The index of the nearest node in the node list.
        :param nearest_node: The nearest node object from which the new node will be extended.
        :return: A new Node object positioned along the direction specified by theta.
        """
        # Create a deep copy of the nearest node to inherit its properties
        new_node = copy.deepcopy(nearest_node)

        # Calculate the new node's coordinates by moving in the direction of theta by the expand distance
        new_node.x += self.expand_dis * math.cos(theta)
        new_node.y += self.expand_dis * math.sin(theta)

        # Update the cost of reaching the new node by adding the expansion distance to the nearest node's cost
        new_node.cost += self.expand_dis

        # Set the parent of the new node to the index of the nearest node
        new_node.parent = n_ind

        # Return the newly created node
        return new_node

    def is_near_goal(self, node):
        """
        Determine if the specified node is near the goal within the threshold defined by the expansion distance.

        :param node: The node to check for proximity to the goal.
        :return: True if the node is within the expansion distance of the goal, False otherwise.
        """
        # Calculate the Euclidean distance from the node to the goal
        d = self.line_cost(node, self.goal)

        # Check if the distance is less than the expansion distance
        if d < self.expand_dis:
            # If the distance is less than the expansion distance, the node is considered near the goal
            return True

        # If the distance is not less than the expansion distance, the node is not considered near the goal
        return False

    def rewire(self, new_node, near_inds):
        """
        Attempt to rewire the tree by changing the parent of nearby nodes to the new node if it provides a shorter path.

        :param new_node: The newly added node to the tree.
        :param near_inds: Indices of nearby nodes that might be rewired.
        """
        # Get the number of nodes currently in the node list
        n_node = len(self.node_list)

        # Iterate through each nearby node index
        for i in near_inds:
            near_node = self.node_list[i]

            # Calculate the Euclidean distance between the new node and this nearby node
            d = math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)

            # Calculate the potential new cost if this nearby node were to be rewired to the new node
            s_cost = new_node.cost + d

            # Check if the new cost is less than the current cost of the nearby node
            if near_node.cost > s_cost:
                # Calculate the angle between the new node and the nearby node
                theta = math.atan2(new_node.y - near_node.y, new_node.x - near_node.x)

                # Check if the path between the new node and the nearby node is collision-free
                if self.check_collision(near_node, theta, d):
                    # If the path is clear, update the parent of the nearby node to the new node
                    near_node.parent = n_node - 1
                    # Update the cost of reaching the nearby node
                    near_node.cost = s_cost

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        """
        Calculate the squared distance from a point p to a line segment vw.

        :param v: The start point of the line segment.
        :param w: The end point of the line segment.
        :param p: The point for which the distance to the segment is calculated.
        :return: The squared distance from point p to the line segment vw.
        """
        # Check if the segment points v and w are equal
        if np.array_equal(v, w):
            # If v and w are equal, the segment is a point
            # Return the squared distance from p to the point v (or w)
            return (p - v).dot(p - v)

        # Calculate the squared length of the segment vw
        # Avoid using sqrt for performance reasons
        l2 = (w - v).dot(w - v)

        # Consider the line extending the segment vw,
        # parameterized as v + t * (w - v), where t is a scalar parameter.
        # The projection of point p onto this line falls at the point
        # where t = [(p - v) · (w - v)] / |w - v|^2

        # Calculate the value of t for the projection point
        t = (p - v).dot(w - v) / l2

        # Clamp t to the range [0, 1] to handle points outside the segment vw
        # If t < 0, the projection falls before v (closer to v)
        # If t > 1, the projection falls after w (closer to w)
        t = max(0, min(1, t))

        # Calculate the projection point on the segment vw
        projection = v + t * (w - v)

        # Return the squared distance from point p to the projection point
        return (p - projection).dot(p - projection)

    def check_segment_collision(self, x1, y1, x2, y2):
        """
        Check if a line segment from (x1, y1) to (x2, y2) collides with any obstacles.

        :param x1: The x-coordinate of the start point of the line segment.
        :param y1: The y-coordinate of the start point of the line segment.
        :param x2: The x-coordinate of the end point of the line segment.
        :param y2: The y-coordinate of the end point of the line segment.
        :return: True if there is no collision with any obstacles, False otherwise.
        """
        # Iterate through each obstacle in the obstacle list
        for (ox, oy, size) in self.obstacle_list:
            # Calculate the squared distance from the obstacle center to the line segment
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]), np.array([x2, y2]), np.array([ox, oy]))
            
            # Check if the squared distance is less than or equal to the square of the obstacle's radius
            if dd <= size ** 2:
                # If true, the line segment collides with the obstacle
                return False  # Collision detected

        # If no collisions are detected with any obstacles, return True
        return True

    def check_collision(self, near_node, theta, d):
        """
        Check if the path from a node to a new position defined by angle and distance is free of obstacles.

        :param near_node: The starting node from which the path originates.
        :param theta: The angle in radians indicating the direction of the path from the starting node.
        :param d: The distance from the starting node to the end of the path.
        :return: True if the path is free of obstacles, False otherwise.
        """
        # Create a temporary node to represent the end of the path
        tmp_node = copy.deepcopy(near_node)
        end_x = tmp_node.x + math.cos(theta) * d
        end_y = tmp_node.y + math.sin(theta) * d

        # Use the check_segment_collision method to determine if the path intersects any obstacles
        return self.check_segment_collision(tmp_node.x, tmp_node.y, end_x, end_y)

    def get_final_course(self, last_index):
        """
        Construct the final path from the start node to the goal by tracing back from the goal.

        :param last_index: The index of the last node in the node list, which is near the goal.
        :return: A list of coordinates representing the path from the start to the goal.
        """

        # Start with nothing but goal in the path
        path = [[self.goal.x, self.goal.y]]

        # Trace back from the goal to the start using the parent links
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent  # Move to the parent node
        
        # Add the start position to complete the path
        path.append([self.start.x, self.start.y])

        # Return the path as constructed from goal to start
        return path

    def draw_graph(self, x_center=None, c_best=None, c_min=None, e_theta=None, rnd=None):
        """
        Draw the search tree, obstacles, and path on a matplotlib plot.

        :param x_center: The center of the search space (optional).
        :param c_best: The cost of the best path found so far (optional).
        :param c_min: The minimum cost between start and goal (optional).
        :param e_theta: The angle of the ellipse's rotation (optional).
        :param rnd: The randomly sampled point (optional).
        """
        self.ax.clear()  # Clear the current figure

        # Stop simulation if 'escape' key is pressed
        self.fig.canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])

        # Plot the randomly sampled point and the search ellipse
        if rnd is not None:
            self.ax.plot(rnd[0], rnd[1], "^k")
            if c_best != float('inf'):
                self.plot_ellipse(x_center, c_best, c_min, e_theta, self.ax)

        # Plot the search tree
        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    self.ax.plot([node.x, self.node_list[node.parent].x],
                             [node.y, self.node_list[node.parent].y], "-g")

        # Plot the obstacles
        for (ox, oy, size) in self.obstacle_list:
            self.ax.plot(ox, oy, "ok", ms=30 * size)

        # Plot the start and goal points
        self.ax.plot(self.start.x, self.start.y, "xr")
        self.ax.plot(self.goal.x, self.goal.y, "xr")

        # Set the plot limits and display the grid
        self.ax.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        self.ax.grid(True)
        plt.pause(0.01)  # Add a small pause to allow the plot to update

    @staticmethod
    def plot_ellipse(x_center, c_best, c_min, e_theta, ax):
        """
        Plot the search ellipse based on the current best path cost.

        :param x_center: The center of the search ellipse.
        :param c_best: The cost of the best path found so far.
        :param c_min: The minimum cost between start and goal.
        :param e_theta: The angle of the ellipse's rotation.
        """
        a = math.sqrt(c_best**2 - c_min**2) / 2.0  # Semi-major axis length
        b = c_best / 2.0  # Semi-minor axis length
        angle = math.pi / 2.0 - e_theta  # Angle of rotation
        cx = x_center[0]  # Center x-coordinate
        cy = x_center[1]  # Center y-coordinate
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)  # Parametric parameter for the ellipse
        x = [a * math.cos(it) for it in t]  # x-coordinates of the ellipse
        y = [b * math.sin(it) for it in t]  # y-coordinates of the ellipse
        fx = rot_mat_2d(-angle) @ np.array([x, y])  # Rotate the ellipse
        px = np.array(fx[0, :] + cx).flatten()  # Translated x-coordinates
        py = np.array(fx[1, :] + cy).flatten()  # Translated y-coordinates
        ax.plot(cx, cy, "xc")  # Plot the center of the ellipse
        ax.plot(px, py, "--c")  # Plot the ellipse

class Node:
    """
    Represents a node in the RRT* search tree.

    Attributes:
        x (float): The x-coordinate of the node.
        y (float): The y-coordinate of the node.
        cost (float): The cost to reach this node from the start node.
        parent (Node): The parent node of this node in the search tree.
    """

    def __init__(self, x, y):
        """
        Initialize a new Node object.

        Args:
            x (float): The x-coordinate of the node.
            y (float): The y-coordinate of the node.
        """
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None

