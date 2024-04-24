# Path Planning with Dynamic Obstacles

This project demonstrates path planning in a dynamic environment using the Informed RRT\* algorithm and the Bug2 algorithm. The code is written in Python and utilizes the matplotlib library for visualization.

## Credits

The Informed RRT\* code used in this project is based on Karan Chawla's and Atsushi Sakai's implementation from the [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) library by Atsushi Sakai.

## Requirements

- Python 3.x
- matplotlib
- numpy

## Installation

1. Clone the repository or download the source code.
2. Install the required dependencies using pip:

```
pip install matplotlib numpy
```

## Usage

To run the path planning simulation, execute the `main.py` script with the desired command-line arguments. The available arguments are:

- `--rand_min`: Minimum random area range (default: -2)
- `--rand_max`: Maximum random area range (default: 15)
- `--num_obstacles`: Number of static obstacles (default: 10)
- `--num_new_obstacles`: Number of new dynamic obstacles introduced after initial path planning (default: 5)
- `--min_radius`: Minimum radius of obstacles (default: 1)
- `--max_radius`: Maximum radius of obstacles (default: 1)
- `--num_tests`: Number of test cases to run (default: 25)

Example usage:

```
python main.py --num_obstacles 15 --num_new_obstacles 8 --num_tests 30
```

## Functionality

The `main.py` script performs the following steps:

1. Initializes the path planning environment with the specified parameters.
2. Generates random start and goal positions, as well as random static obstacles.
3. Runs the Informed RRT\* algorithm to find an initial path from the start to the goal.
4. Introduces new dynamic obstacles and checks for collisions with the initial path.
5. If a collision is detected, the Bug2 algorithm is used to navigate around the obstacles and reach the goal.
6. Visualizes the path planning process, including the search tree, obstacles, and the final path.
7. Repeats the process for the specified number of test cases.
8. Plots the aggregated results, including path lengths, runtimes, and the number of nodes explored.

## Output

The script will display the path planning process in a matplotlib plot, showing the search tree, obstacles, and the final path. The plot will be updated dynamically as the algorithm progresses.

After all test cases are completed, the script will display a summary plot with three subplots:

- Path Lengths: Bar graph showing the path lengths for each test case.
- Runtimes: Bar graph showing the runtime for each test case.
- Number of Nodes: Bar graph showing the number of nodes explored in each test case.

The bars in the subplots are color-coded to indicate whether the Bug2 algorithm was used (purple) or not (gray).

## Notes

- The script uses randomization to generate start and goal positions, as well as obstacle locations and sizes. Therefore, the results may vary between runs.
- The `InformedRRTStar` class is defined in the `informed_rrt_star.py` file located in the `ThirdPartyLibraryAlgorithms` folder.
- The `Bug2` class is defined in the `bug2.py` file located in the `Algorithms` folder.
- Both the `ThirdPartyLibraryAlgorithms` and `Algorithms` folders are assumed to be in the same directory as `main.py`.
- The script assumes a 2D environment for path planning.

Feel free to modify the script and experiment with different parameters to observe their impact on the path planning process.
