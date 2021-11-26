# DWA path planner
# v.1.0
# Sarkisov Y
# 25.11.2021

"""
Minimalistic solution for dynamic windows approach (aka DWA)
"""


"""
Assumptions:

- field is fixed in terms of dimensions (field_size=30 by field_size=30 meters) - so obstacles are randomly distributed there 
- obstacles are static (non moving)
- obstacles have the same geometry (circular with fixed radius)
- robot is a circle
- robot is velocity-controlled. Moreover, angular and linear velocities are independently controlled
- no noise in the prediction/measurement data (deterministic case)
- onboard sensors and PC are nice so we have quite huge prediction time
- 2 types of stucking observed:
    a) when robot in close proximity and perpendicular to the obstacle (dummy solution so far...thinking on options)
    b) when selected by user target coordinates are aligned or very close to the randomly generated obstacles (no solution so far)

Credits:

Introduction to Mobile Robotics: Path Planning and Collision Avoidance, Wolfram Burgard et al
https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf
Local path planning of mobile robot based on self- adaptive dynamic window approach, Jian hua Zhang et al 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math, random
from tkinter import *
tk = Tk()


class Robot:
    """
    Description of the robot: circular robot with radius r and certain parameters
    """
    def __init__(self, radius):
        # robot geometry
        self.robot_radius = radius  # m

        # robot actuation limits
        self.v_max = 1.5  # m/s
        self.v_min = -1  # m/s
        self.dot_v = 0.5  # m/s^2

        self.w_max = math.radians(45.0)  # rad/s
        self.w_min = - math.radians(45.0)  # rad/s
        self.dot_w = math.radians(25.0)  # rad/s^2

        # robot controller parameters
        self.sampling_time = 0.15  # s, time loop
        self.prediction_time = 4  # s, how far we can see
        self.v_dynamic_window_scale = 0.01  # scale for searching for v
        self.w_dynamic_window_scale = 0.01  # scale for searching for w

        # dwa parameters
        self.heading_gain = 1
        self.dist_gain = 0.01
        self.vel_gain = 0.5
        self.sigma_smooth_gain = 1.0

        # stuck check for the speed
        self.stuck = 0.01


class Obstacles:
    """
    Description of the obstacle (circle center (X,Y), radius)
    """

    def __init__(self, radius):
        self.obstacle_radius = radius  # m
        self.obstacle_amount_min = 10
        self.obstacle_amount_max = 30
        self.field_size = 30  # m    filed is a square with side of 30 meters
        self.obstacleN = int(random.uniform(self.obstacle_amount_min, self.obstacle_amount_max))  # random amount of obstacles on the field
        self.coordinates = np.zeros((self.obstacleN, 2))  # x,y coordinates of the obstacle centers

    def obstacles_generation(self):
        """
        generate random amount of randomly located obstacles
        """
        for next_obstacle in range(self.obstacleN):

            obstacleX = int(random.uniform(-self.field_size, self.field_size))
            obstacleY = int(random.uniform(-self.field_size, self.field_size))

            # (to do) all obstacles which are close to the robot are placed behind the working area
            # to allow the robot starting operation smoothly (not the best solution, but a fast one)
            # if (abs(abs(obstacleX) - abs(start[0]))>=4 and abs(abs(obstacleY) - abs(start[1]))>=4):
            #     self.coordinates[next_obstacle] = [obstacleX, obstacleY]
            # else:
            #     self.coordinates[next_obstacle] = [obstacleX - 60, obstacleY - 60]

            self.coordinates[next_obstacle] = [obstacleX, obstacleY]
        return self.coordinates


def motion_step(state, control):
    """
    Robot kinematics: one step within sampling time
    state = [x, y, theta, v, w]
    control = [v,w]
    """
    state = np.array(state)

    state[2] = state[2] + control[1] * robot.sampling_time
    state[0] = state[0] + control[0] * math.cos(state[2]) * robot.sampling_time
    state[1] = state[1] + control[0] * math.sin(state[2]) * robot.sampling_time
    state[3] = control[0]
    state[4] = control[1]

    return state


def dynamic_window_generation(state):
    """
    Dynamic window calculation: selection only reachable speeds that can be reached
    """
    dynamic_window = [
    max(state[3] - robot.dot_v * robot.sampling_time, robot.v_min),
    min(state[3] + robot.dot_v * robot.sampling_time, robot.v_max),
    max(state[4] - robot.dot_w * robot.sampling_time, robot.w_min),
    min(state[4] + robot.dot_w * robot.sampling_time, robot.w_max)
                     ]

    return dynamic_window


def prediction(state, control):
    """
    Predict trajectory: being at the state and applying the control, we move along trajectory
    """
    trajectory = np.array(state)

    for sample in range(int(robot.prediction_time / robot.sampling_time)):
        state = motion_step(state, control)
        trajectory = np.vstack((trajectory, state))

    return trajectory


def heading_cost(trajectory):
    """
    target heading cost: calculate angle between target and last point in trajectory
    """
    state = trajectory[-1]
    angle_to_target = math.atan2(target[1] - state[1], target[0] - state[0])
    angle_target_robot = angle_to_target - state[2]
    angle_target_robot_scaled = abs(math.atan2(math.sin(angle_target_robot), math.cos(angle_target_robot))) # limit [-pi, pi]
    heading_cost_score = abs((math.pi - angle_target_robot_scaled))

    return heading_cost_score


def dist_cost(trajectory):
    """
    obstacle avoidance cost
    """
    ox = obstacle_coordinates[:, 0]
    oy = obstacle_coordinates[:, 1]

    score = []    # to evaluate obstacle
    for state in trajectory:
        distance = np.sqrt((ox - state[0])**2 + (oy - state[1])**2)
        # if there is slightly (1.2) enough space to fit in
        if all(i > 1.2 * (obstacle.obstacle_radius + robot.robot_radius) for i in distance):
            score.append(min(distance))   # distance to the closest obstacle (most risky)
        else:
            # boom!
            return 0

    return sum(score)


def dist(trajectory):
    """
    Distance to calculate admissible velocities (fut work: merge with dist_cost())
    """
    ox = obstacle_coordinates[:, 0]
    oy = obstacle_coordinates[:, 1]

    distances = []    # to evaluate obstacle
    for state in trajectory:
        distance = np.sqrt((ox - state[0])**2 + (oy - state[1])**2)

        if all(i > (obstacle.obstacle_radius + robot.robot_radius) for i in distance):
            distances.append(min(distance))   # distance to the closest obstacle (most risky)
        else:
            distances.append(float('inf'))

    return min(distances)


def vel_cost(trajectory):
    """
    velocity cost: the higher velocity the better
    """
    speed = trajectory[-1, 3]
    if speed > robot.v_max: speed = robot.v_max

    return abs(speed)


def normalization(vector):
    """
    Normalize input vector in [0,1]. For Dist_cost - sometimes there are max=min, so we catch 'nan'
    """
    if max(vector) - min(vector) == 0:
        normal_vector = [-float('inf') for _ in range(len(vector))] #  catching nan
    else:
        normal_vector = ((np.array(vector) - min(vector)) / (max(vector) - min(vector)))
    # normal_vector = np.float64((np.array(vector) - min(vector)) / (max(vector) - min(vector)))
    return np.array(normal_vector)


def optimal_trajectory_selection(state):
    """
    Select optimal trajectory within dynamic window
    """
    dynamic_window = dynamic_window_generation(state)

    state_start = state
    heading, distance, velocity = [],[],[]
    trajectories = np.array(state)
    controls = np.zeros(2)

    for v in np.arange(dynamic_window[0], dynamic_window[1], robot.v_dynamic_window_scale):
        for w in np.arange(dynamic_window[2], dynamic_window[3], robot.w_dynamic_window_scale):

            control = [v,w]

            trajectory = prediction(state_start, control)
            size = trajectory.shape[0]

            dist_check = dist(trajectory)

            # Selecting only cases with admissible velocities (i.e., robot can stop before collision)
            if (v <= np.sqrt(2 * dist_check * robot.dot_v)) and (w <= np.sqrt(2 * dist_check * robot.dot_w)):

                trajectories = np.vstack((trajectories, trajectory))
                controls = np.vstack((controls, control))

                heading.append(heading_cost(trajectory))
                distance.append(dist_cost(trajectory))
                velocity.append(vel_cost(trajectory))

    # postprocessing
    heading =  robot.heading_gain * normalization(heading)
    distance = robot.dist_gain * normalization(distance)
    velocity = robot.vel_gain * normalization(velocity)

    final_cost = robot.sigma_smooth_gain * (heading + distance + velocity)

    optimal_index = np.argmax(final_cost) # maximize cost function
    optimal_control = controls[optimal_index + 1] # shifted at one (vstack vs append)
    optimal_trajectory = trajectories[(1 + optimal_index * size):(1 + optimal_index * size + size)]

    #if we don't put any speed to the robot but it is not at the target yet turn the robot or move it back
    if optimal_control[0] < robot.stuck and optimal_control[1] < robot.stuck:
        optimal_control = [0, robot.w_min]

    elif abs(optimal_control[0]) < robot.stuck and abs(optimal_control[1]) > robot.stuck:
        optimal_control = [robot.v_min, robot.w_min]

    return optimal_control, optimal_trajectory


def plot_field():
    """
    Plotting the field and defining static parameters
    """
    ax.set_ylim(-obstacle.field_size, obstacle.field_size)
    ax.set_xlim(-obstacle.field_size, obstacle.field_size)
    ax.grid()
    ax.plot([start[0]],[start[1]],'b1', markersize=15)
    ax.plot([target[0]], [target[1]], 'g*', markersize=30)


def plot_obstacles(radius):
    """
    Plotting the obstacles generated in obstacle_coordinates (initialized below)
    """
    for instance in obstacle_coordinates:
        ax.add_artist(plt.Circle((instance[0], instance[1]), radius, fc='r'))


def init_robot():
    """
    Initialize robot figure entity

    patch: robot body
    orientation_line: orientation of the robot
    predict: predicted optimal trajectory

    """
    patch.center = (start[0], start[1])
    ax.add_patch(patch)

    orientation_line.set_xdata([start[0], start[0] + robot.robot_radius * 2 * math.cos(theta_init)])
    orientation_line.set_ydata([start[1], start[1] + robot.robot_radius * 2 * math.sin(theta_init)])

    predict.set_xdata([start[0], start[0]])
    predict.set_ydata([start[1], start[1]])


def frame_count():
    """
    Stop condition for the drawing

    when reward > 50, we send frames = 0 to the FuncAnimation in def main()
    """
    global reward
    i = 0
    while reward <= 20:
        i += 1
        yield i


def plot_robot(i):
    """
    plotting robot

    patch: robot body
    predict: predicted optimal trajectory
    orientation_line: orientation of the robot
    """

    global reward
    global robot_state

    x, y = patch.center
    x_old = x
    y_old = y

    # 1. Pick the best trajectory and corresponding control input
    ctrl, traj = optimal_trajectory_selection(robot_state)

    # 2. Simulate motion kinematics
    robot_state = motion_step(robot_state, ctrl)

    x = robot_state[0]
    y = robot_state[1]

    # 3. Draw  optimal predicted trajectory
    predict.set_xdata(traj[:, 0])
    predict.set_ydata(traj[:, 1])

    # 4. Draw orientation or the robot
    orientation_line.set_xdata([x, x + robot.robot_radius * 2 * math.cos(robot_state[2])])
    orientation_line.set_ydata([y, y + robot.robot_radius * 2 * math.sin(robot_state[2])])

    # 5. Draw past trajectory
    ax.plot([x_old, x],[y_old, y], 'k:', lw=0.6)

    # 6. Draw robot body
    patch.center = (x, y)

    # 7. Condition to stop program (target is reached)
    if np.sqrt((x - target[0])**2 + (y - target[1])**2) <= 2*robot.robot_radius:
        reward = 50

    return patch


def get_value(entryWidget):
    """
    Credits: https://coderoad.ru/46036863/
    getting entry as something real
    """
    value = entryWidget.get()
    try:
        return int(value)
    except ValueError:
        return None


def TakeInput():
    """
    taking user input
    """
    global start
    global target
    global robot_state

    x_start = get_value(tb1)
    y_start = get_value(tb2)
    x_target = get_value(tb3)
    y_target = get_value(tb4)

    start = [x_start, y_start]
    target = [x_target, y_target]
    robot_state = [start[0], start[1], theta_init, 0.01, 0.6]
    tk.destroy()
    tk.quit()

"""
**************************
Global variable definition
**************************
"""

# User defined parameters (will be replaced by tkinter)
start  = [0,0]
target = [0,0]

# Robot instance generation (with radius in [m])
robot = Robot(1)

# obstacle instance generation with radius in [m]
obstacle = Obstacles(1)

# choose one option for obstacles (1 - random , 2 - static)
obstacle_coordinates = obstacle.obstacles_generation() # random obstacles
# obstacle_coordinates = np.array([[5,5],[10,10],[15,15],[20,20],[7,15],[7,17],[15,7],[17,7], [7,24],[20,24]]) # static obstacles

# Figure entities
fig, ax = plt.subplots() # matplotlib canvas
patch = plt.Circle(([],[]), robot.robot_radius, fc='g') # robot body
orientation_line, = plt.plot([], [], 'g-', lw=2) # orientation of the robot
predict, = plt.plot([],[],'b', lw=2) # future optimal path

# Initial conditions of the robot state (x, y, theta, v, w)
theta_init = math.pi / 8
robot_state = [start[0], start[1], theta_init, 0.01, 0.6]

# variable to denote the mission end
reward = 0

# HMI label
var = StringVar()
var.set("Enter X start, Y start, X target, Y target in the range of -30 to 30")
greeting = Message(tk, textvariable=var, relief=RAISED )
greeting.pack()

# HMI Entries
tb1 = Entry(tk)
tb1.pack()

tb2 = Entry(tk)
tb2.pack()

tb3 = Entry(tk)
tb3.pack()

tb4 = Entry(tk)
tb4.pack()

# HMI Button
b = Button(tk, text="Start simulation", command=TakeInput)
b.pack()


def main():
    """
    Launching
    """
    tk.mainloop()

    plot_field()
    plot_obstacles(obstacle.obstacle_radius)
    ani = animation.FuncAnimation(fig, plot_robot, frames = frame_count, init_func=init_robot, interval=10)
    plt.show()

if __name__ == '__main__':
    main()