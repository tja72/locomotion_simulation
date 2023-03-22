"""Example of whole body controller on A1 robot."""
import sys

from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import numpy as np
import os
import scipy.interpolate
import time
import math

import pickle
import pybullet_data
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error

from locomotion.agents.whole_body_controller import com_velocity_estimator
from locomotion.agents.whole_body_controller import gait_generator as gait_generator_lib
from locomotion.agents.whole_body_controller import locomotion_controller
from locomotion.agents.whole_body_controller import openloop_gait_generator
from locomotion.agents.whole_body_controller import raibert_swing_leg_controller
from locomotion.agents.whole_body_controller import torque_stance_leg_controller

from locomotion.robots import a1
from locomotion.robots import a1_robot
from locomotion.robots import robot_config
from locomotion.robots.gamepad import gamepad_reader

import matplotlib.pyplot as plt

import scipy.integrate as integrate

from discordwebhook import Discord

flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("use_gamepad", False,
                  "whether to use gamepad to provide control input.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 1., "maximum time to run the robot.")
FLAGS = flags.FLAGS

_NUM_SIMULATION_ITERATION_STEPS = 300
_MAX_TIME_SECONDS = 30.

_STANCE_DURATION_SECONDS = [
                               0.3
                           ] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# Standing
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Tripod
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)

# maximum speed values:
max_x = 0.6
max_y = 0.4
max_rot = 0.8




def _setup_controller(robot):
    """Demonstrates how to create a locomotion controller."""
    desired_speed = (0, 0)
    desired_twisting_speed = 0

    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=_STANCE_DURATION_SECONDS,
        duty_factor=_DUTY_FACTOR,
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE)
    window_size = 20 if not FLAGS.use_real_robot else 60
    state_estimator = com_velocity_estimator.COMVelocityEstimator(
        robot, window_size=window_size)
    sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_height=robot.MPC_BODY_HEIGHT,
        foot_clearance=0.01)

    st_controller = torque_stance_leg_controller.TorqueStanceLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=robot.MPC_BODY_HEIGHT)

    controller = locomotion_controller.LocomotionController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        swing_leg_controller=sw_controller,
        stance_leg_controller=st_controller,
        clock=robot.GetTimeSinceReset)
    return controller


def _update_controller_params(controller, lin_speed, ang_speed):
    controller.swing_leg_controller.desired_speed = lin_speed
    controller.swing_leg_controller.desired_twisting_speed = ang_speed
    controller.stance_leg_controller.desired_speed = lin_speed
    controller.stance_leg_controller.desired_twisting_speed = ang_speed


def main(argv):
    """Runs the locomotion controller example."""

    del argv  # unused

    # create folder to store results
    if FLAGS.logdir:
        logdir = os.path.join(FLAGS.logdir,
                              datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(logdir)


    nr_of_seeds = 3
    #defines the desired velocities; x,y,rotation velocitiy, time hhow long to follow this vector; after this time the model follows the next vector in the list
    desired_vel_lst_lst = [[[1, 0, 0, None]], [[-1, 0, 0, None]], [[0, 1, 0, None]], [[0, -1, 0, None]],
                           [[1, 1, 0, None]],[[1, -1, 0, None]],[[-1, 1, 0, None]],[[-1, -1, 0, None]]]

    split_points = [0]
    states = [list() for i in range(38)]
    actions_pos = []
    actions_torque = []
    for j in range(len(desired_vel_lst_lst)):
        print("Starting simulation of ", j + 1, '/', len(desired_vel_lst_lst))
        for i in range(nr_of_seeds):
            print("    direction ", j + 1, '/', len(desired_vel_lst_lst))
            print("    with seed=" + str(i), "; desired_vel_lst=", desired_vel_lst_lst[j])
            states_temp, action_pos_temp, actions_torque_temp = execute_controller(
                desired_vel_lst=desired_vel_lst_lst[j], seed=i)
            assert len(states_temp) == len(states), "has to be the same length/obs spec"
            split_points.append(len(states_temp[0]) + split_points[-1])
            states = [states[k] + states_temp[k] for k in range(len(states))]
            actions_pos += action_pos_temp
            actions_torque += actions_torque_temp

    # store one long dataset
    if FLAGS.logdir:
        np.savez(os.path.join(logdir, 'actions_position.npz'), action=actions_pos)
        np.savez(os.path.join(logdir, 'actions_torque.npz'), action=actions_torque)
        np.savez(os.path.join(logdir, 'states.npz'),  # rpy
                 q_trunk_tx=np.array(states[0]), q_trunk_ty=np.array(states[1]), q_trunk_tz=np.array(states[2]),
                 q_trunk_rotation=np.array(states[5]), q_trunk_list=np.array(states[3]),
                 q_trunk_tilt=np.array(states[4]),
                 q_FR_hip_joint=np.array(states[6]), q_FR_thigh_joint=np.array(states[7]),
                 q_FR_calf_joint=np.array(states[8]),
                 q_FL_hip_joint=np.array(states[9]), q_FL_thigh_joint=np.array(states[10]),
                 q_FL_calf_joint=np.array(states[11]),
                 q_RR_hip_joint=np.array(states[12]), q_RR_thigh_joint=np.array(states[13]),
                 q_RR_calf_joint=np.array(states[14]),
                 q_RL_hip_joint=np.array(states[15]), q_RL_thigh_joint=np.array(states[16]),
                 q_RL_calf_joint=np.array(states[17]),
                 dq_trunk_tx=np.array(states[18]), dq_trunk_ty=np.array(states[19]),
                 dq_trunk_tz=np.array(states[20]),
                 dq_trunk_rotation=np.array(states[23]), dq_trunk_list=np.array(states[21]),
                 dq_trunk_tilt=np.array(states[22]),
                 dq_FR_hip_joint=np.array(states[24]), dq_FR_thigh_joint=np.array(states[25]),
                 dq_FR_calf_joint=np.array(states[26]),
                 dq_FL_hip_joint=np.array(states[27]), dq_FL_thigh_joint=np.array(states[28]),
                 dq_FL_calf_joint=np.array(states[29]),
                 dq_RR_hip_joint=np.array(states[30]), dq_RR_thigh_joint=np.array(states[31]),
                 dq_RR_calf_joint=np.array(states[32]),
                 dq_RL_hip_joint=np.array(states[33]), dq_RL_thigh_joint=np.array(states[34]),
                 dq_RL_calf_joint=np.array(states[35]),
                 dir_arrow=np.array(states[36]), goal_speed=np.array(states[37]),
                 split_points=np.array(split_points))
        logging.info("logged to: {}".format(logdir))




def follow_trajectory(pos_desired_last, pos, vel_desired, vel, pos_errors, dt, angle):
    """
    calculates the next desired point and the required velocities to reach this point
    """
    # controller params
    kp = [200, 100, 100]
    kd = [10, 20, 10]
    ki = [20, 30, 1]

    # next desired position
    pos_desired = pos_desired_last + dt * 0.46 * vel_desired  # 0.53

    pos_error = pos_desired - pos
    # handle rotation error in corresponding shorter direction
    pos_error[2] = ((pos_error[2] + np.pi) % (2 * np.pi) - np.pi)
    # append the new error to the list for ki
    pos_errors.append(pos_error)

    u = kp * (pos_error) + kd * (vel_desired - vel) + ki * np.trapz(pos_errors, dx=dt,
                                                                    axis=0)

    # rotate resulting corrections corresponding to robots orientation else it does the correction calculated in the absolute coordinate sys from its own perspective
    rotated_x = np.cos(-angle) * np.array(u[0]) - np.sin(-angle) * np.array(u[1])
    rotated_y = np.sin(-angle) * np.array(u[0]) + np.cos(-angle) * np.array(u[1])

    # calc the maximum velocities in x,y dir corresponding to the values from the kp-controller
    alpha = np.arctan2(rotated_y, rotated_x)
    L_max = np.sqrt(np.square(np.cos(alpha) * max_x) + np.square(np.sin(alpha) * max_y))
    max_vel = np.array([np.cos(alpha) * L_max, np.sin(alpha) * L_max])

    # clip the resulted velocities
    u[0] = np.clip(rotated_x, -max_vel[0],
                   max_vel[0])  # np.clip(rotated_x, -max_vel[0], max_vel[0]) #np.clip(u[0], -max_x, max_x) #
    u[1] = np.clip(rotated_y, -max_vel[1],
                   max_vel[1])  # np.clip(rotated_y, -max_vel[1], max_vel[1]) #np.clip(u[1], -max_y, max_y) #
    u[2] = np.clip(u[2], -max_rot, max_rot)

    return u, pos_desired, pos_errors


def execute_controller(desired_vel_lst, seed=0):
    # Construct simulator
    if FLAGS.show_gui and not FLAGS.use_real_robot:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane_big.urdf")

    # Construct robot class:
    if FLAGS.use_real_robot:
        robot = a1_robot.A1Robot(
            pybullet_client=p,
            motor_control_mode=robot_config.MotorControlMode.HYBRID,
            enable_action_interpolation=False,
            time_step=0.002,
            action_repeat=1)
    else:
        robot = a1.A1(p,
                      motor_control_mode=robot_config.MotorControlMode.HYBRID,
                      enable_action_interpolation=False,
                      reset_time=2,
                      time_step=0.002,
                      action_repeat=1)

    controller = _setup_controller(robot)

    controller.reset()
    if FLAGS.use_gamepad:
        raise NotImplementedError("Isn't implemented since we only need the model for data generation")

    start_time = robot.GetTimeSinceReset()
    current_time = start_time

    #for logging
    actions_pos = []
    actions_torque = []
    states = [list() for i in range(38)]
    des_vels = [[], [], []]

    # for normalization of the actions
    high = np.array(
        [0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917
            , 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297])
    low = np.array(
        [-0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917
            , -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433, ])
    norm_act_mean_pos = (high + low) / 2.0
    norm_act_delta_pos = (high - low) / 2.0
    high = np.array([34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34])
    low = np.array([-34, -34, -34, -34, -34, -34, -34, -34, -34, -34, -34, -34])
    norm_act_mean_torque = (high + low) / 2.0
    norm_act_delta_torque = (high - low) / 2.0

    #random seed of the noise
    np.random.seed(seed)

    # for plotting
    set_point = [[], [], []]
    act_point = [[], [], []]

    desired_vel_counter = 0
    # for plotting/finetuning
    velocities = [[], [], []]
    lin_speed = [[], [], []]

    # next sub position/ intermediate step; need for direction changes
    sub_pos_desired = np.append(robot.GetBasePosition()[:2], robot.GetTrueBaseRollPitchYaw()[2])
    pos_errors = [[0, 0, 0]]
    # position where to start
    init_pos = np.append(robot.GetBasePosition()[:2], robot.GetTrueBaseRollPitchYaw()[2])


    while current_time - start_time < FLAGS.max_time_secs:

        start_time_robot = current_time
        start_time_wall = time.time()
        # Updates the controller behavior parameters. -> from the original implementation
        # lin_speed, ang_speed, e_stop = command_function(t=current_time, vx=vx, vy=vy, wz=wz)

        # get current state
        angle = robot.GetTrueBaseRollPitchYaw()[2]
        x, y = robot.GetBasePosition()[:2]
        act_pos = np.array([x, y, ((angle + np.pi) % (2 * np.pi) - np.pi)])
        vx, vy = robot.GetBaseVelocity()[:2]
        v_angle = robot.GetTrueBaseRollPitchYawRate()[2]
        act_vel = np.array([vx, vy, v_angle])

        # if speed vec is executed long enough
        if current_time == desired_vel_lst[desired_vel_counter][3]:
            if (desired_vel_counter < len(desired_vel_lst)):
                desired_vel_counter += 1
                init_pos = sub_pos_desired
            else:
                break

        # for plotting/finetuning
        set_point[0].append(sub_pos_desired[0])
        set_point[1].append(sub_pos_desired[1])
        set_point[2].append(sub_pos_desired[2])
        act_point[0].append(act_pos[0])
        act_point[1].append(act_pos[1])
        act_point[2].append(act_pos[2])

        dt = robot.time_step
        speed_vec = np.array(desired_vel_lst[desired_vel_counter])
        # .85 as a range between wanted velo and maxium possible velo
        vel_desired_angle = np.clip(speed_vec[2], -0.85 * max_rot, 0.85 * max_rot)
        # calculates the desired velocity depending on the maximum velocities in x,y,rot out of the input speed_vec
        alpha = np.arctan2(speed_vec[1], speed_vec[0])
        # angle to consider the maximum velo the robot is able to walk in this direction
        if vel_desired_angle != 0:
            # if the vel_desired_angle is not 0 we want the robot to turn, so we consider the actual trunk rotation
            alpha_rot = alpha - angle
        else:
            # if the vel_desired_angle is 0 we dont want the robot to turn; so we only substract the initial rotation of the trunk
            alpha_rot = alpha - init_pos[2]
        # max length of the desired vel vector depending on the maxima and the direction where we want to walk in
        L_max = np.sqrt(np.square(np.cos(alpha_rot) * max_x * 0.85) + np.square(np.sin(alpha_rot) * max_y * 0.85))
        # if speed_vec = [0,0] desired velocity should be 0 instead of max
        vel_desired_x = (np.cos(alpha) * L_max) if speed_vec[:2].any() else 0
        vel_desired_y = (np.sin(alpha) * L_max) if speed_vec[:2].any() else 0
        # for calcuation of the next sub point
        vel_desired = np.array([vel_desired_x, vel_desired_y, vel_desired_angle])

        #calculate the new velocitiy vectors to reach the new desired subpoint
        u, sub_pos_desired, pos_errors, = follow_trajectory(pos=act_pos, vel_desired=vel_desired, vel=act_vel, dt=dt,
                                                            pos_errors=pos_errors, angle=angle,
                                                            pos_desired_last=sub_pos_desired)

        # parameters for the controller
        lin_speed[:2] = u[:2]
        ang_speed = u[2]
        lin_speed[2] = 0  # z velocity is always 0
        # for plotting/finetuning
        velocities[0].append(lin_speed[0])
        velocities[1].append(lin_speed[1])
        velocities[2].append(ang_speed)
        des_vels[0].append(vel_desired_x)
        des_vels[1].append(vel_desired_y)
        des_vels[2].append(vel_desired_angle)

        #update the controller
        _update_controller_params(controller, lin_speed, ang_speed)
        controller.update()
        hybrid_action, info = controller.get_action()  # time consuming

        # add noise if neccesary
        noise = np.random.rand(60) * 0.1
        if seed != 0:
            hybrid_action += noise

        # collect states data
        temp = robot.GetBasePosition()
        temp = np.append(temp, robot.GetTrueBaseRollPitchYaw())
        temp = np.append(temp, robot.GetTrueMotorAngles())
        temp = np.append(temp, robot.GetBaseVelocity())
        temp = np.append(temp, robot.GetTrueBaseRollPitchYawRate())
        temp = np.append(temp, robot.GetTrueMotorVelocities())
        #different z coordinate systems in MPC and mujoco
        temp[2] -= 0.43
        for i in np.arange(len(states) - 2): # last state will be appended later
            states[i].append(temp[i])
        # collect actions - position
        actions_pos.append((robot.GetTrueMotorAngles() - norm_act_mean_pos) / norm_act_delta_pos)
        # collect actions - torques
        actions_torque.append((robot.GetTrueMotorTorques() - norm_act_mean_torque) / norm_act_delta_torque)

        # store direction arrow/matrix
        # calc direction from the point of view of the robot
        # rotate arrow with 0 if 0 rotation is wanted else with the actual reached rotation
        wanted_rot = 0 if speed_vec[2] == 0 else angle
        vel_desired_x = np.cos(alpha - wanted_rot) * L_max if speed_vec.any() else 0
        vel_desired_y = np.sin(alpha - wanted_rot) * L_max if speed_vec.any() else 0
        vel_desired = np.array([vel_desired_x, vel_desired_y, np.clip(speed_vec[2], -max_rot, max_rot)])

        # calc angle and turn into matrix for direction arrow
        angle_goal = np.arctan2(vel_desired[1], vel_desired[0])
        R = np.array(
            [[np.cos(angle_goal), -np.sin(angle_goal), 0], [np.sin(angle_goal), np.cos(angle_goal), 0], [0, 0, 1]])
        arrow = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))
        rotation_matrix_dir = np.dot(R, arrow).reshape((9,))
        states[36].append(rotation_matrix_dir)

        # store goal velocity
        states[37].append(L_max)

        # perform step
        robot.Step(hybrid_action)
        current_time = robot.GetTimeSinceReset()
        if not FLAGS.use_real_robot:
            expected_duration = current_time - start_time_robot
            actual_duration = time.time() - start_time_wall
            if actual_duration < expected_duration:
                time.sleep(expected_duration - actual_duration)

    # Plotting ---------------------------------------------------------------------------------------------------------

    # actual vs desired velocities
    act_point = np.array(act_point)
    set_point = np.array(set_point)
    data = {
        "act_vel_x": states[18],
        "act_vel_y": states[19],
        # "act_vel_rot": states[23],
        "des_vel_x": des_vels[0],
        "des_vel_y": des_vels[1],
        # "des_vel_rot": des_vels[2]
    }
    fig = plt.figure()
    ax = fig.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
        ax.set_ylim(-0.85, 0.85)
        ax.hlines(y=0.11, xmin=0, xmax=len(v[1]), linewidth=1, color="red")
        ax.hlines(y=-0.13, xmin=0, xmax=len(v[1]), linewidth=2, color="red")
    plt.legend(loc=4)
    plt.xlabel("Time")
    plt.ylabel("Vel")
    plt.savefig("act_vel.png")

    # actual vs desired x-position
    act_point = np.array(act_point)
    set_point = np.array(set_point)
    data = {
        "setpoint": set_point[0],
        "actpoint": act_point[0]
    }
    fig = plt.figure()
    ax = fig.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.legend(loc=4)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.savefig("x.png")

    # actual vs desired y-position
    data = {
        "setpoint": set_point[1],
        "actpoint": act_point[1]
    }
    fig = plt.figure()
    ax = fig.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.legend(loc=4)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.savefig("y.png")

    # actual velocities
    data = {
        "x_vel": velocities[0],
        "y_vel": velocities[1],
        # "rot_vel": velocities[2]
    }
    fig = plt.figure()
    ax = fig.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.legend(loc=4)
    plt.xlabel("Time")
    plt.ylabel("Vel")
    plt.savefig("velocities.png")

    # actual vs desired rotation position
    data = {
        "setpoint": set_point[2],
        "actpoint": act_point[2]
    }
    fig = plt.figure()
    ax = fig.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.legend(loc=4)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.savefig("rot.png")

    # for logging
    return states, actions_pos, actions_torque


if __name__ == "__main__":
    app.run(main)

# commands to use the model/data generation
# python3 -m locomotion.examples.whole_body_controller_example --use_gamepad=False --show_gui=True --use_real_robot=False --max_time_secs=10
# python3 -m locomotion.examples.whole_body_controller_example --use_gamepad=False --show_gui=False --use_real_robot=False --max_time_secs=102.05 --logdir=log/2D_Walking
