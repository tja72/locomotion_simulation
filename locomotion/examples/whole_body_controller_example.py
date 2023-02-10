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

#maximum speed values:
max_x = 0.6
max_y = 0.4
max_z = 0.8

def _generate_example_linear_angular_speed(t, vx, vy, wz):
    """Creates an example speed profile based on time for demo purpose."""
    # vx = 0#-.6 #max is 1
    # vy = 0.4
    # wz =  -0.016 #-0.016 for left side; 0.028 for right side; 0.086 for Back right; 0.0419 for Front right; -0.0319 for Front Left; -0.043 for Back Left
    # vz= .1

    time_points = (0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 70, 80, 100, 200, 300, 400)
    speed_points = ((0, 0, 0, 0),
                    (vx, vy, 0, wz), (vx, vy, 0, wz), (vx, vy, 0, wz),
                    (vx, vy, 0, wz), (vx, vy, 0, wz), (vx, vy, 0, wz),
                    (vx, vy, 0, wz), (vx, vy, 0, wz), (vx, vy, 0, wz),
                    (vx, vy, 0, wz), (vx, vy, 0, wz), (vx, vy, 0, wz),
                    (vx, vy, 0, wz), (vx, vy, 0, wz), (vx, vy, 0, wz),
                    )
    # x,y,z, omega (rotation)

    speed = scipy.interpolate.interp1d(time_points,
                                       speed_points,
                                       kind="previous",
                                       fill_value="extrapolate",
                                       axis=0)(t)

    return speed[0:3], speed[3], False


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

    #create folder to store results
    if FLAGS.logdir:
        logdir = os.path.join(FLAGS.logdir,
                              datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(logdir)




    if not FLAGS.logdir:
        desired_state_lst = [[0, 0, np.pi], [1, 1, np.pi], [1, 0, np.pi/2], [0, 0, np.pi]]
        desired_state_lst = [[50, 0, 0]]#, [1, 1, 0]]

        execute_controller(desired_state_lst=desired_state_lst)
    else:
        discord = Discord(url="https://discordapp.com/api/webhooks/1065410497010213004/ufF-beaSOUyU-_LbbMkThY8P1j06HyWoSSHc8DCNVkNpMVhqGPcVomKStbYKwIrGEol6")
        try:
            nr_of_seeds = 1
            #desired_state_lst_lst = [[[0,100000,0]], [[0,-100000,0]], [[100000,100000,0]], [[100000,-100000,0]], [[-100000, 100000,0]], [[-100000,-100000,0]], [[100000,0,0]],[[-100000,0,0]]]# List of list of states
            desired_state_lst_lst = [[[0,0,np.pi], [1,1,np.pi], [1,0,np.pi/2], [0,0,np.pi]]] #[[[-50, 0, 0]]] #[[[50, 0, 0]], [[-50, 0, 0]], [[0, -50, 0]], [[0, 50, 0]], [[50, 50, 0]], [[50, -50, 0]], [[-50, 50, 0]], [[-50, -50, 0]]]

            #appendix = ["_50k_left", "_50k_right", "_50k_FL", "_50k_FR", "_50k_BL", "_50k_BR", "_50k_forward", "_50k_backward"]



            # wz = -0.016  # -0.016 for left side; 0.028 for right side; 0.086 for Back right; 0.0419 for Front right; -0.0319 for Front Left; -0.043 for Back Left

            # var to contain all concatenated trajectories
            split_points = [0]
            states = [list() for i in range(38)]
            actions_pos=[]
            actions_torque=[]
            for j in range(len(desired_state_lst_lst)):
                print("Starting simulation of ", j+1, '/', len(desired_state_lst_lst))
                for i in range(nr_of_seeds):
                    print("    direction ", j+1, '/', len(desired_state_lst_lst))
                    print("    with seed=" + str(i), "; desired_state_lst=", desired_state_lst_lst[j])
                    states_temp, action_pos_temp, actions_torque_temp = execute_controller(desired_state_lst=desired_state_lst_lst[j], seed=i)
                    assert len(states_temp) == len(states), "has to be the same length/obs spec"
                    split_points.append(len(states_temp[0])+split_points[-1])
                    states = [states[k] + states_temp[k] for k in range(len(states))]
                    # [states[k].append(states_temp[k]) for k in range(len(states))]
                    actions_pos += action_pos_temp
                    actions_torque += actions_torque_temp


            # store one long dataset
            if FLAGS.logdir:

                np.savez(os.path.join(logdir, 'actions_position.npz'), action=actions_pos)
                np.savez(os.path.join(logdir, 'actions_torque.npz'), action=actions_torque)
                np.savez(os.path.join(logdir, 'states.npz'),  # rpy
                         q_trunk_tx=np.array(states[0]), q_trunk_ty=np.array(states[1]), q_trunk_tz=np.array(states[2]),
                         q_trunk_tilt=np.array(states[5]), q_trunk_list=np.array(states[3]),
                         q_trunk_rotation=np.array(states[4]),
                         q_FR_hip_joint=np.array(states[6]), q_FR_thigh_joint=np.array(states[7]),
                         q_FR_calf_joint=np.array(states[8]),
                         q_FL_hip_joint=np.array(states[9]), q_FL_thigh_joint=np.array(states[10]),
                         q_FL_calf_joint=np.array(states[11]),
                         q_RR_hip_joint=np.array(states[12]), q_RR_thigh_joint=np.array(states[13]),
                         q_RR_calf_joint=np.array(states[14]),
                         q_RL_hip_joint=np.array(states[15]), q_RL_thigh_joint=np.array(states[16]),
                         q_RL_calf_joint=np.array(states[17]),
                         dq_trunk_tx=np.array(states[18]), dq_trunk_tz=np.array(states[19]),
                         dq_trunk_ty=np.array(states[20]),
                         dq_trunk_tilt=np.array(states[23]), dq_trunk_list=np.array(states[21]),
                         dq_trunk_rotation=np.array(states[22]),
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
                # pickle.dump(states, open(os.path.join(logdir, 'states.pkl'), 'wb'))
                logging.info("logged to: {}".format(logdir))





            message = "successfully finished all " + str(len(desired_state_lst_lst)*nr_of_seeds) + " datasets!"
            discord.post(content=message)
        except NotImplementedError as e:
            type, value, traceback = sys.exc_info()
            message = "Exception occured: Typ " + str(type) + "Str " + str(e) + "Traceback " + str(traceback.tb_frame) + str(traceback.tb_lineno)
            discord.post(content=message)



def scale_rotate_xy_speed(dt, e1, e2, e3, angle):
    de1 = e1[-1] - e1[-2]
    de2 = e2[-1] - e2[-2]
    de3 = e3[-1] - e3[-2]

    max_x = 0.6
    max_y = 0.4

    # k controllers to reach desired_state
    kp1 = 12# if e1[-1]*12 > max_x else 20000 # 15
    kp2 = 1200000 #if e2[-1]*12 > max_y else 20000
    kp3 = 12

    ki1 = 1
    ki2 = 1
    ki3 = 10

    kd1 = 100000
    kd2 = 100000
    kd3 = 10

    #time 100.002

    temp_x = kp1 * e1[-1] + ki1 * np.trapz(e1, dx=dt) #+ kd1 * (de1 / dt)
    temp_y = kp2 * e2[-1] + ki2 * np.trapz(e2, dx=dt) #+ kd2 * (de2 / dt)

    #temp_x = kp1 *

    # needed because lin_spped is from the view of the robot -> needs to turn absolute velos in persepective ones
    rotated_x = np.cos(-angle) * np.array(temp_x) - np.sin(-angle) * np.array(temp_y)
    rotated_y = np.sin(-angle) * np.array(temp_x) + np.cos(-angle) * np.array(temp_y)
    # to keep the rato from x to y -> don't use full speed all time (if 1,1 should be reached speed 0.4, 0.4 instead of 0.6, 0.4)

    factor1 = 1 if np.abs(rotated_x) < max_x else np.abs(max_x / (rotated_x))
    factor2 = 1 if np.abs(rotated_y * factor1) < max_y else np.abs(max_y / (rotated_y * factor1))
    vx = rotated_x * factor1 * factor2
    vy = rotated_y * factor1 * factor2

    L = np.linalg.norm([rotated_x, rotated_y])
    alpha = np.arccos(rotated_x / L)
    L_max = np.sqrt(np.square(np.cos(alpha)*max_x) + np.square(np.sin(alpha)*max_y))
    #vx = np.cos(alpha) * L_max
    #vy = np.sin(alpha) * L_max


    ang_diff = ((e3[-1]+np.pi) % (2*np.pi) -np.pi)
    ang_speed = kp3 * ang_diff #+ ki3 * np.trapz(e3, dx=dt) + kd3 * (de3 / dt)

    ang_speed = np.clip(ang_speed, -0.8, 0.8)
    return vx, vy, ang_speed

def follow_trajectory(pos_desired_last, pos, vel_desired, vel, dt, pos_errors, final_pos, angle, pos_error_last): # todo weniger parameter

    kp = [120,100,170] # todo to array - finetune
    kd = [30, 25, 25]
    ki = [0.2, .3, .2]







    #next desired position
    pos_desired_temp = pos_desired_last + dt * vel_desired
    #if final desired point is nearly reached -> point is final point

    pos_desired = [
        pos_desired_temp[0] if not np.isclose(pos_desired_last[0], final_pos[0], 2*dt, dt) else final_pos[0],
        pos_desired_temp[1] if not np.isclose(pos_desired_last[1], final_pos[1], 2*dt, dt) else final_pos[1],
        pos_desired_temp[2] if not np.isclose(pos_desired_last[2], final_pos[2], 2*dt, dt) else final_pos[2]

    ]

    vel_desired = [
        vel_desired[0] if not np.isclose(pos_desired_last[0], final_pos[0], 2 * dt, dt) else 0,
        vel_desired[1] if not np.isclose(pos_desired_last[1], final_pos[1], 2 * dt, dt) else 0,
        vel_desired[2] if not np.isclose(pos_desired_last[2], final_pos[2], 2 * dt, dt) else 0
    ]




    # append next pos_error
    pos_error = pos_desired - pos
    pos_error[2] = ((pos_error[2] + np.pi) % (2 * np.pi) - np.pi)


    for i in range(len(pos_errors)):
        pos_errors[i] = np.append(pos_errors[i], pos_error[i])

    de = pos_error - pos_error_last

    #u = kp * (pos_error) + kd * (vel_desired - vel) + ki * np.trapz(pos_errors, dx=dt)
    u = kp * (pos_error) + kd * (de/dt) + ki * np.trapz(pos_errors, dx=dt)

    rotated_x = np.cos(-angle) * np.array(u[0]) - np.sin(-angle) * np.array(u[1])
    rotated_y = np.sin(-angle) * np.array(u[0]) + np.cos(-angle) * np.array(u[1])


    alpha = np.arctan2(rotated_y, rotated_x)  # np.arccos(rotated_x / L) if L != 0 else 0
    L_max = np.sqrt(np.square(np.cos(alpha) * max_x) + np.square(np.sin(alpha) * max_y))
    max_vel = np.array([np.cos(alpha) * L_max, np.sin(alpha) * L_max])

    # TODO values in u can be higher than max -> loose prop -> walks more in x than in y
    a = np.clip(rotated_x, -max_x, max_x)
    b = np.clip(rotated_y, -max_y, max_y)
    c = np.clip(rotated_x, -max_vel[0], max_vel[0])
    d = np.clip(rotated_y, -max_vel[1], max_vel[1])
    u[0] = np.clip(rotated_x, -max_vel[0], max_vel[0])
    u[1] = np.clip(rotated_y, -max_vel[1], max_vel[1])
    u[2] = np.clip(u[2], -max_z, max_z)


    return u, pos_desired, pos_errors

def execute_controller(desired_state_lst, seed=0, appendix=''):
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
        gamepad = gamepad_reader.Gamepad()
        command_function = gamepad.get_command
    else:
        command_function = _generate_example_linear_angular_speed



    start_time = robot.GetTimeSinceReset()
    current_time = start_time
    actions_pos = []
    actions_torque = []

    # for normalization of the actions
    states = [list() for i in range(38)]

    # for action normalization
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
    np.random.seed(seed)

    set_point = [[],[],[]]
    act_point = [[],[],[]]
    desired_state = desired_state_lst[0]
    last_desired_state = np.append(robot.GetBasePosition()[:2], robot.GetTrueBaseRollPitchYaw()[2])
    desired_state_counter = 0
    velocities = [[],[], []] # for plotting/fietuning
    lin_speed=[[],[],[]]
    # next sub position/ intermediate step
    sub_pos_desired = np.append(robot.GetBasePosition()[:2], robot.GetTrueBaseRollPitchYaw()[2]) # todo---------------------
    pos_errors = [[0], [0], [0]]
    pos_error_last = 0
    while current_time - start_time < FLAGS.max_time_secs:

        start_time_robot = current_time
        start_time_wall = time.time()
        # Updates the controller behavior parameters.
        #lin_speed, ang_speed, e_stop = command_function(t=current_time, vx=vx, vy=vy, wz=wz)




        angle = robot.GetTrueBaseRollPitchYaw()[2]
        x, y = robot.GetBasePosition()[:2]
        act_pos = np.array([x, y, angle])
        vx, vy = robot.GetBaseVelocity()[:2]
        v_angle = robot.GetTrueBaseRollPitchYawRate()[2]
        act_vel = np.array([vx, vy, v_angle])


        # if desired_state reached todo ----------
        e = 0.01

        if np.abs(act_pos[0]-desired_state[0])<e and np.abs(act_pos[1]-desired_state[1])<e and desired_state == [1,0,np.pi/2]:
            print('x:', np.abs(act_pos[0] - desired_state[0]))
            print('y:', np.abs(act_pos[1] - desired_state[1]))
            print('r:', np.abs(act_pos[2] - desired_state[2]))
            print("----")
        if (np.abs(act_pos[0]-desired_state[0]) < e) and (np.abs(act_pos[1]-desired_state[1])) < e and (np.abs(act_pos[2]-desired_state[2]) < e):
            desired_state_counter += 1
            if (desired_state_counter<len(desired_state_lst)):
                last_desired_state = desired_state
                desired_state = desired_state_lst[desired_state_counter]
                print(desired_state)
                #e1 = [0]
                #e2 = [0]
                #e3 = [0]
            else:
                break




        # for plotting
        set_point[0].append(desired_state[0])
        set_point[1].append(desired_state[1])
        set_point[2].append(desired_state[2])
        act_point[0].append(act_pos[0])
        act_point[1].append(act_pos[1])
        act_point[2].append(act_pos[2])

        # todo obsolete
        #e1.append(desired_state[0] - act_pos[0])
        #e2.append(desired_state[1] - act_pos[1])
        #e3.append(desired_state[2] - act_pos[2])

        dt = robot.time_step





        vel_desired = np.array([0.6, 0.0, 0.0]) # todo ellipse and maybe move into function

        rotated_x = np.cos(-angle) * np.array(desired_state[0]) - np.sin(-angle) * np.array(desired_state[1])
        rotated_y = np.sin(-angle) * np.array(desired_state[0]) + np.cos(-angle) * np.array(desired_state[1])
        rotated_x = desired_state[0] - last_desired_state[0]
        rotated_y = desired_state[1] - last_desired_state[1]
        L = np.linalg.norm([rotated_x, rotated_y])
        alpha = np.arctan2(rotated_y, rotated_x)#np.arccos(rotated_x / L) if L != 0 else 0
        L_max = np.sqrt(np.square(np.cos(alpha) * max_x) + np.square(np.sin(alpha) * max_y))
        vel_desired = np.array([np.cos(alpha) * L_max, np.sin(alpha) * L_max, np.clip(desired_state[2] - last_desired_state[2], -max_z, max_z)])



        u, sub_pos_desired, pos_errors = follow_trajectory(sub_pos_desired, act_pos,vel_desired, act_vel, dt, pos_errors, desired_state, angle, pos_error_last)
        #todo schöner machen
        pos_error_last = sub_pos_desired - act_pos


        lin_speed[:2] = u[:2]
        ang_speed = u[2]
        lin_speed[2] = 0 # z velocity is always 0
        # for plotting/finetuning
        velocities[0].append(lin_speed[0])
        velocities[1].append(lin_speed[1])
        velocities[2].append(ang_speed)
        e_stop = False








        if e_stop:
            logging.info("E-stop kicked, exiting...")
            break
        _update_controller_params(controller, lin_speed, ang_speed)
        controller.update()
        hybrid_action, info = controller.get_action()  # time consuming
        noise = np.random.rand(60) * 0.1
        if seed != 0:
            hybrid_action += noise

        # collect states data
        temp = list(robot.GetBasePosition())
        temp[2] -= 0.43
        temp = temp + list(robot.GetTrueBaseRollPitchYaw())
        temp = temp + list(robot.GetTrueMotorAngles())
        temp = temp + list(robot.GetBaseVelocity())
        temp = temp + list(robot.GetTrueBaseRollPitchYawRate())
        temp = temp + list(robot.GetTrueMotorVelocities())
        for i in np.arange(len(states)-2):
            states[i].append(temp[i])
        # collect actions - position
        actions_pos.append((robot.GetTrueMotorAngles() - norm_act_mean_pos) / norm_act_delta_pos)
        # collect actions - torques
        actions_torque.append((robot.GetTrueMotorTorques() - norm_act_mean_torque) / norm_act_delta_torque)
        # print(np.arccos(np.clip(np.dot(lin_speed[0], lin_speed[1]), -1.0, 1.0)))


        #states for direction arrow todo adapt to new function
        #start_speed_vec = scale_rotate_xy_speed(dt=dt, e1=[0, desired_state[0] - last_desired_state[0]],
         #                                       e2=[0, desired_state[1] - last_desired_state[1]],
          #                                      e3=[0, desired_state[2] - last_desired_state[2]],
           #                                     angle=last_desired_state[2])

        start_speed_vec = [0, 0,0]
        # turn angle into matrix
        angle_goal = np.arctan2(desired_state[1] - last_desired_state[1], desired_state[0] - last_desired_state[0])

        angle_goal -= last_desired_state[2] # + start_speed_vec[2]
        R = np.array([[np.cos(angle_goal), -np.sin(angle_goal), 0], [np.sin(angle_goal), np.cos(angle_goal), 0], [0, 0, 1]])
        arrow = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))
        rotation_matrix_dir = np.dot(R, arrow).reshape((9,))
        states[36].append(rotation_matrix_dir)

        # states for goal velocity; fixed per desired state
        speed = np.round(np.linalg.norm(start_speed_vec[:2]), 5)
        states[37].append(speed)


        # print('State: ', temp)
        robot.Step(hybrid_action)
        current_time = robot.GetTimeSinceReset()
        if not FLAGS.use_real_robot:
            expected_duration = current_time - start_time_robot
            actual_duration = time.time() - start_time_wall
            if actual_duration < expected_duration:
                time.sleep(expected_duration - actual_duration)


    # Plotting
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

    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
    data = {
        "x_vel": velocities[0],
        "y_vel": velocities[1],
        "rot_vel": velocities[2]
    }

    fig = plt.figure()
    ax = fig.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, v in enumerate(data.items()):
        ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
    plt.legend(loc=4)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.savefig("velocities.png")








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

    # ------------------------------------------------------------------------------------------------------------------

    if FLAGS.use_gamepad:
        gamepad.stop()

    return states, actions_pos, actions_torque





if __name__ == "__main__":
    app.run(main)

# python3 -m locomotion.examples.whole_body_controller_example --use_gamepad=False --show_gui=True --use_real_robot=False --max_time_secs=10
# python3 -m locomotion.examples.whole_body_controller_example --use_gamepad=False --show_gui=False --use_real_robot=False --max_time_secs=102.05 --logdir=log/2D_Walking
