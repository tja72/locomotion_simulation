"""Example of whole body controller on A1 robot."""
from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import numpy as np
import os
import scipy.interpolate
import time

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


def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = .6 #max is 1
  vy = 0.2
  wz = 0.8
  vz= .1

  time_points = (0, 2, 4, 6, 8, 10, 30, 40, 50, 70, 80, 100, 200, 300, 400, 500, 600)
  speed_points = ((0, 0, 0, 0), (vx, 0, vz, 0), (vx, 0, vz, 0), (vx, 0, vz, 0),
                  (vx, 0, vz, 0), (vx, 0, vz, 0), (vx, 0, vz, 0), (vx, 0, vz, 0),
                  (vx, 0, vz, 0), (vx, 0, vz, 0), (vx, 0, vz, 0), (vx, 0, vz, 0),
                  (vx, 0, vz, 0), (vx, 0, vz, 0), (vx, 0, vz, 0), (vx, 0, vz, 0), (vx, 0, vz, 0)) #x,y,z, omega (rotation)

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
                  time_step=0.002, # TODO: Need Freq ------------------------------------------------ 0.002 ------------
                  action_repeat=1)

  controller = _setup_controller(robot)

  controller.reset()
  if FLAGS.use_gamepad:
    gamepad = gamepad_reader.Gamepad()
    command_function = gamepad.get_command
  else:
    command_function = _generate_example_linear_angular_speed

  if FLAGS.logdir:
    logdir = os.path.join(FLAGS.logdir,
                          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(logdir)

  start_time = robot.GetTimeSinceReset()
  current_time = start_time
  actions_pos = []
  actions_torque = []


  # for normalization of the actions
  states = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
  high = np.array([0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917
      , 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297])
  low = np.array([-0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917
      , -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433, ])
  norm_act_mean = (high + low) / 2.0
  norm_act_delta = (high - low) / 2.0
  while current_time - start_time < FLAGS.max_time_secs:
    start_time_robot = current_time
    start_time_wall = time.time()
    # Updates the controller behavior parameters.
    lin_speed, ang_speed, e_stop = command_function(current_time)
    # print(lin_speed)
    if e_stop:
      logging.info("E-stop kicked, exiting...")
      break
    _update_controller_params(controller, lin_speed, ang_speed)
    controller.update()
    hybrid_action, info = controller.get_action() # time consuming
    noise = np.random.rand(60)*1e-4
    hybrid_action+=noise

    # collect states data
    temp = list(robot.GetBasePosition())
    temp[2] -= 0.43
    temp = temp + list(robot.GetTrueBaseRollPitchYaw())
    temp = temp + list(robot.GetTrueMotorAngles())
    temp = temp + list(robot.GetBaseVelocity())
    temp = temp + list(robot.GetTrueBaseRollPitchYawRate())
    temp = temp + list(robot.GetTrueMotorVelocities())
    for i in np.arange(len(states)):
        states[i].append(temp[i])
    #collect actions - position
    actions_pos.append((robot.GetTrueMotorAngles()-norm_act_mean)/norm_act_delta)  # TODO: ------------------------- other format, and position
    # collect actions - torques
    actions_torque.append((robot.GetTrueMotorTorques()-norm_act_mean)/norm_act_delta)

    #print('State: ', temp)

    robot.Step(hybrid_action)
    current_time = robot.GetTimeSinceReset()
    if not FLAGS.use_real_robot:
      expected_duration = current_time - start_time_robot
      actual_duration = time.time() - start_time_wall
      if actual_duration < expected_duration:
        time.sleep(expected_duration - actual_duration)

  if FLAGS.use_gamepad:
    gamepad.stop()

  if FLAGS.logdir:
    np.savez(os.path.join(logdir, 'actions_position.npz'), action=actions_pos)
    np.savez(os.path.join(logdir, 'actions_torque.npz'), action=actions_torque)
    np.savez(os.path.join(logdir, 'states.npz'), #rpy
             q_trunk_tx=np.array(states[0]), q_trunk_ty=np.array(states[1]), q_trunk_tz=np.array(states[2]),
             q_trunk_tilt=np.array(states[5]), q_trunk_list=np.array(states[3]), q_trunk_rotation=np.array(states[4]),
             q_FR_hip_joint=np.array(states[6]), q_FR_thigh_joint=np.array(states[7]), q_FR_calf_joint=np.array(states[8]),
             q_FL_hip_joint=np.array(states[9]), q_FL_thigh_joint=np.array(states[10]), q_FL_calf_joint=np.array(states[11]),
             q_RR_hip_joint=np.array(states[12]), q_RR_thigh_joint=np.array(states[13]), q_RR_calf_joint=np.array(states[14]),
             q_RL_hip_joint=np.array(states[15]), q_RL_thigh_joint=np.array(states[16]), q_RL_calf_joint=np.array(states[17]),
             dq_trunk_tx=np.array(states[18]), dq_trunk_tz=np.array(states[19]), dq_trunk_ty=np.array(states[20]),
             dq_trunk_tilt=np.array(states[21]), dq_trunk_list=np.array(states[22]), dq_trunk_rotation=np.array(states[23]),
             dq_FR_hip_joint=np.array(states[24]), dq_FR_thigh_joint=np.array(states[25]), dq_FR_calf_joint=np.array(states[26]),
             dq_FL_hip_joint=np.array(states[27]), dq_FL_thigh_joint=np.array(states[28]), dq_FL_calf_joint=np.array(states[29]),
             dq_RR_hip_joint=np.array(states[30]), dq_RR_thigh_joint=np.array(states[31]), dq_RR_calf_joint=np.array(states[32]),
             dq_RL_hip_joint=np.array(states[33]), dq_RL_thigh_joint=np.array(states[34]), dq_RL_calf_joint=np.array(states[35]))
    #pickle.dump(states, open(os.path.join(logdir, 'states.pkl'), 'wb'))
    logging.info("logged to: {}".format(logdir))


if __name__ == "__main__":
  app.run(main)



# python3 -m locomotion.examples.whole_body_controller_example --use_gamepad=False --show_gui=True --use_real_robot=False --max_time_secs=10