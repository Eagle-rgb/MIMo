"""
This module contains a simple experiment in which MIMo tries to roll over.

MIMo starts either in prone or supine position. This can be adjusted below.
The task is to roll over to the opposite position.

The scene consists only of MIMo. His head is fixed.
Sensory input consists of proprioceptive and vestibular inputs,
using the default configurations for both.

MIMo initial position is determined by slightly randomizing all joint
positions from a standing position and then letting the simulation settle.
This leads to MIMo being in a slightly random prone or supine position each
episode. All episodes have a fixed length, there are no goal or failure states.

Reward shaping is employed, such that MIMo is penalized for using muscle
inputs and large inputs in particular. Additionally, he is rewarded each step
for the current rotation of his hip.

The class with the env is :class:`~mimoEnv.envs.standup.MIMoRollOverEnv` while
the path to the scene XML is defined in :data:`ROLL_OVER_XML`.
"""

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, \
    DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS
from mimoActuation.actuation import SpringDamperModel
import mujoco
import numpy as np
import os
from utils import align_to_floor, get_minimal_z_coordinate

STARTING_POSITION = "supine"
""" Initial position of MIMo. Can be 'prone' or 'supine'.

:meta hide-value:
"""

ROLL_OVER_XML = os.path.join(SCENE_DIRECTORY, "roll_over_prone_scene.xml")
""" Path to the roll over scene.

:meta hide-value:
"""


class MIMoRollOverEnv(MIMoEnv):
    """
    MIMo learns to roll over from prone or supine position.

    Attributes and parameters are the same as in the base class, but the
    default arguments are adapted for the scenario. Specifically we have
    :attr:`.done_active` and :attr:`.goals_in_observation` as ``False`` and
    touch and vision sensors disabled.

    Even though we define a success condition in :meth:
    `~mimoEnv.envs.standup.MIMoStandupEnv._is_success`, it is disabled since
    :attr:`.done_active` is set to ``False``. The purpose of this is to enable
    extra information for the logging features of stable baselines.

    Attributes:
        init_position (numpy.ndarray): The initial position.
    """

    def __init__(self,
                 model_path=ROLL_OVER_XML,
                 initial_qpos=None,
                 frame_skip=2,
                 age=None,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 actuation_model=SpringDamperModel,
                 starting_position='prone',
                 reward_function='linear',
                 reward_success=500,  # the big reward granted on success.
                 isr=True, # Initial State Randomization. Turned off for testing.
                 nopen=False, # No Penalize: Do not penalize actions.
                 potsq=False, # Use squared euclidean potentials in PBRS instead of
                    # simple euclidean distance from achieved_goal to desired_goal.
                 **kwargs):

        if starting_position not in ["prone", "supine", "alternating"]:
            msg = f"Unknown starting position '{starting_position}'. "
            msg += "Needs to be 'prone', 'supine' or 'alternating'."
            raise ValueError(msg)

        if reward_function not in ['winkel', 'linear', 'quad']:
            msg = f"Unknown reward function '{reward_function}'. "
            msg += "Needs to be 'winkel', 'linear' or 'quad'."
            raise ValueError(msg)

        self.reward_function=reward_function
        self.reward_success=reward_success
        self.isr=isr
        self.nopen=nopen
        self.potsq=potsq

        self.starting_position=starting_position
        self.alternating_starting_position=self.starting_position=='alternating'
        if self.alternating_starting_position:
            self.starting_position='prone'  # start in 'prone' starting position and alternate from there.

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         age=age,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=True,
                         done_active=True,
                         **kwargs)

        self.init_position=self.data.qpos.copy()
        self.put_in_starting_position()

        # 07.01.2026 The potential of the last state. Used in PBRS (Potential Based Reward Shaping) reward
        # function. Set in overloaded function 'step(...)' before calling the parent 'step', i.e. before
        # performing the environment dynamics to receive the next state. Reset to 0 on environment reset.
        self.pbrs_last_state_potential=0

    def is_success(self, achieved_goal, desired_goal):
        """ Did we reach our goal rotation.

        Arguments:
            achieved_goal (float): The achieved hip rotation.
            desired_goal (float): This target hip rotation.

        Returns:
            bool: If the achieved hip rotation exceeds the desired rotation.
        """

        success = (achieved_goal >= desired_goal)

        return success

    def is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns ``False``.

        Arguments:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``
        """
        return False

    def is_truncated(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def get_starting_quat(self):
        """ Returns the starting rotation of MIMo as quaternion.

        Returns angles (x,y,z) as quaternion with
        - x: Random between -90° and +90° if initial state randomization is
            active, else 0.
        - y: 90° if prone, else -90° if supine.
        - z: 0°
        """
        euler = np.zeros(3)
        if self.isr:
            euler[0] = np.random.uniform(low=-1, high=1) * np.pi / 2.0

        if self.starting_position=='prone':
            euler[1] = np.pi / 2.0
        else:
            euler[1] = np.pi / 2.0

        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, euler, 'xyz')
        return quat

    def put_in_starting_position(self):
        """ Puts MIMo back in starting position - prone or supine.
        
        Returns:
            Nothing     
        """
        qpos = self.init_position.copy()

        # qpos[0:3] describe the location of MIMo, qPos[3:7] describe the quaternion
        # rotation.

        # Set initial positions stochastically.
        random = self.np_random.uniform(
            low=-0.01, high=0.01, size=len(qpos[7:])
        )
        qpos[7:] = qpos[7:] + random

        # Set initial rotation.
        qpos[3:7] = self.get_starting_quat()

        # Align MIMo to the floor by calculating minimal z coordinate of
        # all geometries and offsetting MIMo by that negated amount.
        self.data.qpos = qpos
        mujoco.mj_forward(self.model, self.data)
        min_z = get_minimal_z_coordinate(self.model, self.data)
        self.data.qpos[2] += min_z
        self.data.qpos[2] += 0.001 # security offset.

        # Set initial velocities to zero.
        qvel = np.zeros(self.data.qvel.shape)

        self.set_state(self.data.qpos, qvel)

        # Perform 1 step with no actions to stabilize initial position.
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        mujoco.mj_step(self.model, self.data, nstep=1)

    def reset_model(self):
        """ Resets the simulation.

        Return the simulation to the XML state, then slightly randomize all
        joint positions. Afterwards we let the simulation settle for a fixed
        number of steps. This leads to MIMo settling into a slightly random
        prone or supine position.

        Returns:
            Dict: Observations after reset.
        """
                # Alternate starting position if that setting is enabled.
        if self.alternating_starting_position:
            if self.starting_position=='prone':
                self.starting_position='supine'
            else:
                self.starting_position='prone'

        # self.set_state(self.init_qpos, self.init_qvel)
        self.put_in_starting_position()
        self.pbrs_last_state_potential=0
        return self._get_obs()

    def sample_goal(self):
        """ Returns the goal rotation.

        We use a fixed goal rotation of 0.8.

        Returns:
            float: 0.8
        """
        return 0.95

    def _get_standardized_rotation(self, body_name):
        """ Get the standardized rotation of a body specified by name.

        Arguments:
            body_name (str): Name of the body to get the rotation for.
                We usually only use 'hip' or 'chest' here.

        Returns:
            float: The standardized rotation of the body.
        """
        # Get the rotation matrix of the body.
        xmat = self.data.body(body_name).xmat.reshape(3, 3)
        R_20 = xmat[2, 0]  # Entry row 2, col 0 in the rotation matrix. This entry describes how much the local
            # x axis looks in the direction of the global z-axis. MIMo has its local axis as follows:
            # local x axis: looking back-to-stomach where he is looking at.
            # local y axis: left arm to right arm
            # local z axis: feet to head
            # So the local axis without modifications are the same as the global axis. After putting MIMo in prone
            # position, the entry xmat[2, 0] is -1: The local x axis looks down to the ground, i.e. in the negative
            # direction of the global z-axis. In supine, this value is +1.
            # We define a roll over prone-to-supine so that the local x-axis looks up, i.e. exactly in the direction
            # of the global z-axis. So we want to measure the angle between the global z-axis and the local x-axis.

        # To do this, we calculate the pitch angle. MIMos local z-axis now goes along the global x-axis and we want
        # that axis 'to stay the same'. He should roll around that z-axis and not stand up.
        # Calculate the euler angle of the y-axis.

        # np.arctan2 returns an angle \phi between -pi and +pi (including both endpoints).
        # The angle is the angle between the unit-length vector (x,y) and the x-axis. The
        # point (x,y) is defined as x=-xmat[2,0] and y chosen as normalizing factor. You will
        # always find an angle between -180 and +180 degrees that works here.
        angle_in_radiants = np.arctan2(
            -xmat[2, 0],
            np.sqrt(xmat[2, 1] ** 2 + xmat[2, 2] ** 2)
        )
        # 'angle_in_degrees' is around 90° for prone starting position and around -90° for supine starting position.
        angle_in_degrees = angle_in_radiants * (180 / np.pi)

        # We use prone as default starting position. We want 0 reward
        # at 'angle_in_degrees' 90°, 0.5 reward at 180° or -180° and 
        # 1 reward at -90°.
        if angle_in_degrees >= -90 and angle_in_degrees <= 90:
            angle_norm = 1 - (angle_in_degrees + 90) / 180
        elif angle_in_degrees < -90: # -180° to -90°
            angle_norm = 1 - (angle_in_degrees + 90) / -180
        else: # 90° to 180°
            angle_norm = (angle_in_degrees - 90) / 180

        return angle_norm

    def get_vertical_component_of_local_x_axis(self, body_name):
        """ Returns the component R[2,0] of the rotation matrix for the specified body.
        This measurement is how much the local x-axis points in the direction of the global z-axis,
        i.e. how much it points up. In the prone position, the local x axis of the hip and chest of MIMo
        point towards (-z) global axis.

        Parameters:
            body_name (str): Name of the body to get the rotation for.

        Returns:
            float: The vertical component R[2,0] of the local x-axis.
        """
        return self.data.body(body_name).xmat.reshape(3, 3)[2, 0]

    def get_achieved_goal_winkel(self):
        """ The very first goal function - with the slight addition that I added
        chest rotation. Calculates the normalized angle of the chest and hip rotation
        and returns the average of them.

        Returns:
            float: The average normalized angle of the chest and hip rotation.
        """
        rot_hip = self._get_standardized_rotation("hip")
        rot_chest = self._get_standardized_rotation("chest")

        # Goal rotation is exactly the opposite for supine starting position.
        if self.starting_position=="supine":
            rot_hip = 1 - rot_hip
            rot_chest = 1 - rot_chest

        return (rot_hip + rot_chest) / 2.0

    def get_achieved_goal_linear(self):
        """ The second goal function - it is linear in xmat[2,0], i.e. the vertical
        component of the local x axis of the hip and the chest. Here also, the values
        are averaged.

        Returns:
            float: The average vertical component of the local x axis of the hip and chest.
        """
        rot_hip = self.get_vertical_component_of_local_x_axis("hip")
        rot_chest = self.get_vertical_component_of_local_x_axis("chest")

        # Goal rotation is exactly the opposite for supine starting position.
        if self.starting_position=="supine":
            rot_hip *= -1
            rot_chest *= -1

        rot_hip = (rot_hip + 1) / 2.0
        rot_chest = (rot_chest + 1) / 2.0
        return (rot_hip + rot_chest) / 2.0

    def get_achieved_goal_quad(self):
        """ The third goal function - it is quadratic in xmat[2,0], i.e. the vertical
        component of the local x axis of the hip and the chest. Here also, the values
        are averaged.

        The idea is the following: We want to accelerate learning and penalize values
        that are very far from the desired rotation. For exmple, the value (R[2,0]+1) / 2)
        is squared to accelerate learning, because it punishes MIMo much more severely if 
        he is far away from the  desired rotation. For example - lying on the side
        would already grant 0.5 reward, which now only results 0.25.

        Returns:
            float: The average vertical component of the local x axis of the hip and chest.
        """
        rot_hip = self.get_vertical_component_of_local_x_axis("hip")
        rot_chest = self.get_vertical_component_of_local_x_axis("chest")

        # Goal rotation is exactly the opposite for supine starting position.
        if self.starting_position=="supine":
            rot_hip *= -1
            rot_chest *= -1

        rot_hip = (rot_hip + 1) / 2.0
        rot_chest = (rot_chest + 1) / 2.0
        rot_hip **= 2.0
        rot_chest **= 2.0
        return (rot_hip + rot_chest) / 2.0

    def get_achieved_goal(self):
        """ Returns the goal calculated from either of the tree goal functions.
        """
        if self.reward_function=='winkel':
            return self.get_achieved_goal_winkel()
        elif self.reward_function=='linear':
            return self.get_achieved_goal_linear()
        else:  # 'quad'
            return self.get_achieved_goal_quad()

    def get_potential(self):
        """ Returns the potential of the current state.

        The potential of the current state is the euclidean distance between the desired
        goal and the achieved goal.
        See [https://arxiv.org/pdf/2201.08299 Goal-Conditioned Reinforcement Learning:
        Problems and Solutions by Liu et al. 2022 pp2-3 section 'Sample Efficiency:
        Towards Sparse Rewards'] for a discussion and the motivation behind this
        lazy approach.
          We handle reaching the goal state in the reward function. The potential of
        a goal state is 0.
        """
        achieved_goal = self.get_achieved_goal()
        desired_goal = self.sample_goal()
        if achieved_goal >= desired_goal:
            return 0

        if self.potsq:
            return -np.linalg.norm(desired_goal. achieved_goal)**2.0
        else:
            return -np.linalg.norm(desired_goal, achieved_goal)

    def step(self, action):
        """ Run one timestep of the environment's dynamics.

        This overloaded function from mimo_env is used for PBRS (Potential Based Reward Shaping) to cache the
        potential of the current state for the reward function to use it in PBRS.
        """
        # Cache potential of current state to be used in calculating next reward.
        self.pbrs_last_state_potential = self.get_potential()
        return super().step(action)

    def compute_reward_v1(self, achieved_goal, desired_goal, info):
        """
        Computes the reward.

        Note this function is archived. We now only want to return a negative
        reward when the goal is not reached.

        The reward consists of the standardized hip and chest rotation with a
        penalty of the square of the control signal.

        Before, we were only using the hip rotation, but this led to MIMo only
        turning the hip and not the chest - not performing a full roll over.
        
        Arguments:
            achieved_goal (float): The achieved hip and chest rotation.
            desired_goal (float): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """

        # Use the hip and chest rotation as the main reward.
        reward = achieved_goal  # [0, 1]

        # Penalize excessive use of force.
        quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()  # [0, 0.44]
        reward -= quad_ctrl_cost

        return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward.

        The reward is a shaped reward consisting of the sum of the sparse high
        positive reward for reaching the goal and the difference of the potential
        of the current state to the last state potential as a potential-based
        shaping function.
          See "Policy Invariance under reward transformations: Theory and
        application to reward shaping" [Ng et. al 1999] for a definition of
        PBRS and the potential-based shaping function.
          Additionally, we subtract the square of the control signal from the
        reward to discourage excessive muscle usage.

        Arguments:
            achieved_goal (float): The achieved hip and chest rotation.
            desired_goal (float): The desired hip and chest rotation so as to
                classify this trajectory as a successfull roll over.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        # Penalize excessive use of force unless disabled by '--nopen' argument.
        if not self.nopen:
            quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()  # [0, 0.44]
        else:
            quad_ctrl_cost = 0

        # If the goal is reached, give a very high positive reward.
        if achieved_goal >= desired_goal:
            return self.reward_success - quad_ctrl_cost

        # Potential of current state.
        curr_potential = self.get_potential()

        # Weight of potential. PBRS gives a very small reward signal without a high
        # weight factor causing the model to not succeed at all.
        w_potential = 100

        return w_potential * (curr_potential - self.pbrs_last_state_potential) - quad_ctrl_cost
    