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
                         goals_in_observation=False,
                         done_active=False,
                         **kwargs)

        self.init_position=self.data.qpos.copy()
        self.put_in_starting_position()

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

    def put_in_starting_position(self):
        """ Puts MIMo back in starting position - prone or supine.
        
        Returns:
            Nothing     
        """
        qpos = self.init_position.copy()

        # qpos[0:3] describe the location of MIMo, qPos[3:7] describe the quaternion
        # rotation.

        # Put MIMo in the correct starting rotation.
        if self.starting_position=='prone':
            # Euler rotation (in degrees) 0 90 0, so 90° rotation around y-axis.
            qpos[3] = 0.707
            qpos[4] = 0.
            qpos[5] = 0.707
            qpos[6] = 0.
        else: # supine starting position
            # Euler rotation (in degrees) 0 -90 0, so -90° rotation around y-axis.
            qpos[3] = -0.707
            qpos[4] = 0.
            qpos[5] = 0.707
            qpos[6] = 0.

        # Set initial positions stochastically.
        random = self.np_random.uniform(
            low=-0.01, high=0.01, size=len(qpos[7:])
        )
        qpos[7:] = qpos[7:] + random

        # Set initial velocities to zero.
        qvel = np.zeros(self.data.qvel.shape)

        self.set_state(qpos, qvel)

        # Perform 100 steps with no actions to stabilize initial position.
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        mujoco.mj_step(self.model, self.data, nstep=100)

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
        return self._get_obs()

    def sample_goal(self):
        """ Returns the goal rotation.

        We use a fixed goal rotation of 0.8.

        Returns:
            float: 0.8
        """
        return 0.8

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
            # direction of the global z-axis.
            # We define a roll over prone-to-supine so that the local x-axis looks up, i.e. exactly in the direction
            # of the global z-axis. So we want to measure the angle between the global z-axis and the local x-axis.

        # To do this, we calculate the pitch angle. MIMos local z-axis now goes along the global x-axis and we want
        # that axis 'to stay the same'. He should roll around that z-axis and not stand up.
        # Calculate the euler angle of the y-axis.
        angle_in_radiants = np.arctan2(
            -xmat[2, 0],
            np.sqrt(xmat[2, 1] ** 2 + xmat[2, 2] ** 2)
        )
        angle_in_degrees = angle_in_radiants * (180 / np.pi)

        # Normalize the angle to [0, 1].
        angle_norm = (angle_in_degrees - (-90)) / (90 - (-90))

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

    def get_reward_winkel(self):
        """ The very first goal function - with the slight addition that I added
        chest rotation. Calculates the normalized angle of the chest and hip rotation
        and returns the average of them.

        Returns:
            float: The average normalized angle of the chest and hip rotation.
        """
        rot_hip = self._get_standardized_rotation("hip")
        rot_chest = self._get_standardized_rotation("chest")

        # Goal rotation is exactly the opposite for prone starting position.
        if self.starting_position=="prone":
            rot_hip = 1 - rot_hip
            rot_chest = 1 - rot_chest

        return (rot_hip + rot_chest) / 2.0

    def get_reward_linear(self):
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

    def get_reward_quad(self):
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
            return self.get_reward_winkel()
        elif self.reward_function=='linear':
            return self.get_reward_linear()
        else:  # 'quad'
            return self.get_reward_quad()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Computes the reward.

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
