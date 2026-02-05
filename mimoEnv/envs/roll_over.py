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
from mimoEnv.utils import get_minimal_z_coordinate
from gymnasium import spaces
from PIL import Image

ROLL_OVER_XML = os.path.join(SCENE_DIRECTORY, "roll_over_prone_scene.xml")
""" Path to the roll over scene.

:meta hide-value:
"""

# 18.01.2026 Copied touch parameters from selfbody and multiplied all scaled by 10
TOUCH_PARAMS = {
    "scales": {
        "left_foot": 0.5,
        "right_foot": 0.5,
        "left_lower_leg": 1,
        "right_lower_leg": 1,
        "left_upper_leg": 1,
        "right_upper_leg": 1,
        "hip": 1,
        "lower_body": 1,
        "upper_body": 1,
        "head": 1,
        "left_upper_arm": 0.1,
        "left_lower_arm": 0.1,
        "right_fingers": 0.1
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}

# If set to 'True' and goals are set to 'intrinsic', renders an image of MIMo where the goal
# observation was observed in and writes the observation to disk.
TEST_INTRINSIC_GOALS_CREATION=False


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
                 touch_params=TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 actuation_model=SpringDamperModel,
                 starting_position='prone',
                 goal_function='angle',
                 # the big reward granted on success.
                 reward_success=500,  
                 # Initial State Randomization. Turned off for testing.
                 isr=True, 
                 # No Penalize: Do not penalize actions.
                 nopen=False, 
                 # Use Potential Based Reward Shaping
                 pbrs=False, 
                 # Weighting of the potential difference in PBRS.
                 # PBRS gives a very small reward signal without a high weight factor
                 # causing the model to not succeed at all.
                 pbrs_w=100, 
                 # Number of steps where MIMo does no action to stabilize mujoco.  
                 steps_after_reset=30,
                 achieved_goal_in_observation=False,
                 **kwargs):

        if starting_position not in ["prone", "supine", "alternating"]:
            msg = f"Unknown starting position '{starting_position}'. "
            msg += "Needs to be 'prone', 'supine' or 'alternating'."
            raise ValueError(msg)

        if goal_function not in ['angle', 'cos', 'intrinsic', 'intrinsic_vesti']:
            msg = f"Unknown reward function '{goal_function}'. "
            msg += "Needs to be 'angle', 'cos' or 'intrinsic' or 'intrinsic_vesti'."
            raise ValueError(msg)

        self.intrinsic_goals_created = False
        self.goal_function=goal_function
        self.reward_success=reward_success
        self.isr=isr
        self.nopen=nopen
        self.pbrs=pbrs
        self.pbrs_w=pbrs_w
        self.steps_after_reset=steps_after_reset

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
                         achieved_goal_in_observation=achieved_goal_in_observation,
                         **kwargs)

        self.init_position=self.data.qpos.copy()
        self.put_in_starting_position()

        # 07.01.2026 The potential of the last state. Used in PBRS (Potential Based Reward Shaping) reward
        # function. Set in overloaded function 'step(...)' before calling the parent 'step', i.e. before
        # performing the environment dynamics to receive the next state. Reset to 0 on environment reset.
        self.pbrs_last_state_potential=0

        if self.goal_function == 'intrinsic':
            print("Creating prone and supine intrinsic goals.")
            self.create_prone_and_supine_intrinsic_goal()
        self.intrinsic_goals_created = True

    def create_prone_and_supine_intrinsic_goal(self):
        # Once for the current starting position and once for the opposite starting
        # position.
        for _ in range(2):
            self.reset_model()

            # Get a goalless observation to use as 'desired_goal'.
            obs = self.get_achieved_goal()
            
            if self.starting_position == 'prone':
                if TEST_INTRINSIC_GOALS_CREATION:
                    frame = self.render()
                    img = Image.fromarray(frame)
                    img.save(os.path.join('.', 'prone_to_supine_goal.png'))
                    proprio_obs = obs['observation']
                    vesti_obs = obs['vestibular']
                    np.savez("intrinsic_goal_prone_to_supine_proprio.npz", proprio_obs)
                    np.savez("intrinsic_goal_prone_to_supine_vesti.npz", vesti_obs)

                self.prone_intrinsic_goal = obs.copy()
                self.starting_position = 'supine'

            else: # supine
                if TEST_INTRINSIC_GOALS_CREATION:
                    frame = self.render()
                    img = Image.fromarray(frame)
                    img.save(os.path.join('.', 'supine_to_prone_goal.png'))
                    proprio_obs = obs['observation']
                    vesti_obs = obs['vestibular']
                    np.savez("intrinsic_goal_supine_to_prone_proprio.npz", proprio_obs)
                    np.savez("intrinsic_goal_supine_to_prone_vesti.npz", vesti_obs)

                self.supine_intrinsic_goal = obs.copy()
                self.starting_position = 'prone'

    def is_success(self, achieved_goal, desired_goal):
        """ Did we reach our goal rotation.

        Arguments:
            achieved_goal (float): The achieved hip rotation.
            desired_goal (float): This target hip rotation.

        Returns:
            bool: If the achieved hip rotation exceeds the desired rotation.
        """
        achieved_rotation = self.get_achieved_goal_cos()
        desired_rotation = 0.95

        return achieved_rotation >= desired_rotation

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
            # euler[0] = np.random.uniform(low=-1, high=1) * np.pi / 2.0
            euler[0] = np.random.beta(a=1, b=3) * np.pi  # 0° - 180°

        if self.starting_position=='prone':
            euler[1] = np.pi / 2.0
        else:
            euler[1] = -np.pi / 2.0

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

        # 26-02-01 Increased from 1 step to 20 steps, because we have issues with
        # vestibular observation not stabilizing in the first ~20 steps.
        # Perform 20 steps with no actions to stabilize initial position.
        if self.steps_after_reset > 0:
            actions = np.zeros(self.action_space.shape)
            self._set_action(actions)
            mujoco.mj_step(self.model, self.data, nstep=self.steps_after_reset)

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
    
    def get_goal_space(self, obs_space):
        if self.goal_function == 'intrinsic_vesti':
            return spaces.Box(-np.inf, np.inf, shape=self.get_vestibular_obs().shape, dtype=np.float64)
        if self.goal_function == 'intrinsic':
            # stableBaselines is incompatible with nested dict spaces, which is quite
            # unfortunate, but this simply means that instead of directly using the
            # observation space as a goal space, we instead calculate a box space by
            # flattening the dict obs space.
            # Since all our observations are scalars, we just add up the shapes
            # of each space.
            space_flattened = 0

            for space in obs_space.values():
                space_flattened += space.shape[0]
            return spaces.Box(-np.inf, np.inf, shape=(space_flattened,), dtype=np.float64)
        else:
            return spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
        
    def get_desired_goal_obs(self):
        if self.goal_function != 'intrinsic':
            return self.goal
        
        # For intrinsic goal function, we cannot simply return the goal, since it has
        # the same shape as the observation, i.e. a dictionary. Instead, we must
        # also flatten the goal like we did in 'get_goal_space'.
        return np.concatenate([self.goal[key] for key in sorted(self.goal.keys())])

    def sample_goal(self):
        """ Returns the goal rotation.

        We use a fixed goal rotation of 0.95 [previously: 0.8]

        For intrinsic goals, we sample a goal in the reset() function.

        Returns:
            np.array[float]: [0.95]
        """
        if self.goal_function == "intrinsic" or self.goal_function == 'intrinsic_vesti':
            # We initialize intrinsic goals at the very end of the constructor of this environment.
            # In between, we do get observation calls that call this function 'sample_goal'. Since
            # the goals 'prone_intrinsic_goal' and 'supine_intrinsic_goal' are not yet created,
            # this function would return an error. Instead, we just return the current observation
            # as goal in that case.
            if self.intrinsic_goals_created:
                if self.starting_position == 'prone':
                    return self.supine_intrinsic_goal.copy()
                else:
                    return self.prone_intrinsic_goal.copy()
            else:
                return self.get_achieved_goal_intrinsic()
        
        return np.array([0.95])

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

    def get_achieved_goal_angle(self):
        """ The very first goal function - with the slight addition that I added
        chest rotation. Calculates the normalized angle of the chest and hip rotation
        and returns the average of them.

        Returns:
            np.array[float]: The average normalized angle of the chest and hip rotation.
        """
        rot_hip = self._get_standardized_rotation("hip")
        rot_chest = self._get_standardized_rotation("chest")

        # Goal rotation is exactly the opposite for supine starting position.
        if self.starting_position=="supine":
            rot_hip = 1 - rot_hip
            rot_chest = 1 - rot_chest

        return np.array([(rot_hip + rot_chest) / 2.0])

    def get_achieved_goal_cos(self):
        """ The second goal function. It is linear in xmat[2,0], i.e. in the
        dot product between MIMo's local x axis of hip and chest and the global
        z axis. This is the same as the negative cosine of the angle between that
        local x axis and the global z axis over a rotation over the global x axis.

        For supine position, this is instead the positive cosine as our target vector
        is the negative global z axis.

        Returns:
            np.array[float]: Average of dot product of local x axis of chest and hip
                and the global z axis.
        """
        rot_hip = self.get_vertical_component_of_local_x_axis("hip")
        rot_chest = self.get_vertical_component_of_local_x_axis("chest")

        # Goal rotation is exactly the opposite for supine starting position.
        if self.starting_position=="supine":
            rot_hip *= -1
            rot_chest *= -1

        rot_hip = (rot_hip + 1) / 2.0
        rot_chest = (rot_chest + 1) / 2.0
        return np.array([(rot_hip + rot_chest) / 2.0])
    
    def get_achieved_goal_intrinsic(self):
        return self._get_obs(without_goals=True)
    
    def get_achieved_goal_intrinsic_vesti(self):
        return self.get_vestibular_obs()

    def get_achieved_goal(self):
        """ Returns the goal calculated from either of the tree goal functions.
        """
        if self.goal_function=='angle':
            return self.get_achieved_goal_angle()
        elif self.goal_function=='cos':
            return self.get_achieved_goal_cos()
        elif self.goal_function=='intrinsic':
            return self.get_achieved_goal_intrinsic()
        else:
            return self.get_achieved_goal_intrinsic_vesti()

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

        if self.goal_function != 'intrinsic':
            desired_goal = self.goal

            if self.is_success(achieved_goal, desired_goal):
                return self.reward_success

            return -np.linalg.norm(desired_goal - achieved_goal)
        
        # For intrinsic goal function, we have factor for each goal observation.
        goal_scaling_dict = {
            'observation': 0.1,
            'vestibular': 1.0,
            'touch': 0.4
        }
        concat_scaled_achieved_goal = np.concatenate([achieved_goal[key] * goal_scaling_dict[key] for key in sorted(achieved_goal.keys())])
        concat_desired_goal = np.concatenate([self.goal[key] * goal_scaling_dict[key] for key in sorted(self.goal.keys())])

        return -np.linalg.norm(concat_desired_goal - concat_scaled_achieved_goal)

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

        If 'pbrs=False' as an argument, we do not use Potential Based Reward
        Shaping and instead simply return the potential of the current state
        as a reward (which is the euclidean distance to the desired goal).

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
        if self.is_success(achieved_goal, desired_goal):
            return self.reward_success - quad_ctrl_cost

        # Potential of current state.
        curr_potential = self.get_potential()

        if not self.pbrs:
            return curr_potential - quad_ctrl_cost
        
        return self.pbrs_w * (curr_potential - self.pbrs_last_state_potential) - quad_ctrl_cost
    