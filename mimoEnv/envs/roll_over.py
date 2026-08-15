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
import collections
import mujoco
import numpy as np
import os
from mimoEnv.utils import get_minimal_z_coordinate
from gymnasium import spaces
from PIL import Image

AGES = [1, 3, 6, 9]

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
                 initial_qpos=None,
                 frame_skip=2,
                 age_morph=9,
                 age_physio=9,
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
                 # Penalization factor for action penalization.
                 pen_factor=0.02,
                 pca=None,
                 intrinsic_goal='all',
                 # Intrinsic goal weightings. Not-specified keys are automatically
                 # defaulted to 1.0.
                 intrinsic_goal_w={},
                 freeze_arm=False,
                 freeze_leg=False,
                 success_at_side_lying=False,
                 # Sparse {0, -1} reward instead of PBRS/distance shaping. This is the point of
                 # the HER experiments: it removes the hand-designed rotation shaping entirely.
                 sparse_reward=False,
                 # If both are set, the target rotation is sampled uniformly from
                 # [goal_low, goal_high] on every reset instead of being fixed. Needed for HER:
                 # with a constant desired_goal the policy never learns to condition on it, and
                 # relabelled transitions teach goals that are never queried.
                 goal_low=None,
                 goal_high=None,
                 # Move the upper end of the sampled goal range along with what has actually been
                 # achieved recently, instead of sampling the full [goal_low, goal_high] from the
                 # start. See ':meth:`._effective_goal_high`' for why this matters under HER.
                 goal_curriculum=False,
                 goal_curriculum_window=50,
                 goal_curriculum_quantile=0.8,
                 goal_curriculum_margin=0.1,
                 # Terminate the episode on success. Must be False for HER -- see the note in
                 # 'compute_reward'.
                 done_active=True,
                 **kwargs):

        if starting_position not in ["prone", "supine", "alternating"]:
            msg = f"Unknown starting position '{starting_position}'. "
            msg += "Needs to be 'prone', 'supine' or 'alternating'."
            raise ValueError(msg)

        if goal_function not in ['angle', 'cos', 'intrinsic']:
            msg = f"Unknown reward function '{goal_function}'. "
            msg += "Needs to be 'angle', 'cos' or 'intrinsic'."
            raise ValueError(msg)
        
        if intrinsic_goal not in ['all', 'vesti', 'vesti_acc', 'sparse_proprio']:
            msg = f"Unknown intrinsic goal '{intrinsic_goal}'. "
            msg += "Needs to be 'all', 'vesti', 'vesti_acc' or 'sparse_proprio'."
            raise ValueError(msg)
        
        # Instead of supplying 'age' as a parameter to the environment directly, we beforehand created the
        # appropriate age scene. So we manually specify the scene location.
        # This is necessary because the parallel RBI runs have problems deleting and creating the temporary
        # scenes at the same time.
        #if age == 18:  # default
        #    model_path = os.path.join(SCENE_DIRECTORY, "roll_over_prone_scene.xml")
        if age_physio in AGES and age_morph in AGES:
            model_path = os.path.join(SCENE_DIRECTORY,
                "roll_over",
                "prone",
                f"scene_act_{age_physio}_body_{age_morph}.xml")
        else:
            raise ValueError("Allowed ages: 1, 3, 6, 9")

        self.intrinsic_goals_created = False
        self.goal_function=goal_function
        self.reward_success=reward_success
        self.isr=isr
        self.nopen=nopen
        self.pbrs=pbrs
        self.pbrs_w=pbrs_w
        self.steps_after_reset=steps_after_reset
        self.pen_factor=pen_factor
        self.intrinsic_goal=intrinsic_goal
        self.success_at_side_lying=success_at_side_lying
        self.sparse_reward=sparse_reward
        self.goal_low=goal_low
        self.goal_high=goal_high

        if (goal_low is None) != (goal_high is None):
            raise ValueError("Provide both 'goal_low' and 'goal_high', or neither.")

        self.goal_curriculum=goal_curriculum
        self.goal_curriculum_quantile=goal_curriculum_quantile
        self.goal_curriculum_margin=goal_curriculum_margin

        if goal_curriculum and goal_low is None:
            raise ValueError("'goal_curriculum' needs a goal range: set 'goal_low' and "
                             "'goal_high'. The curriculum moves the upper end of that range, so "
                             "there is nothing for it to do with a fixed goal.")
        if not 0.0 <= goal_curriculum_quantile <= 1.0:
            raise ValueError("'goal_curriculum_quantile' must lie in [0, 1].")
        if goal_curriculum_margin <= 0.0:
            raise ValueError("'goal_curriculum_margin' must be positive: it is what keeps the "
                             "sampled range from degenerating to a single point, and what lets "
                             "the curriculum reach beyond what has already been achieved.")

        # Highest rotation reached during the episode that is currently running, and the same
        # quantity for the last 'goal_curriculum_window' finished episodes. Only read by
        # '_effective_goal_high'; maintained unconditionally so the statistics are available for
        # logging even when the curriculum is off.
        self._episode_max_achieved = None
        self._recent_episode_max = collections.deque(maxlen=goal_curriculum_window)

        # Achieved goal at the start of the current step. Cached so that the potential-based
        # shaping term can be recomputed from the arguments alone (it needs two consecutive
        # states), which is what makes PBRS survive HER goal relabelling.
        self._prev_achieved_goal = None
        # Initialize intrinsic goal weighting dict, i.e. add missing sensor keys.
        for key in ['observation', 'touch', 'vestibular']:
            if key not in intrinsic_goal_w:
                intrinsic_goal_w[key] = 1.0

        self.intrinsic_goal_w = intrinsic_goal_w

        self.starting_position=starting_position
        self.alternating_starting_position=self.starting_position=='alternating'
        if self.alternating_starting_position:
            self.starting_position='prone'  # start in 'prone' starting position and alternate from there.

        # DISS. Set this variable to a np.array of shape qpos[7:].shape to set a fixed
        # initial state randomization. If this variable is 'None', the initial rotation
        # is sampled uniformly between [-0.1, +0.1].
        self.deterministic_initial_state_sampling = None

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         age=None,  # we created age model scenes beforehand, we explicitly do not want to
                            # create them again.
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=True,
                         done_active=done_active,
                         achieved_goal_in_observation=achieved_goal_in_observation,
                         pca=pca,
                         freeze_arm=freeze_arm,
                         freeze_leg=freeze_leg,
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
        self.fix_top_camera_rotation_supine()

    def fix_top_camera_rotation_supine(self):
        """ For 'supine' starting position, rotate 'top' camera 180°, because else MIMo's head is at the bottom of the screen. """
        if self.starting_position == 'supine':
            cam_top_id = self.model.camera('top').id
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, [0.0, 0.0, 0.0], 'xyz')
            self.model.cam_quat[cam_top_id] = quat

    def set_embodiment(self, morph_age, physio_age):
        """ Sets the embodiment to the MuJoCo .xml file that fits to the
        morphological age 'morph_age' and physiological age 'pyhsio_age'. """
        xml_path = f"mimoEnv/assets/roll_over/prone/scene_act_{physio_age}_body_{morph_age}.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.fix_top_camera_rotation_supine()

        self.initialize()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.reset()

        # This was here for testing that MGC curriculum would actually grow MIMo. I checked
        # this by rendering after 'set_embodiment' and checking that the body has grown.
        # render_top_down_and_save(self, self.starting_position, '/home/leon/MIMo', f'test_embodiment_age{morph_age}')

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

    def disable_isr(self):
        self.isr = False

    def is_success(self, achieved_goal, desired_goal):
        """ Did we reach our goal rotation.

        This is a **pure** function of its arguments: it does not read the live simulation
        state. That matters because HER relabels goals offline and recomputes the reward from
        stored (achieved_goal, desired_goal) pairs. The previous implementation ignored both
        arguments and read ``self.get_achieved_goal_cos()`` instead, which made every relabelled
        transition silently keep the reward of the real transition.

        The threshold used to be hardcoded (0.95, or 0.5 under 'success_at_side_lying'). It now
        lives in the goal itself -- :meth:`.sample_goal` returns 0.5 instead of 0.95 in the
        side-lying case -- so the behaviour is unchanged while the function becomes relabelable.

        The 'intrinsic' goal function is exempt: its goals are dictionaries of sensor readings,
        not rotations, and success there was always measured by the cosine rotation regardless of
        the goal. That path is left as it was and is not HER-compatible.

        Arguments:
            achieved_goal (float | np.ndarray): The achieved hip/chest rotation. May be batched.
            desired_goal (float | np.ndarray): The target rotation. May be batched.

        Returns:
            bool | np.ndarray: Whether the achieved rotation reaches the desired rotation.
        """
        if self.goal_function == 'intrinsic':
            achieved_rotation = self.get_achieved_goal_cos()[0]
            return achieved_rotation >= (0.5 if self.success_at_side_lying else 0.95)

        ag = np.asarray(achieved_goal, dtype=np.float64)
        dg = np.asarray(desired_goal, dtype=np.float64)
        result = ag.reshape(-1) >= dg.reshape(-1)
        return bool(result[0]) if result.size == 1 else result

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
        #euler[0] = np.pi / 3.0
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

        # DISS. Fix initial randomization or sample random initial offset.
        if self.deterministic_initial_state_sampling is not None:
            random = self.deterministic_initial_state_sampling
        else:
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

        # Re-sample the goal for the new episode.
        #
        # This has to happen here rather than in 'MIMoEnv._reset_simulation', which is where it
        # looks like it happens: that method is part of the pre-1.0 gymnasium MujocoEnv API and
        # nothing calls it any more (gymnasium 1.0.0's 'MujocoEnv.reset' goes straight to
        # 'mj_resetData' and 'reset_model'). The consequence was that 'sample_goal' ran exactly
        # once, at construction, and 'self.goal' then never changed again -- invisible while the
        # goal was the constant 0.95, fatal for goal sampling and HER.
        #
        # It must also come after the alternating-position flip above, so that the goal matches
        # the position MIMo is actually starting from.
        #
        # Retire the episode that just ended into the curriculum statistics first, so the goal for
        # the new episode already accounts for it.
        if self._episode_max_achieved is not None:
            self._recent_episode_max.append(self._episode_max_achieved)
        self._episode_max_achieved = None

        self.goal = self.sample_goal()
        self._prev_achieved_goal = None

        # self.set_state(self.init_qpos, self.init_qvel)
        self.put_in_starting_position()
        self.pbrs_last_state_potential=0

        return self._get_obs()
    
    def get_goal_space(self, obs_space):
        if self.goal_function != 'intrinsic':
            return spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)

        # stableBaselines is incompatible with nested dict spaces, which is quite
        # unfortunate, but this simply means that instead of directly using the
        # observation space as a goal space, we instead calculate a box space by
        # flattening the dict obs space.
        # Since all our observations are scalars, we just add up the shapes
        # of each space.
                    
        # Intrinsic goal space. Here we need to check the chose intrinsic goal.
        if self.intrinsic_goal == 'all':
            # Take entire observation as goal.
            space_flattened = 0

            for space in obs_space.values():
                space_flattened += space.shape[0]
            return spaces.Box(-np.inf, np.inf, shape=(space_flattened,), dtype=np.float64)
        
        if self.intrinsic_goal == 'vesti':
            # Take only vestibular observation as goal.
            return spaces.Box(-np.inf, np.inf, shape=self.get_vestibular_obs().shape, dtype=np.float64)
        
        if self.intrinsic_goal == 'vesti_acc':
            # Take only accelerometer of vestibular observation as goal.
            return spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
        
        if self.intrinsic_goal == 'sparse_proprio':
            # Take entire observation as goal, but only sparse proprio observation (i.e. only joint qpos)
            # as proprio observation.
            space_flattened = self.proprioception.get_sparse_proprioception_obs().shape[0]

            keys_without_proprio = list(obs_space.keys())
            keys_without_proprio.remove('observation')

            for key in keys_without_proprio:
                space_flattened += obs_space[key].shape[0]
            return spaces.Box(-np.inf, np.inf, shape=(space_flattened,), dtype=np.float64)
        
        raise NotImplementedError
        
    def get_desired_goal_obs(self):
        if self.goal_function != 'intrinsic':
            return self.goal
        
        if self.intrinsic_goal == 'vesti' or self.intrinsic_goal == 'vesti_acc':
            return self.goal
        
        # For intrinsic goal function, we cannot simply return the goal, since it has
        # the same shape as the observation, i.e. a dictionary. Instead, we must
        # also flatten the goal like we did in 'get_goal_space'.
        return np.concatenate([self.goal[key] for key in sorted(self.goal.keys())])

    def sample_goal(self):
        """ Returns the goal rotation.

        By default this is the fixed rotation that :meth:`.is_success` used to hardcode: 0.95 for
        a full roll, 0.5 under 'success_at_side_lying'. Moving the threshold from the success
        check into the goal is what lets the success check be a pure function of its arguments.

        If 'goal_low'/'goal_high' are set, the target is instead drawn uniformly from that range
        on every reset. HER needs this: with a constant desired_goal the policy has no reason to
        condition on the goal input, and the relabelled transitions describe goals that are never
        asked for at evaluation time. Evaluation should always pin the goal to 0.95.

        For intrinsic goals, we sample a goal in the reset() function.

        Returns:
            np.array[float]: The target rotation, shape (1,).
        """
        if self.goal_function == "intrinsic":
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
                return self.get_achieved_goal()

        if self.goal_low is not None:
            high = self._effective_goal_high()
            # A degenerate range means a fixed goal. Return it without drawing: consuming a
            # random number would shift every later draw from the same generator, in particular
            # the initial joint randomisation in 'put_in_starting_position', which changes which
            # episodes you get. That made evaluation results depend on whether the goal was
            # pinned via goal_low/goal_high or left at the default (measured: 94% vs 98% on the
            # same policy and the same seeds).
            if self.goal_low == high:
                return np.array([float(self.goal_low)])
            return np.array([self.np_random.uniform(self.goal_low, high)])

        return np.array([0.5 if self.success_at_side_lying else 0.95])

    def _effective_goal_high(self):
        """ Upper end of the goal range for the episode that is about to start.

        Without the curriculum this is simply 'goal_high'. With it, the range only extends a
        little past what MIMo has recently managed:

            high = clip(quantile(recent episode maxima) + margin, goal_low + margin, goal_high)

        The reason is a measured failure mode of HER, not a general curriculum argument. **HER
        only ever relabels onto goals that were actually reached.** A run that plateaus at, say,
        rho ~ 0.6 therefore has no relabelled transition anywhere above 0.6; with
        'n_sampled_goal=4' four out of five sampled transitions are relabelled, so the region
        above the plateau is trained almost exclusively on the original transitions, and those
        all carry the sparse reward -1. The policy does not merely fail to learn high goals -- it
        learns something actively wrong there. Measured on the third E3b seed (2026-08-14),
        deterministic, 30 episodes each:

            goal 0.25 -> rho_max 0.546     goal 0.75 -> rho_max 0.091
            goal 0.50 -> rho_max 0.786     goal 0.95 -> rho_max 0.092

        A policy that ignored 'desired_goal' entirely would score 0.786 everywhere, so
        conditioning on an out-of-distribution goal is worse than not conditioning at all. Keeping
        the sampled goals inside the region HER can actually relabel into is what this avoids.

        Returns:
            float: The upper end of the range to sample the goal from.
        """
        if not self.goal_curriculum:
            return self.goal_high

        floor = min(self.goal_low + self.goal_curriculum_margin, self.goal_high)
        if not self._recent_episode_max:
            # No finished episode yet, e.g. the observation calls during construction.
            return floor

        reached = np.quantile(np.asarray(self._recent_episode_max, dtype=np.float64),
                              self.goal_curriculum_quantile)
        return float(np.clip(reached + self.goal_curriculum_margin, floor, self.goal_high))
    
    def get_rotation_degrees_to_goal_z_axis(self, body_name):
        """ Returns the rotation in degrees of the body part 'body_name' local x axis
        to the global z axis using rotation around y axis (i.e. rolling). Respects
        the current 'starting_position', i.e. for prone, it returns the rotation to
        the global positive z axis, while for supine, it returns the rotation to the
        global negative z axis.
        """
        # Some comments copied from old function:
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
        xmat = self.data.body(body_name).xmat.reshape(3, 3)
        dot_product = xmat[2,0]  # dot product from local x axis to global z axis.
        if self.starting_position == 'supine':
            dot_product *= -1

        # Calculating rotation in radiants by only rotating around y axis. (That is the
        # normalization term as second argument to 'arctan2')
        angle_rad = np.arctan2(np.sqrt(xmat[2, 2] ** 2 + xmat[2, 1] ** 2), dot_product)
        angle_deg = angle_rad * 180.0 / np.pi

        return angle_deg
    
    def get_achieved_rotation_degrees(self, body_name):
        """ Returns the achieved rotation of body 'body_name's local x axis to global z axis.
        This is simply 180° minus whatever 'get_rotation_degrees_to_global_z_axis'."""
        abs_degrees = abs(self.get_rotation_degrees_to_goal_z_axis(body_name))
        abs_degrees = 180 - abs_degrees
        return abs_degrees

    def _get_standardized_rotation(self, body_name):
        """ Get the standardized rotation of a body specified by name.

        Arguments:
            body_name (str): Name of the body to get the rotation for.
                We usually only use 'hip' or 'chest' here.

        Returns:
            float: The standardized rotation of the body.
        """
        # 180°: We are at starting position.
        # 90° or -90°: We are halfway there
        # 0°: We are done.
        angle_in_degrees = self.get_rotation_degrees_to_goal_z_axis(body_name)

        # We want a value in [0,1]. So we scale linear from 180°=0 to 0°=1 with abs(..)
        return abs(angle_in_degrees) / 180.0
    
    def get_dot_local_x_to_global_z(self, body_name):
        """ Returns the dot product of the local x axis to the global z axis for the specified body.
        This is just the R[2,0] entry in the rotation matrix.

        Parameters:
            body_name (str): Name of the body to get the rotation for.

        Returns:
            float: The dot product between the body's local x axis and the global z axis.
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
        rot_hip = self.get_dot_local_x_to_global_z("hip")
        rot_chest = self.get_dot_local_x_to_global_z("chest")

        # Goal rotation is exactly the opposite for supine starting position.
        if self.starting_position=="supine":
            rot_hip *= -1
            rot_chest *= -1

        rot_hip = (rot_hip + 1) / 2.0
        rot_chest = (rot_chest + 1) / 2.0
        return np.array([(rot_hip + rot_chest) / 2.0])
    
    def get_achieved_goal_intrinsic(self):
        if self.intrinsic_goal == 'vesti':
            return self.get_vestibular_obs()
        
        if self.intrinsic_goal == 'vesti_acc':
            return self.get_vestibular_obs()[:3]
        
        if self.intrinsic_goal == 'sparse_proprio':
            return self._get_obs(without_goals=True, sparse_proprio=True)
        
        if self.intrinsic_goal == 'all':
            return self._get_obs(without_goals=True, sparse_proprio=False)
        
        raise NotImplementedError

    def get_achieved_goal(self):
        """ Returns the goal calculated from either of the tree goal functions.
        """
        if self.goal_function=='angle':
            return self.get_achieved_goal_angle()
        elif self.goal_function=='cos':
            return self.get_achieved_goal_cos()
        elif self.goal_function=='intrinsic':
            return self.get_achieved_goal_intrinsic()
        
        raise NotImplementedError

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

        if self.goal_function != 'intrinsic' or self.intrinsic_goal == 'vesti' or self.intrinsic_goal == 'vesti_acc':
            desired_goal = self.goal

            if self.is_success(achieved_goal, desired_goal):
                return self.reward_success

            return -np.linalg.norm(desired_goal - achieved_goal)
        
        concat_scaled_achieved_goal = np.concatenate([achieved_goal[key] * self.intrinsic_goal_w[key] for key in sorted(achieved_goal.keys())])
        concat_desired_goal = np.concatenate([self.goal[key] * self.intrinsic_goal_w[key] for key in sorted(self.goal.keys())])

        return -np.linalg.norm(concat_desired_goal - concat_scaled_achieved_goal)
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info['chest_deg'] = self.get_achieved_rotation_degrees('chest')
        info['hip_deg'] = self.get_achieved_rotation_degrees('hip')
        achieved_goal = self.get_achieved_goal_cos()
        info['side_lying'] = 1.0 if achieved_goal >= 0.5 else 0.0
        info['45_deg'] = 1.0 if achieved_goal >= 0.25 else 0.0
        info['ctrl_cost'] = 0
        return obs, info
    
    def step(self, action):
        """ Run one timestep of the environment's dynamics.

        This overloaded function from mimo_env is used for PBRS (Potential Based Reward Shaping) to cache the
        potential of the current state for the reward function to use it in PBRS.
        """
        # Cache potential of current state to be used in calculating next reward.
        self.pbrs_last_state_potential = self.get_potential()
        # Same quantity, but stored as a goal rather than a potential, so that the shaping term
        # can be rebuilt from (achieved_goal, desired_goal, info) after HER rewrites the goal.
        if self.goal_function != 'intrinsic':
            self._prev_achieved_goal = np.asarray(self.get_achieved_goal(), dtype=np.float64).copy()

        obs, reward, terminated, truncated, info = super().step(action)

        # Write achieved hip and chest rotation in info dict.
        info['chest_deg'] = self.get_achieved_rotation_degrees('chest')
        info['hip_deg'] = self.get_achieved_rotation_degrees('hip')
        achieved_goal = self.get_achieved_goal_cos()
        info['side_lying'] = 1.0 if achieved_goal >= 0.5 else 0.0
        info['45_deg'] = 1.0 if achieved_goal >= 0.25 else 0.0
        info['raw_ctrl_cost'] = self.compute_raw_penalization_of_action(action)

        # Whether MIMo actually rolled over, independent of whatever goal was sampled or
        # relabelled for this episode. Under sampled goals 'is_success' can report a healthy
        # rate against easy targets while no real roll ever happens, so this is the quantity to
        # report -- cf. the rho_max ISR artefact that invalidated the earlier COMPOSER logs.
        info['rolled_over'] = 1.0 if achieved_goal >= 0.95 else 0.0

        # Running maximum for the goal curriculum. Retired into '_recent_episode_max' on reset.
        rho = float(np.asarray(achieved_goal).reshape(-1)[0])
        self._episode_max_achieved = (rho if self._episode_max_achieved is None
                                      else max(self._episode_max_achieved, rho))
        # On the last step of an episode this is the episode's maximum, which is the quantity the
        # evaluation protocol reports. Note that 'side_lying' and 'rolled_over' above describe the
        # *final* step instead, so the two disagree whenever MIMo rolls and then rolls back.
        info['episode_rho_max'] = self._episode_max_achieved
        info['goal_high_effective'] = self._effective_goal_high() if self.goal_low is not None \
            else float('nan')

        # Goal-INDEPENDENT reward terms, stored so HER can reuse them verbatim for the virtual
        # transitions it builds. Requires HerReplayBuffer(copy_info_dict=True).
        info['ctrl_cost'] = self.compute_penalization()
        if self._prev_achieved_goal is not None:
            info['prev_achieved_goal'] = self._prev_achieved_goal

        return obs, reward, terminated, truncated, info

    def compute_raw_penalization_of_action(self, action):
        """ Computes penalization (sum of squared values) of the given action without
        the penalization factor. """
        return np.square(action).sum()
    
    def compute_penalization(self):
        return 0 if self.nopen else self.pen_factor * np.square(self.data.ctrl).sum()

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
        reward -= self.compute_penalization()

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

        This function is **pure and vectorized** for the scalar goal functions ('angle', 'cos'):
        it depends only on its arguments, and it accepts either a single goal of shape (1,) or a
        batch of shape (N, 1). Both properties are required by HER, which rewrites desired_goal
        offline and then calls this method on whole batches via ``env_method``.

        The three reward terms differ in how they behave under relabelling, and that is what
        drives the implementation:

        * the success term depends on (achieved_goal, desired_goal) and must be recomputed;
        * the action penalty depends only on the controls, so it is goal-INDEPENDENT and is
          carried through ``info['ctrl_cost']``;
        * the PBRS term needs two consecutive states, which the (achieved, desired) pair alone
          cannot express, so the earlier achieved goal is carried through
          ``info['prev_achieved_goal']``.

        When those info keys are absent -- i.e. during ordinary stepping, or under
        ``copy_info_dict=False`` -- the live simulation values are used instead, which reproduces
        the original behaviour exactly.

        The 'intrinsic' goal function keeps the old live-state implementation: its goals are
        dictionaries of sensor readings and it is not HER-compatible.

        Arguments:
            achieved_goal (float | np.ndarray): The achieved hip and chest rotation, possibly
                batched with shape (N, 1).
            desired_goal (float | np.ndarray): The desired hip and chest rotation.
            info (dict | Sequence[dict]): Carries the goal-independent reward terms. May be a
                single dict or one per batch element.

        Returns:
            float | np.ndarray: A float for a single goal, an array of shape (N,) for a batch.
        """
        if self.goal_function == 'intrinsic':
            return self._compute_reward_intrinsic(achieved_goal, desired_goal, info)

        ag = np.asarray(achieved_goal, dtype=np.float64)
        batched = ag.ndim >= 2
        ag = ag.reshape(-1)
        dg = np.asarray(desired_goal, dtype=np.float64).reshape(-1)
        n = ag.shape[0]

        # Penalize excessive use of force unless disabled by '--nopen' argument.
        quad_ctrl_cost = self._info_column(info, 'ctrl_cost', self.compute_penalization(), n)

        success = ag >= dg

        if self.sparse_reward:
            reward = np.where(success, 0.0, -1.0)
        else:
            curr_potential = self._potential(ag, dg)
            if self.pbrs:
                prev_ag = self._info_column(
                    info, 'prev_achieved_goal',
                    self._prev_achieved_goal if self._prev_achieved_goal is not None else ag, n)
                reward = self.pbrs_w * (curr_potential - self._potential(prev_ag, dg))
            else:
                reward = curr_potential
            # If the goal is reached, give a very high positive reward.
            reward = np.where(success, float(self.reward_success), reward)

        reward = reward - quad_ctrl_cost
        return reward if batched else float(reward[0])

    def _potential(self, achieved, desired):
        """ Potential of a state, as a pure function of the achieved and desired rotation.

        Deliberately **continuous**, unlike :meth:`.get_potential`, which jumps to
        ``+reward_success`` at the goal.

        That jump is unreachable in the shaping difference as long as episodes terminate on
        success: the previous state of a step can then never be a goal state, and a current state
        that is a goal state is handled by the early success branch in :meth:`.compute_reward`
        instead. So dropping it changes nothing for the ordinary PPO/SAC pipeline -- verified by
        the PBRS regression check in ``mimoEnv/goalenv_check.py``.

        Under HER it is not unreachable, and that is why this method exists. HER relabels the
        goal to a rotation MIMo actually reached mid-trajectory, and MIMo routinely drifts back
        out of that rotation a few steps later. With the jump in place such a transition pays
        ``pbrs_w * (-reward_success)``: measured -50002.0 for a goal of 0.40 and a drift from
        0.45 to 0.38, which drives SAC's critic loss to ~4e7 within a few thousand steps.
        Terminating episodes do not help, because they only ever terminate on the real goal.
        """
        return -np.abs(desired - achieved)

    @staticmethod
    def _info_column(info, key, default, n):
        """ Pull a per-transition scalar out of an info dict, a sequence of them, or nothing.

        HER hands over one info dict per batch element (as a numpy array of objects), the
        environment step hands over a single dict, and the SB3 env checker hands over a
        two-element array. Anything missing falls back to 'default', which is the live value.
        """
        fallback = np.full(n, np.asarray(default, dtype=np.float64).reshape(-1)[0]
                           if np.size(default) else 0.0, dtype=np.float64)
        if info is None:
            return fallback
        if isinstance(info, dict):
            if key not in info:
                return fallback
            return np.full(n, np.asarray(info[key], dtype=np.float64).reshape(-1)[0])

        entries = np.asarray(info, dtype=object).reshape(-1)
        if entries.shape[0] != n:
            return fallback
        values = np.empty(n, dtype=np.float64)
        for i, entry in enumerate(entries):
            if isinstance(entry, dict) and key in entry:
                values[i] = np.asarray(entry[key], dtype=np.float64).reshape(-1)[0]
            else:
                values[i] = fallback[i]
        return values

    def _compute_reward_intrinsic(self, achieved_goal, desired_goal, info):
        """ The original live-state reward, kept for the 'intrinsic' goal function. """
        quad_ctrl_cost = self.compute_penalization()

        if self.is_success(achieved_goal, desired_goal):
            return self.reward_success - quad_ctrl_cost

        curr_potential = self.get_potential()

        if not self.pbrs:
            return curr_potential - quad_ctrl_cost

        return self.pbrs_w * (curr_potential - self.pbrs_last_state_potential) - quad_ctrl_cost
    