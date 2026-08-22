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

# 19.08.2026 The "non-scalar, non-extrinsic" goal.
#
# Every other goal function here ('angle', 'cos') is computed from the first 7 entries of
# 'data.qpos', i.e. from the root free joint. That is the body's absolute orientation in the world,
# and MIMo cannot sense it: proprioception only reports joints named 'robot:*', which excludes the
# free joint, so the quantity being optimised is not present in the observation at all. The goal
# below is built exclusively from things MIMo does observe:
#
#   * the joint angles of the six joints listed in 'INTRINSIC_GOAL_JOINTS' -- all of them 1-DoF
#     hinges, so one scalar each. These are exactly the values the 'Joint' sliders in
#     'mujoco.viewer' write: 'data.qpos[model.jnt_qposadr[joint_id]]', in radians. The identical
#     numbers are already inside 'obs["observation"]' (see 'SimpleProprioception'), which is the
#     whole point -- nothing new is being sensed, the goal is just a projection of the observation.
#   * one or more components of the vestibular accelerometer, 'obs["vestibular"][:3]'. This is the
#     only part of the goal that distinguishes prone from supine: the six joint angles are all
#     satisfiable without rolling at all, and measured across resets they barely move between the
#     two postures (hip_bend1 -0.32 prone vs -0.25 supine, in the normalised units below).
#
#     19.08.2026 The axis is **x**, not z. The accelerometer reports in the *site's local frame*,
#     and the 'vestibular' site sits on the head with its local x axis pointing along world +z --
#     the same convention as this module's own 'get_dot_local_x_to_global_z'. Measured at reset:
#     acc = [-9.74, -0.02, +0.37] prone against [+9.58, -0.10, -0.48] supine. Component z is
#     within noise of zero in both postures, so the original plan of using it would have produced
#     a goal that cannot tell prone from supine at all.
#
# Deliberately NOT the actuator state ('actuation_model.observations()'): that is the motor
# command, not the sensed configuration, so a goal defined on it could be reached by emitting the
# right control signal while lying perfectly still.
# 20.08.2026 The 'gravity' goal function -- the successor to 'intrinsic', which does not work
# (see docs/roll_over.md 3.4 for the measurements that killed it).
#
# It estimates the direction of gravity in MIMo's HIP frame and takes its x component. That is the
# same quantity 'get_dot_local_x_to_global_z("hip")' returns (+1 supine, -1 prone), but
# reconstructed from things MIMo can sense instead of read off the root free joint:
#
#     t = 0, MIMo demonstrably at rest:  g_site <- normalize(accelerometer)
#     every step, gyroscope only:        g_site <- rotate(g_site, -omega * dt)
#     goal:                              (R_hip^T R_site @ g_site)[0]
#
# Why each of the three defects of 'intrinsic' disappears:
#
#   * Shaking. Linear acceleration does not enter the integration at all, so there is nothing to
#     forge. The old goal could be driven 77% of the way to the target while lying still.
#   * Head turning. Self-cancelling rather than merely counterweighted: turning the head makes the
#     gyroscope rotate 'g_site' by exactly the amount that leaves 'R_rel @ g_site' unchanged. The
#     joint-angle targets that used to serve this purpose were measured to be ~2x too weak.
#   * Anti-correlation. The joints are no longer goal dimensions; they appear only in the frame
#     transform, which is where they belong.
#
# The root free joint cancels out of 'R_hip^T R_site' (measured residual 0.017 deg), so this stays
# non-extrinsic. Validated offline before implementation over 14 recorded episodes: correlation
# with the truth 0.9991 on a policy that rolls, drift 0.004 over a full 250-step episode, and the
# estimate stays at +0.995 for the policy that games the old goal.


# 21.08.2026 Which body frames the gravity direction is expressed in. Both, not just the hip:
# a hip-only version was trained for 1M steps and plateaued at rho 0.385 with the hip at 101.9 deg
# and the chest at 42.5 deg -- MIMo twists the pelvis and leaves the torso lying. That is the same
# failure the 'cos' goal ran into historically and fixed the same way; Kept as two separate dimensions
# rather than averaged, because an average cannot tell
# (hip -1, chest +1) from (hip 0, chest 0), while the vector distance penalises the gap directly.
GRAVITY_GOAL_BODIES = ("hip", "chest")


def rotate_by_angular_velocity(v, omega, dt):
    """ Rotate 'v' by '-omega * dt' (Rodrigues), keeping it a unit vector.

    'v' is a WORLD-fixed direction expressed in a ROTATING frame, so the transport theorem gives
    dv/dt = -omega x v: if the frame turns by +omega, the coordinates of a fixed vector in it turn
    by -omega. MuJoCo's gyro reports omega of the site in the site's own frame, which is exactly
    the omega this needs -- no conversion.

    Rodrigues is the closed-form exact solution for constant omega over the interval, not an Euler
    step; the only error left is that omega is not really constant within one 10 ms sample.
    """
    magnitude = np.linalg.norm(omega)
    theta = magnitude * dt
    if theta < 1e-12:
        return v
    k = -omega / magnitude
    rotated = (v * np.cos(theta) + np.cross(k, v) * np.sin(theta)
               + k * np.dot(k, v) * (1.0 - np.cos(theta)))
    return rotated / np.linalg.norm(rotated)


INTRINSIC_GOAL_JOINTS = [
    "robot:head_swivel",
    "robot:head_tilt_side",
    "robot:head_tilt",
    "robot:hip_lean1",
    "robot:hip_rot1",
    "robot:hip_bend1",
]

# Gravity, in m/s^2, used to bring the accelerometer into the same [-1, 1] range as the
# range-normalised joint angles. Without this the raw units are radians (hip_rot1 spans +-0.31)
# against m/s^2 (acc_x spans +-9.81) and a plain euclidean distance is ~97% accelerometer.
INTRINSIC_ACC_SCALE = 9.81

# Which accelerometer components go into the goal vector, as a subset of 'xyz'. See the axis note
# above for why the default is 'x' alone.
INTRINSIC_ACC_AXES = 'x'
_ACC_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def parse_acc_axes(axes):
    """ 'xz' -> [0, 2]. The empty string means no accelerometer dimensions at all. """
    axes = (axes or '').strip().lower()
    unknown = sorted(set(axes) - set(_ACC_AXIS_INDEX))
    if unknown:
        raise ValueError(f"Unknown accelerometer axes {unknown}; expected a subset of 'xyz'.")
    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate accelerometer axis in '{axes}'.")
    return [_ACC_AXIS_INDEX[axis] for axis in axes]


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
                 # --- 'intrinsic' goal function (see INTRINSIC_GOAL_JOINTS above) --------------
                 # The joints whose angles make up the goal vector. Configurable so the ablations
                 # (joints only, accelerometer only) cost nothing.
                 intrinsic_goal_joints=None,
                 # Which vestibular accelerometer components are appended to the goal vector, as
                 # a subset of 'xyz'. Empty string for none. Default 'x' -- see the axis note at
                 # the top of this module, the other two carry no posture signal.
                 intrinsic_acc_axes=INTRINSIC_ACC_AXES,
                 # Weight of the accelerometer dimension relative to the joint dimensions. Folded
                 # into the goal vector itself rather than applied in the distance, so that
                 # ':meth:`._potential`' stays a plain norm and therefore stays relabelable.
                 intrinsic_acc_w=1.0,
                 # Success radius: 'is_success' is '||achieved - desired|| <= intrinsic_goal_eps'.
                 # A continuous vector goal is never matched exactly, so "reached" has to mean
                 # "close enough"; under '--sparse_reward' this value IS the task definition.
                 intrinsic_goal_eps=0.15,
                 # Number of resets averaged into the recorded reference posture. A single reset
                 # would pin the goal to whatever 'head_swivel' happened to settle at under the
                 # initial joint noise, and MIMo would then be asked to reproduce that draw.
                 intrinsic_reference_samples=20,
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

        if goal_function not in ['angle', 'cos', 'intrinsic', 'gravity']:
            msg = f"Unknown reward function '{goal_function}'. "
            msg += "Needs to be 'angle', 'cos', 'intrinsic' or 'gravity'."
            raise ValueError(msg)
        
        # 19.08.2026 The 'intrinsic_goal' sub-modes ('all', 'vesti', 'vesti_acc',
        # 'sparse_proprio') were removed. They returned dict-valued goals, which SB3 cannot use as
        # a goal space and HER cannot relabel, and only 3 of 539 stored runs ever used them (2 of
        # those with a value that was no longer even in 'choices'). 'intrinsic' now means the
        # single flat posture goal defined at the top of this module.
        if intrinsic_goal_eps <= 0.0:
            raise ValueError(f"'intrinsic_goal_eps' must be positive, got {intrinsic_goal_eps}.")
        if intrinsic_reference_samples < 1:
            raise ValueError("'intrinsic_reference_samples' must be at least 1, got "
                             f"{intrinsic_reference_samples}.")
        
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
        self.intrinsic_goal_joints = list(INTRINSIC_GOAL_JOINTS if intrinsic_goal_joints is None
                                          else intrinsic_goal_joints)
        self.intrinsic_acc_axes = (intrinsic_acc_axes or '').strip().lower()
        self._intrinsic_acc_idx = parse_acc_axes(self.intrinsic_acc_axes)
        self.intrinsic_acc_w = intrinsic_acc_w
        self.intrinsic_goal_eps = intrinsic_goal_eps
        self.intrinsic_reference_samples = intrinsic_reference_samples
        # Filled in by ':meth:`.initialize`', which runs both at construction and on every
        # embodiment hot-swap, so the addresses follow 'set_embodiment' rebuilding the model.
        self._intrinsic_qpos_adr = None
        self._intrinsic_qpos_lo = None
        self._intrinsic_qpos_hi = None
        # Per-dimension spread of the recorded reference, kept for diagnostics: a dimension with a
        # large std across resets carries no posture information and is a candidate for dropping.
        self.intrinsic_goal_std = {}
        # 'gravity' goal function: the running estimate of the gravity direction in the vestibular
        # site's frame, and the sim time it was last advanced to. Reset to None on every episode so
        # the next read re-seeds it from the accelerometer while MIMo is still settled.
        self._g_site = None
        self._g_site_time = None
        self._vestibular_site_id = None
        self._gravity_body_ids = None
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

        if self.goal_function in ('intrinsic', 'gravity'):
            print(f"Creating prone and supine reference goals ('{self.goal_function}').")
            self.create_prone_and_supine_intrinsic_goal()
        self.intrinsic_goals_created = True
        if self.goal_function in ('intrinsic', 'gravity'):
            # 19.08.2026 Re-sample now that the references exist. 'initialize()' already set
            # 'self.goal', but it ran before this point, so 'sample_goal' took its
            # not-yet-created fallback branch and stored the constructor-time *achieved* goal.
            # Training never sees it (reset_model re-samples before the first step), but any
            # analysis script that reads 'env.goal' without resetting first gets a goal that is
            # simply MIMo's current posture -- which silently measures distance from where he
            # started instead of distance to the target.
            self.goal = self.sample_goal()
        self.fix_top_camera_rotation_supine()

    def fix_top_camera_rotation_supine(self):
        """ For 'supine' starting position, rotate 'top' camera 180°, because else MIMo's head is at the bottom of the screen. """
        if self.starting_position == 'supine':
            cam_top_id = self.model.camera('top').id
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, [0.0, 0.0, 0.0], 'xyz')
            self.model.cam_quat[cam_top_id] = quat

    def initialize(self):
        """ Called at construction and again on every embodiment hot-swap.

        Overridden to (re)resolve the addresses of the joints that make up the intrinsic goal.
        'set_embodiment' rebuilds 'self.model'/'self.data' from a different scene XML, which
        invalidates any cached index, and the base implementation ends by calling 'sample_goal',
        which for the intrinsic goal function reads the achieved goal -- so the addresses have to
        exist before 'super().initialize()' runs, not after.

        The recorded reference posture is deliberately *not* re-recorded here. It is stored in the
        range-normalised space, and joint ranges are the same across the four age scenes, so it
        stays comparable across an MGC embodiment swap.
        """
        if self.goal_function == 'gravity':
            self._vestibular_site_id = self.model.site('vestibular').id
            self._gravity_body_ids = [self.model.body(name).id
                                      for name in GRAVITY_GOAL_BODIES]
            self._g_site = None
            self._g_site_time = None

        if self.goal_function == 'intrinsic':
            ids = [self.model.joint(name).id for name in self.intrinsic_goal_joints]
            for name, jid in zip(self.intrinsic_goal_joints, ids):
                if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError(
                        f"Intrinsic goal joint '{name}' is not a hinge. The goal assumes one "
                        f"scalar per joint (this is what a 'Joint' slider in mujoco.viewer sets); "
                        f"a ball or free joint has no such scalar.")
            self._intrinsic_qpos_adr = np.asarray([self.model.jnt_qposadr[jid] for jid in ids])
            limits = self.model.jnt_range[ids]
            self._intrinsic_qpos_lo = limits[:, 0].astype(np.float64)
            self._intrinsic_qpos_hi = limits[:, 1].astype(np.float64)
            span = self._intrinsic_qpos_hi - self._intrinsic_qpos_lo
            if np.any(span <= 0):
                bad = [n for n, sp in zip(self.intrinsic_goal_joints, span) if sp <= 0]
                raise ValueError(f"Intrinsic goal joints without a usable range: {bad}. The goal "
                                 f"normalises each angle by its range, which needs a positive one.")
        super().initialize()

    @property
    def intrinsic_goal_dim(self):
        """ Length of the intrinsic goal vector: one entry per joint, plus accelerometer z. """
        return len(self.intrinsic_goal_joints) + len(self._intrinsic_acc_idx)

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
        """ Record the reference posture vector in prone and in supine.

        These are the two 'desired_goal's: starting prone, the target is the supine reference,
        and vice versa. Both are recorded once, at construction.

        Two things differ from the original single-shot implementation, and both were bugs waiting
        to happen:

        * The reference is **averaged over 'intrinsic_reference_samples' resets with ISR off**. A
          single reset fixes the goal at one draw of the initial joint noise -- 'head_swivel' in
          particular settles anywhere within a wide range -- and MIMo would then be scored on
          reproducing that specific draw rather than on the posture.
        * The RNG state is **saved and restored** around the recording. The reference resets draw
          from 'self.np_random', the same generator that produces the initial joint noise of every
          training episode, so without this the number of samples taken here would silently shift
          which episodes the run sees. This is the same failure that made evaluation depend on
          whether the goal was pinned via goal_low/goal_high (measured 94% vs 98%); see
          ':meth:`.sample_goal`'.
        """
        rng_state = self.np_random.bit_generator.state
        saved_isr = self.isr
        saved_position = self.starting_position
        self.isr = False

        try:
            for position in ('prone', 'supine'):
                self.starting_position = position
                samples = []
                for _ in range(self.intrinsic_reference_samples):
                    self.reset_model()
                    samples.append(np.asarray(self.get_achieved_goal(), dtype=np.float64))
                samples = np.asarray(samples, dtype=np.float64)
                reference = samples.mean(axis=0)
                spread = samples.std(axis=0)

                if TEST_INTRINSIC_GOALS_CREATION:
                    frame = self.render()
                    img = Image.fromarray(frame)
                    img.save(os.path.join('.', f'{position}_intrinsic_goal.png'))
                    np.savez(f"intrinsic_goal_{position}.npz",
                             reference=reference, std=spread, samples=samples)

                if position == 'prone':
                    self.prone_intrinsic_goal = reference
                else:
                    self.supine_intrinsic_goal = reference
                self.intrinsic_goal_std[position] = spread

                labels = self.intrinsic_goal_labels()
                print(f"Intrinsic reference posture ({position}, "
                      f"n={self.intrinsic_reference_samples}):")
                for label, value, sd in zip(labels, reference, spread):
                    print(f"    {label:<22} {value:+.3f}  (sd {sd:.3f})")
        finally:
            self.isr = saved_isr
            self.starting_position = saved_position
            self.np_random.bit_generator.state = rng_state

    def intrinsic_goal_labels(self):
        """ Names of the intrinsic goal dimensions, in order. For logging and diagnostics. """
        if self.goal_function == 'gravity':
            return [f'gravity_x_in_{name}_frame' for name in GRAVITY_GOAL_BODIES]
        labels = [name.replace('robot:', '') for name in self.intrinsic_goal_joints]
        labels += [f'vestibular_acc_{axis}' for axis in self.intrinsic_acc_axes]
        return labels

    def disable_isr(self):
        self.isr = False

    def _as_goal_batch(self, goal):
        """ Reshape a goal argument to (N, goal_dim), whatever shape it arrived in.

        HER hands over batches of shape (N, d), the environment step hands over a single (d,), and
        the SB3 checker hands over both. Everything downstream works on (N, d), so the scalar goal
        functions (d = 1) and the intrinsic posture goal (d = 7) share one code path.
        """
        return np.asarray(goal, dtype=np.float64).reshape(-1, self.goal_dim)

    @property
    def goal_dim(self):
        """ Length of a single goal. 1 for the scalar rotation goals. """
        if self.goal_function == 'intrinsic':
            return self.intrinsic_goal_dim
        if self.goal_function == 'gravity':
            return len(GRAVITY_GOAL_BODIES)
        return 1

    def _success_mask(self, achieved, desired):
        """ Success as a (N,) boolean array, from goals already shaped (N, goal_dim).

        For the scalar goal functions success is 'achieved >= desired': the goal is a rotation
        threshold, and overshooting it is still success.

        For the intrinsic posture goal it is '||achieved - desired|| <= intrinsic_goal_eps'. A
        threshold comparison makes no sense on a vector, and equality never happens on continuous
        sensor readings, so the goal is a point and success is a ball around it.
        """
        if self.goal_function in ('intrinsic', 'gravity'):
            # Both are point goals, not thresholds: 'gravity' runs from +1 (supine) to -1 (prone),
            # so 'achieved >= desired' would be satisfied at the start of a prone->supine episode.
            return np.linalg.norm(desired - achieved, axis=-1) <= self.intrinsic_goal_eps
        return (achieved >= desired).reshape(-1)

    def is_success(self, achieved_goal, desired_goal):
        """ Did we reach our goal.

        This is a **pure** function of its arguments: it does not read the live simulation
        state. That matters because HER relabels goals offline and recomputes the reward from
        stored (achieved_goal, desired_goal) pairs. The original implementation ignored both
        arguments and read ``self.get_achieved_goal_cos()`` instead, which made every relabelled
        transition silently keep the reward of the real transition.

        The threshold used to be hardcoded (0.95, or 0.5 under 'success_at_side_lying'). It now
        lives in the goal itself -- :meth:`.sample_goal` returns 0.5 instead of 0.95 in the
        side-lying case -- so the behaviour is unchanged while the function becomes relabelable.

        19.08.2026 The 'intrinsic' goal function used to be exempt, reading the *extrinsic* cosine
        rotation off live state no matter what goal it was handed -- which meant its goal and its
        success criterion measured different things. It now goes through the same pure path, with
        the vector success criterion described in ':meth:`._success_mask`'.

        Arguments:
            achieved_goal (float | np.ndarray): The achieved goal. May be batched.
            desired_goal (float | np.ndarray): The target goal. May be batched.

        Returns:
            bool | np.ndarray: Whether the goal is reached. A plain bool for a single goal, an
            array of shape (N,) for a batch.
        """
        result = self._success_mask(self._as_goal_batch(achieved_goal),
                                    self._as_goal_batch(desired_goal))
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

        # Drop the gravity estimate so the first read below re-seeds it from the accelerometer.
        # This has to come AFTER 'put_in_starting_position': that method drops MIMo onto the floor
        # and runs 'steps_after_reset' settle steps, and only once those are done is he at rest,
        # which is the single condition under which specific force equals gravity. Seeding before
        # it would both read a moving accelerometer and integrate the settle steps as if they were
        # part of the episode.
        self._g_site = None
        self._g_site_time = None

        return self._get_obs()
    
    def get_goal_space(self, obs_space):
        """ The goal space.

        A flat Box in both cases: shape (1,) for the scalar rotation goals, shape
        ('intrinsic_goal_dim',) for the intrinsic posture goal. Flatness is what makes the
        intrinsic goal usable at all -- SB3 cannot take a nested dict as a goal space, and
        HerReplayBuffer relabels by writing into an array.
        """
        return spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float64)

    def get_desired_goal_obs(self):
        return self.goal

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
        if self.goal_function in ("intrinsic", "gravity"):
            # The intrinsic goal is the reference posture recorded in the *opposite* position, so
            # it is fixed per starting position rather than sampled. HER still sees plenty of goal
            # variation, because every relabelled goal is an achieved posture vector from the
            # trajectory -- which is why this goal function needs neither --goal_low/--goal_high
            # nor the goal curriculum that the scalar goals do.
            #
            # The references are recorded at the very end of the constructor. Observation calls
            # before that point reach this function while they do not exist yet, so fall back to
            # the current achieved goal -- it is only used to shape the observation space.
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
        """ The intrinsic posture goal: what MIMo can actually sense about his own configuration.

        Returns a flat vector of 'intrinsic_goal_dim' entries, all of them O(1) in magnitude:

        * one entry per joint in 'intrinsic_goal_joints', its angle mapped from its own
          [lower, upper] limit onto [-1, 1]. The raw value is 'data.qpos[jnt_qposadr[id]]' in
          radians -- the number a 'Joint' slider in 'mujoco.viewer' writes, and the same number
          that already sits in the 'qpos' block of 'obs["observation"]'.
        * one entry per accelerometer axis in 'intrinsic_acc_axes' (default just x), divided by
          gravity and weighted by 'intrinsic_acc_w'. The weight is folded in here rather than
          applied in the distance so
          that ':meth:`._potential`' stays a plain norm of its two arguments, and therefore stays
          a pure function that HER can recompute.

        Nothing here reads the root free joint, which is the whole point: unlike
        ':meth:`.get_achieved_goal_cos`' this quantity is available to the policy.
        """
        angles = self.data.qpos[self._intrinsic_qpos_adr]
        scaled = 2.0 * (angles - self._intrinsic_qpos_lo) \
            / (self._intrinsic_qpos_hi - self._intrinsic_qpos_lo) - 1.0

        if not self._intrinsic_acc_idx:
            return scaled.astype(np.float64)

        acc = self.get_vestibular_obs()[self._intrinsic_acc_idx] \
            / INTRINSIC_ACC_SCALE * self.intrinsic_acc_w
        return np.concatenate([scaled, acc]).astype(np.float64)

    def get_achieved_goal_gravity(self):
        """ The gravity direction in each of 'GRAVITY_GOAL_BODIES', x component.

        One entry per body, all on the same scale: +1 supine, -1 prone -- the same scale as
        'get_dot_local_x_to_global_z(body)', but
        reconstructed from the gyroscope and the proprioceptive joint chain instead of from the
        root free joint that proprioception does not report.

        The estimate is a running one, advanced lazily: the first read after a reset seeds it from
        the accelerometer (the one moment MIMo is demonstrably at rest, so specific force IS
        gravity), and every later read integrates the gyroscope forward to the current 'data.time'.
        Keying on the simulation clock rather than on a step counter makes this idempotent -- the
        environment reads the achieved goal several times per step (observation, reward, info) and
        all of them must see the same value.

        Note this makes 'get_achieved_goal' history-dependent, unlike the other goal functions.
        'compute_reward' stays pure -- it only ever sees the stored achieved/desired pair -- so HER
        relabelling is unaffected.
        """
        self._advance_gravity_estimate()
        R_site = self.data.site_xmat[self._vestibular_site_id].reshape(3, 3)
        # R_body^T @ R_site depends only on the joints between that body and the head; the root
        # rotation cancels, which is what keeps this quantity non-extrinsic.
        return np.array([float(((self.data.xmat[body_id].reshape(3, 3).T @ R_site)
                                @ self._g_site)[0])
                         for body_id in self._gravity_body_ids])

    def _advance_gravity_estimate(self):
        """ Seed or integrate ':attr:`._g_site`' up to the current simulation time. """
        now = float(self.data.time)
        if self._g_site is None or self._g_site_time is None:
            acc = self.get_vestibular_obs()[:3]
            norm = np.linalg.norm(acc)
            # A degenerate reading would be free fall, which cannot happen at reset; fall back to
            # the site's own +x rather than dividing by zero.
            self._g_site = acc / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
            self._g_site_time = now
            return
        dt = now - self._g_site_time
        if dt <= 0.0:
            return
        self._g_site = rotate_by_angular_velocity(
            self._g_site, self.get_vestibular_obs()[3:6], dt)
        self._g_site_time = now

    def get_achieved_goal(self):
        """ Returns the goal calculated from either of the tree goal functions.
        """
        if self.goal_function=='angle':
            return self.get_achieved_goal_angle()
        elif self.goal_function=='cos':
            return self.get_achieved_goal_cos()
        elif self.goal_function=='intrinsic':
            return self.get_achieved_goal_intrinsic()
        elif self.goal_function=='gravity':
            return self.get_achieved_goal_gravity()
        
        raise NotImplementedError

    def get_potential(self):
        """ Returns the potential of the *current live state*, against the current goal.

        The negative euclidean distance between the desired goal and the achieved goal.
        See [https://arxiv.org/pdf/2201.08299 Goal-Conditioned Reinforcement Learning:
        Problems and Solutions by Liu et al. 2022 pp2-3 section 'Sample Efficiency:
        Towards Sparse Rewards'] for a discussion and the motivation behind this
        lazy approach.

        Not used by the reward any more -- ':meth:`.compute_reward`' uses the pure, relabelable
        ':meth:`._potential`' instead, and this method's jump to '+reward_success' inside the goal
        region is exactly what made the reward diverge under HER. It is kept because
        'goalenv_check.py' and 'results/collect_observation_util.py' read it as a diagnostic.

        Works for both the scalar goals and the intrinsic posture vector.
        """
        achieved_goal = self._as_goal_batch(self.get_achieved_goal())
        desired_goal = self._as_goal_batch(self.goal)

        if self._success_mask(achieved_goal, desired_goal)[0]:
            return self.reward_success

        return float(self._potential(achieved_goal, desired_goal)[0])

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
        # The achieved goal before the step, so that the potential-based shaping term can be
        # rebuilt from (achieved_goal, desired_goal, info) after HER rewrites the goal. Cached for
        # every goal function including 'intrinsic', where it is a whole posture vector -- the
        # guard that used to skip it there would have made PBRS silently unrelabelable.
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
        as a reward (which is the negative euclidean distance to the desired goal).

        This function is **pure and vectorized**: it depends only on its arguments, and it accepts
        either a single goal of shape (goal_dim,) or a batch of shape (N, goal_dim). Both
        properties are required by HER, which rewrites desired_goal offline and then calls this
        method on whole batches via ``env_method``.

        19.08.2026 This now covers the 'intrinsic' goal function too. It used to have its own
        live-state implementation ('_compute_reward_intrinsic'), which is exactly the shape of bug
        this docstring warns about: it read 'self.get_potential()' and ignored its arguments, so
        relabelling it was a no-op. The only difference between the goal functions is now the
        dimensionality of the goal vector and the success test; the reward structure is shared.

        The three reward terms differ in how they behave under relabelling, and that is what
        drives the implementation:

        * the success term depends on (achieved_goal, desired_goal) and must be recomputed;
        * the action penalty depends only on the controls, so it is goal-INDEPENDENT and is
          carried through ``info['ctrl_cost']``;
        * the PBRS term needs two consecutive states, which the (achieved, desired) pair alone
          cannot express, so the earlier achieved goal is carried through
          ``info['prev_achieved_goal']`` (a full vector under the intrinsic goal function).

        When those info keys are absent -- i.e. during ordinary stepping, or under
        ``copy_info_dict=False`` -- the live simulation values are used instead, which reproduces
        the original behaviour exactly.

        Arguments:
            achieved_goal (float | np.ndarray): The achieved goal, possibly batched with shape
                (N, goal_dim).
            desired_goal (float | np.ndarray): The desired goal.
            info (dict | Sequence[dict]): Carries the goal-independent reward terms. May be a
                single dict or one per batch element.

        Returns:
            float | np.ndarray: A float for a single goal, an array of shape (N,) for a batch.
        """
        raw = np.asarray(achieved_goal, dtype=np.float64)
        batched = raw.ndim >= 2
        ag = self._as_goal_batch(raw)
        dg = self._as_goal_batch(desired_goal)
        n = ag.shape[0]

        # Penalize excessive use of force unless disabled by '--nopen' argument.
        quad_ctrl_cost = self._info_column(info, 'ctrl_cost', self.compute_penalization(), n)

        success = self._success_mask(ag, dg)

        if self.sparse_reward:
            reward = np.where(success, 0.0, -1.0)
        else:
            curr_potential = self._potential(ag, dg)
            if self.pbrs:
                fallback = self._prev_achieved_goal if self._prev_achieved_goal is not None else ag
                prev_ag = self._info_block(info, 'prev_achieved_goal', fallback, n, self.goal_dim)
                reward = self.pbrs_w * (curr_potential - self._potential(prev_ag, dg))
            else:
                reward = curr_potential
            # If the goal is reached, give a very high positive reward.
            reward = np.where(success, float(self.reward_success), reward)

        reward = reward - quad_ctrl_cost
        return reward if batched else float(reward[0])

    def _potential(self, achieved, desired):
        """ Potential of a state, as a pure function of the achieved and desired goal.

        The negative euclidean distance between the two, taken over the goal dimensions. For the
        scalar goal functions that is just '-|desired - achieved|'; for the intrinsic posture goal
        it is the norm over all 'goal_dim' entries. Both arguments must already be shaped
        (N, goal_dim) -- use ':meth:`._as_goal_batch`'.

        Deliberately **continuous**, unlike :meth:`.get_potential`, which jumps to
        ``+reward_success`` at the goal.

        That jump is unreachable in the shaping difference as long as episodes terminate on
        success: the previous state of a step can then never be a goal state, and a current state
        that is a goal state is handled by the early success branch in :meth:`.compute_reward`
        instead. So dropping it changes nothing for the ordinary PPO/SAC pipeline -- verified by
        the PBRS regression check in ``mimoEnv/goalenv_check.py``.

        Under HER it is not unreachable, and that is why this method exists. HER relabels the
        goal to a state MIMo actually reached mid-trajectory, and MIMo routinely drifts back
        out of it a few steps later. With the jump in place such a transition pays
        ``pbrs_w * (-reward_success)``: measured -50002.0 for a goal of 0.40 and a drift from
        0.45 to 0.38, which drives SAC's critic loss to ~4e7 within a few thousand steps.
        Terminating episodes do not help, because they only ever terminate on the real goal.
        """
        return -np.linalg.norm(desired - achieved, axis=-1) * self._potential_scale

    @property
    def _potential_scale(self):
        """ Puts the shaping potential of every goal function on the same [0, 1] scale.

        21.08.2026 'cos' measures progress in [0, 1], so a reset sits at potential -1 and the goal
        at 0. The 'gravity' goal runs from +1 to -1 per body, so with two bodies a reset sits at
        -2.83 -- the PBRS term was 2.83x larger than in the baseline while 'reward_success' stayed
        at 500, i.e. the terminal bonus was relatively 2.83x weaker. Measured consequence: the
        policy happily farmed shaping reward up to rho 0.48 and had little incentive to pay for
        the expensive last part of the roll.

        Dividing by the reset distance (2*sqrt(n_bodies)) makes '--pbrs_w' and 'reward_success'
        mean the same thing for 'gravity' as for 'cos', which is what makes the two comparable.

        Note this deliberately does NOT scale ':attr:`.intrinsic_goal_eps`': the success radius
        stays in the readable +-1 units of the goal itself, where eps = 2*(1 - rho_target) per
        body -- eps 0.21 for two bodies is about rho 0.925.
        """
        if self.goal_function == 'gravity':
            return 1.0 / (2.0 * np.sqrt(len(GRAVITY_GOAL_BODIES)))
        return 1.0

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

    @staticmethod
    def _info_block(info, key, default, n, width):
        """ The vector-valued counterpart of ':meth:`._info_column`', returning (n, width).

        Needed because under the intrinsic goal function 'prev_achieved_goal' is a whole posture
        vector rather than a scalar, and PBRS has to reconstruct it per transition for HER's
        relabelled batches.
        """
        fallback = np.asarray(default, dtype=np.float64).reshape(-1, width)
        if fallback.shape[0] == 1:
            fallback = np.repeat(fallback, n, axis=0)
        if fallback.shape[0] != n:
            fallback = np.repeat(fallback[:1], n, axis=0)

        if info is None:
            return fallback
        if isinstance(info, dict):
            if key not in info:
                return fallback
            return np.repeat(np.asarray(info[key], dtype=np.float64).reshape(1, width), n, axis=0)

        entries = np.asarray(info, dtype=object).reshape(-1)
        if entries.shape[0] != n:
            return fallback
        values = fallback.copy()
        for i, entry in enumerate(entries):
            if isinstance(entry, dict) and key in entry:
                values[i] = np.asarray(entry[key], dtype=np.float64).reshape(width)
        return values

