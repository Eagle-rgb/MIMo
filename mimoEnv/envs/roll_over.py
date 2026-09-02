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

from mimoEnv.envs.roll_over_look import LookReward
from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, \
    DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS
from mimoActuation.actuation import SpringDamperModel
import mujoco
import numpy as np
import os
from mimoEnv.utils import get_minimal_z_coordinate
from gymnasium import spaces

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

# 01.09.2026 Vision. The motivation is the blindness literature: only 20 % of blind infants roll
# at 9 months against 100 % of same-aged sighted infants, so vision belongs in this experiment.
#
# The upstream 'DEFAULT_VISION_PARAMS' renders 256x256 RGB *per eye*, i.e. 393216 values per
# observation. That is what ran the cluster GPUs out of VRAM, for three compounding reasons:
#
#   - SB3's 'CombinedExtractor' builds one NatureCNN per image key. Its three strided convs
#     (8/4, 4/2, 3/1) shrink the image by ~8x, so the flatten before the output layer is
#     64 * (H/8)^2: 50176 at 256x256, hence a 50176x256 Linear = 12.8 M parameters *per eye*.
#     SAC instantiates that for the actor, both critics and both critic targets, and Adam keeps
#     two moments per parameter.
#   - Conv activations scale with H*W and are held for the backward pass, once per batch element.
#   - The replay buffer holds 'obs' and 'next_obs': 393 kB per observation is 79 GB for a 100k
#     HER buffer. That is host RAM rather than VRAM, but it kills a run just as reliably.
#
# 64x64 is the standard resolution for pixel RL (DQN/Atari onwards, and what NatureCNN was tuned
# for). Against 256 it cuts the observation 16x (393216 -> 24576 values) and the CNN's final
# Linear 49x (1024x256 = 262 k per eye). '--vision_grayscale' takes another 3x off, and
# '--vision_eyes=left|right' a further 2x; rolling needs no binocular disparity.
#
# Do not go below 36: NatureCNN's convs collapse the feature map to zero width, and SB3 itself
# warns about image spaces smaller than 36x36.
VISION_RESOLUTION_DEFAULT = 64
VISION_RESOLUTION_MIN = 36

# What runs saved before 01.09.2026 were trained at: those used 'DEFAULT_VISION_PARAMS'
# unchanged. 'load_model_yaml' substitutes this when a stored 'data.yml' has 'vision: true' but
# no 'vision_resolution', so reloading such a run does not silently rebuild it at 64x64.
VISION_RESOLUTION_LEGACY = 256

EYES = {"both": ["eye_left", "eye_right"], "left": ["eye_left"], "right": ["eye_right"]}
""" The camera sets selectable with ``--vision_eyes``. The camera names must exist in the scene
XML.

:meta hide-value:
"""


def scene_path(age_physio, age_morph, playroom=False):
    """ The pre-generated scene for one (physiological, morphological) age pair.

    Args:
        age_physio (int): Actuation age, one of :data:`.AGES`.
        age_morph (int): Body age, one of :data:`.AGES`.
        playroom (bool): Load the '_playroom' variant, which adds the toys. Default ``False``.

    Returns:
        str: Path to the scene xml.
    """
    if age_physio not in AGES or age_morph not in AGES:
        raise ValueError(f"Allowed ages: {AGES}, got physio={age_physio}, morph={age_morph}")
    name = f"scene_act_{age_physio}_body_{age_morph}{'_playroom' if playroom else ''}.xml"
    return os.path.join(SCENE_DIRECTORY, "roll_over", "prone", name)


def make_vision_params(resolution=VISION_RESOLUTION_DEFAULT, grayscale=False, eyes="both",
                       shadows=False, fovy=60):
    """ Builds the vision configuration for the roll-over experiment.

    This replaces :data:`~mimoEnv.envs.mimo_env.DEFAULT_VISION_PARAMS`, whose 256x256 per eye does
    not fit on the cluster GPUs. See the comment above :data:`.VISION_RESOLUTION_DEFAULT` for the
    arithmetic.

    Args:
        resolution (int): Edge length of the square image rendered per eye. Default
            :data:`.VISION_RESOLUTION_DEFAULT` (64). Must be at least
            :data:`.VISION_RESOLUTION_MIN` (36).
        grayscale (bool): Render RGB but return a single luminance channel, i.e. `(H, W, 1)`
            instead of `(H, W, 3)`. Default ``False``.
        eyes (str): Which cameras to use, one of the keys of :data:`.EYES`. Default ``"both"``.
        shadows (bool): Render shadows and floor reflections. Default ``False``: they cost about
            eight times the render and do not depend on the resolution. See
            :meth:`~mimoVision.vision.SimpleVision.get_vision_obs`.
        fovy (float): Vertical field of view in degrees. Default 60, as upstream.

    Returns:
        Dict: A vision configuration for :class:`~mimoVision.vision.SimpleVision`.
    """
    if resolution < VISION_RESOLUTION_MIN:
        raise ValueError(
            f"--vision_resolution={resolution} is below {VISION_RESOLUTION_MIN}. SB3's NatureCNN "
            f"reduces the image by a factor of ~8 over three strided convolutions and the "
            f"feature map collapses below that size.")
    if eyes not in EYES:
        raise ValueError(f"--vision_eyes={eyes} is not one of {sorted(EYES)}.")
    # A fresh dict per eye: 'MIMoEnv.vision_setup' mutates the per-camera entries in place.
    return {eye: {"width": resolution, "height": resolution, "fovy": fovy,
                  "acuity": False, "foveation": False, "grayscale": grayscale,
                  "shadows": shadows}
            for eye in EYES[eyes]}


def vision_obs_size(vision_params):
    """ Number of values one vision observation contributes, summed over the cameras. """
    if not vision_params:
        return 0
    return sum(params["width"] * params["height"] * (1 if params.get("grayscale") else 3)
               for params in vision_params.values())

# 26.08.2026 The 'angle' and 'intrinsic' goal functions were REMOVED. Two goal functions are
# left, selected by '--goal_achievement_function':
#
#   'cos'      the scalar rotation ':meth:`.get_achieved_goal_cos`', rho in [0, 1].
#   'gravity'  the 2-vector ':meth:`.get_achieved_goal_gravity`', +1 supine to -1 prone per body.
#
# 'angle' was the first goal function ever written here and was already documented as broken.
# 'intrinsic' was the "non-scalar, non-extrinsic" goal: a 7-vector of six range-normalised hinge
# angles plus vestibular acc-x, built only from what MIMo can sense, since 'cos' reads the root
# free joint, which proprioception excludes. It did not work -- two 1M-step PPO runs reached
# rho_max 0.019 and 0.038 against the baseline's 0.951, because the accelerometer reports gravity
# *plus self-acceleration* and the policy forged the prone gravity signature by shaking while
# lying supine. The measurements are in docs/roll_over.md 3.4. Models trained with
# '--goal_achievement_function=angle|intrinsic' can no longer be loaded against this environment.
#
# 26.08.2026 'gravity' was removed earlier the same day and READDED here. The argument for
# removing it was that its *mean over the two bodies* equals what 'cos' measures, so it made the
# same claim about sensing at the cost of a second training configuration -- and that claim is
# now made without training at all, by 'results/intrinsic/intrinsic_rho_check.py'.
#
# What that argument missed is that HER does not see the mean, it sees the vector. The goal is
# two-dimensional and its success criterion is a ball rather than a threshold, and under a sparse
# reward those are exactly the two properties that decide what a relabelled transition is worth:
# 'gravity' trained without --goal_low/--goal_high where 'cos' needs them, 14/16 seeds against
# 3/16 (docs/roll_over.md 3.6). '--goal_tolerance' was added to give 'cos' the ball criterion
# alone; it cannot give it the second dimension, and (hip -1, chest +1) is indistinguishable from
# (hip 0, chest 0) once averaged, which is the gap a vector goal closes.


# 20.08.2026 The 'gravity' goal function -- the successor to 'intrinsic', which does not work
# (see docs/roll_over.md 3.4 for the measurements that killed it).
#
# It estimates the direction of gravity in MIMo's body frames and takes its x component. That is
# the same quantity 'get_dot_local_x_to_global_z(body)' returns (+1 supine, -1 prone), but
# reconstructed from things MIMo can sense instead of read off the root free joint:
#
#     t = 0, MIMo demonstrably at rest:  g_site <- normalize(accelerometer)
#     every step, gyroscope only:        g_site <- rotate(g_site, -omega * dt)
#     goal:                              (R_body^T R_site @ g_site)[0]
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
# The root free joint cancels out of 'R_body^T R_site' (measured residual 0.017 deg), so this
# stays non-extrinsic. Validated offline before implementation over 14 recorded episodes:
# correlation with the truth 0.9991 on a policy that rolls, drift 0.004 over a full 250-step
# episode, and the estimate stays at +0.995 for the policy that games the old goal.


# 21.08.2026 Which body frames the gravity direction is expressed in. Both, not just the hip:
# a hip-only version was trained for 1M steps and plateaued at rho 0.385 with the hip at 101.9 deg
# and the chest at 42.5 deg -- MIMo twists the pelvis and leaves the torso lying. That is the same
# failure the 'cos' goal ran into historically and fixed the same way; kept as two separate
# dimensions rather than averaged, because an average cannot tell (hip -1, chest +1) from
# (hip 0, chest 0), while the vector distance penalises the gap directly.
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
                 # Which quantity the goal is defined on: 'cos' (scalar rho) or 'gravity' (the
                 # 2-vector of gravity-x per body). See the module header.
                 goal_function='cos',
                 # --- 'gravity' goal function ---------------------------------------------
                 # Success radius: 'is_success' is '||achieved - desired|| <= gravity_goal_eps'.
                 # A threshold makes no sense on a vector and exact equality never happens on a
                 # continuous sensor reading, so the goal is a point and success is a ball.
                 # In the goal's own +-1 units, eps = 2*(1 - rho_target) per body, so 0.15 over
                 # two bodies is about rho 0.925.
                 # Previously: --intrinsic_goal_eps
                 gravity_goal_eps=0.15,
                 # How many resets the prone/supine reference goals are averaged over. A single
                 # reset would pin the goal to one draw of the initial joint noise.
                 # Previously: --intrinsic_reference_samples
                 gravity_reference_samples=20,
                 # 01.09.2026 Load the playroom variant of the scene: the same MIMo on the same
                 # floor, with ten toys ringed around him just out of reach. Without it there is
                 # nothing in either eye -- supine MIMo looks at a flat gradient skybox, prone he
                 # looks at the floor 1 cm from his face -- so any vision-driven reward is
                 # vacuous. Built by 'mimoEnv/assets/roll_over/generate_playroom_scenes.py'.
                 playroom=False,
                 # --- the looking reward --------------------------------------------------
                 # 01.09.2026 Reward MIMo for what he can see rather than for how far he has
                 # turned. See 'mimoEnv/envs/roll_over_look.py' for the mechanism and the
                 # measurements behind the defaults. Requires 'playroom=True'.
                 look_reward=False,
                 look_w=100.0,
                 look_habituation_steps=50,
                 # 0 = a toy already looked at stays spent for the rest of the episode, which is
                 # what makes the episode reward "how many different toys did you find".
                 look_recovery_steps=0,
                 look_novelty_w=200.0,
                 look_seen_threshold=0.01,
                 look_fovea=0.35,
                 look_eyes='left',
                 # Whether rho contributes to the reward at all. '--no_rotation_reward' drops the
                 # sparse/PBRS/success terms and leaves only the looking reward and the control
                 # cost, which is the configuration that makes "does looking produce rolling" a
                 # question the run can answer instead of assume. rho stays in 'info' either way.
                 rotation_reward=True,
                 # 25.08.2026 Turns the scalar success test from the threshold
                 # 'achieved >= desired' into the band '|achieved - desired| <= goal_tolerance',
                 # None keeps the threshold, so every stored 'data.yml' reloads unchanged.
                 #
                 # Why this exists: under a sparse reward HER relabels onto goals the policy
                 # actually reached, and early in training those sit inside the jitter band of
                 # rho (measured: rho spans 0..0.007 under a random policy). A threshold sitting
                 # inside that band is a coin flip -- 41 % of relabelled transitions score 0 --
                 # so 'close to the goal' does not look any better to the critic than 'far from
                 # it', and there is no gradient in goal space to climb. The band labels every
                 # nearby goal 0 and only the distant real goal -1, which is what forces the
                 # critic to represent the distance to the goal. That is the mechanism the
                 # 'gravity' goal function gets for free from its ball criterion: 100 % of its
                 # relabelled transitions scored 0 at the start of training, and it trained
                 # without --goal_low/--goal_high, 14/16 seeds against 3/16 for 'cos'.
                 #
                 # 0.05 is the matched radius: 'gravity' used 0.15 over two bodies, i.e. 0.106
                 # per body in its +-1 units, and rho is that quantity halved.
                 #
                 # At the real goal this changes nothing. 'sample_goal' returns 1.0 instead of
                 # 0.95 when a tolerance is set, and rho never exceeds 1.0, so
                 # '|rho - 1.0| <= 0.05' is exactly 'rho >= 0.95'. Only the *relabelled* goals
                 # see a different rule -- which is what makes this a clean A/B.
                 goal_tolerance=None,
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
                 # Terminate the episode on success. Must be False for HER -- see the note in
                 # 'compute_reward'.
                 done_active=True,
                 **kwargs):

        if starting_position not in ["prone", "supine", "alternating"]:
            msg = f"Unknown starting position '{starting_position}'. "
            msg += "Needs to be 'prone', 'supine' or 'alternating'."
            raise ValueError(msg)

        if goal_function not in ['cos', 'gravity']:
            msg = f"Unknown goal function '{goal_function}'. Needs to be 'cos' or 'gravity'. "
            msg += "('angle' and 'intrinsic' were removed on 26.08.2026 -- see the module header.)"
            raise ValueError(msg)

        if goal_tolerance is not None:
            if goal_tolerance <= 0.0:
                raise ValueError(f"'goal_tolerance' must be positive, got {goal_tolerance}.")
            if goal_function == 'gravity':
                # That is already a point goal with a radius; a second one would be ambiguous.
                raise ValueError(
                    "'goal_tolerance' applies to the scalar goal function ('cos') only, got "
                    f"'{goal_function}'. The gravity goal is a point goal already -- use "
                    "'gravity_goal_eps' to set its success radius.")

        if gravity_goal_eps <= 0.0:
            raise ValueError(f"'gravity_goal_eps' must be positive, got {gravity_goal_eps}.")
        if gravity_reference_samples < 1:
            raise ValueError("'gravity_reference_samples' must be at least 1, got "
                             f"{gravity_reference_samples}.")
        
        # Instead of supplying 'age' as a parameter to the environment directly, we beforehand created the
        # appropriate age scene. So we manually specify the scene location.
        # This is necessary because the parallel RBI runs have problems deleting and creating the temporary
        # scenes at the same time.
        #if age == 18:  # default
        #    model_path = os.path.join(SCENE_DIRECTORY, "roll_over_prone_scene.xml")
        self.playroom = playroom
        model_path = scene_path(age_physio, age_morph, playroom)

        self.look_reward = look_reward
        self.rotation_reward = rotation_reward
        if look_reward and not playroom:
            raise ValueError("'look_reward' needs the playroom -- without toys there is nothing "
                             "to look at and the reward is identically zero. Pass "
                             "'playroom=True' (--playroom).")
        if not rotation_reward and not look_reward:
            raise ValueError("'rotation_reward=False' with no 'look_reward' leaves nothing but "
                             "the control cost, whose optimum is to lie still.")
        if look_reward and not isr:
            # Measured 01.09.2026, 10 episodes of 200 steps under a random policy from supine:
            # 0.0 % of steps earned anything and 0/10 episodes saw a single toy. That is not a
            # sparse reward, it is no reward -- supine the eye points at the sky and nothing a
            # random policy does brings a toy into the fovea. With '--isr' the start angle is
            # drawn from Beta(1,3) over 0-180 degrees, some episodes begin where the reward is
            # non-zero, and the figure becomes 0.7 % of steps and 1/10 episodes (2.5 % and 2/10
            # at '--look_fovea=0.6'). 'ISRCallback' then anneals it away at 75 % of training, so
            # the final policy is still comparable to a non-ISR run.
            print("WARNING: 'look_reward' without 'isr'. A random policy starting supine earns "
                  "exactly zero looking reward -- there is nothing to bootstrap from. Pass "
                  "'--isr' unless you are deliberately measuring that.")
        if not rotation_reward and done_active:
            # Terminating the episode on success would make rolling *cost* MIMo the rest of the
            # episode's looking reward, i.e. the reward would actively select against the
            # behaviour the experiment is about. Same reasoning as '--no_done_active' under HER.
            raise ValueError("'rotation_reward=False' requires 'done_active=False' "
                             "(--no_done_active): ending the episode on a roll deletes the "
                             "looking reward the roll was supposed to earn.")

        self.reward_success=reward_success
        self.isr=isr
        self.nopen=nopen
        self.pbrs=pbrs
        self.pbrs_w=pbrs_w
        self.steps_after_reset=steps_after_reset
        self.pen_factor=pen_factor
        self.goal_function=goal_function
        self.gravity_goal_eps=gravity_goal_eps
        self.gravity_reference_samples=gravity_reference_samples
        self.goal_tolerance=goal_tolerance

        # 'gravity' goal function: the reference posture vectors recorded at the end of the
        # constructor, and a flag saying whether they exist yet -- 'sample_goal' runs before they
        # do, from inside 'MIMoEnv.initialize'.
        self.reference_goals_created = False
        self.prone_reference_goal = None
        self.supine_reference_goal = None
        self.reference_goal_std = {}
        # The running estimate of the gravity direction in the vestibular site's frame, and the
        # sim time it was last advanced to. Reset to None on every episode so the next read
        # re-seeds it from the accelerometer while MIMo is still settled.
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

        # Highest rotation reached during the episode that is currently running. Reported
        # through info['episode_rho_max'], which is what the callbacks and the evaluation
        # protocol read; reset to None at the start of every episode.
        self._episode_max_achieved = None


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

        # Built here rather than above, because it needs the compiled model that
        # 'super().__init__' produces.
        self.look = None
        if self.look_reward:
            self.look = LookReward(self, cameras=EYES[look_eyes], weight=look_w,
                                   habituation_steps=look_habituation_steps,
                                   recovery_steps=look_recovery_steps, fovea=look_fovea,
                                   novelty_weight=look_novelty_w,
                                   seen_threshold=look_seen_threshold)

        if self.goal_function == 'gravity':
            print("Creating prone and supine reference goals ('gravity').")
            self.create_prone_and_supine_reference_goals()
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

        Overridden to (re)resolve the MuJoCo ids the 'gravity' goal reads. 'set_embodiment'
        rebuilds 'self.model'/'self.data' from a different scene XML, which invalidates any
        cached index, and the base implementation ends by calling 'sample_goal', which for that
        goal function reads the achieved goal -- so the ids have to exist before
        'super().initialize()' runs, not after.

        The recorded reference postures are deliberately *not* re-recorded here: the gravity goal
        lives in the +-1 units of a direction cosine, which do not depend on the scene.
        """
        if self.goal_function == 'gravity':
            self._vestibular_site_id = self.model.site('vestibular').id
            self._gravity_body_ids = [self.model.body(name).id
                                      for name in GRAVITY_GOAL_BODIES]
            self._g_site = None
            self._g_site_time = None

        super().initialize()

    def create_prone_and_supine_reference_goals(self):
        """ Record the reference goal vector in prone and in supine ('gravity' only).

        These are the two 'desired_goal's: starting prone, the target is the supine reference,
        and vice versa. Both are recorded once, at construction.

        Two things differ from a naive single-shot implementation, and both were bugs waiting to
        happen:

        * The reference is **averaged over 'gravity_reference_samples' resets with ISR off**. A
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
                for _ in range(self.gravity_reference_samples):
                    self.reset_model()
                    samples.append(np.asarray(self.get_achieved_goal(), dtype=np.float64))
                samples = np.asarray(samples, dtype=np.float64)
                reference = samples.mean(axis=0)
                spread = samples.std(axis=0)

                if position == 'prone':
                    self.prone_reference_goal = reference
                else:
                    self.supine_reference_goal = reference
                self.reference_goal_std[position] = spread

                print(f"Reference goal ({position}, n={self.gravity_reference_samples}):")
                for label, value, sd in zip(self.goal_labels(), reference, spread):
                    print(f"    {label:<22} {value:+.3f}  (sd {sd:.3f})")
        finally:
            self.isr = saved_isr
            self.starting_position = saved_position
            self.np_random.bit_generator.state = rng_state
            self.reference_goals_created = True

    def goal_labels(self):
        """ Names of the goal dimensions, in order. For logging and diagnostics. """
        if self.goal_function == 'gravity':
            return [f'gravity_x_in_{name}_frame' for name in GRAVITY_GOAL_BODIES]
        return ['rho']


    def set_embodiment(self, morph_age, physio_age):
        """ Sets the embodiment to the MuJoCo .xml file that fits to the
        morphological age 'morph_age' and physiological age 'pyhsio_age'. """
        xml_path = scene_path(physio_age, morph_age, self.playroom)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.fix_top_camera_rotation_supine()

        self.initialize()
        # Re-resolve the toy geom ids and rebuild the segmentation renderer against the new
        # model, exactly as 'initialize()' does for the gravity goal's site and body ids.
        if getattr(self, 'look', None) is not None:
            self.look.initialize()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.reset()

        # This was here for testing that MGC curriculum would actually grow MIMo. I checked
        # this by rendering after 'set_embodiment' and checking that the body has grown.
        # render_top_down_and_save(self, self.starting_position, '/home/leon/MIMo', f'test_embodiment_age{morph_age}')



    def disable_isr(self):
        self.isr = False

    def _as_goal_batch(self, goal):
        """ Reshape a goal argument to (N, goal_dim), whatever shape it arrived in.

        HER hands over batches of shape (N, 1), the environment step hands over a single (1,),
        and the SB3 checker hands over both. Everything downstream works on (N, 1).
        """
        return np.asarray(goal, dtype=np.float64).reshape(-1, self.goal_dim)

    @property
    def goal_dim(self):
        """ Length of a single goal. 1 for the scalar 'cos' rotation, one entry per body for
        'gravity'.

        Kept as a property rather than inlined because 'compute_reward' and '_info_block' pass it
        around, and because the width is what separates the two goal functions.
        """
        if self.goal_function == 'gravity':
            return len(GRAVITY_GOAL_BODIES)
        return 1

    def _success_mask(self, achieved, desired):
        """ Success as a (N,) boolean array, from goals already shaped (N, goal_dim).

        Three rules, all of them pure functions of the two arguments:

        * 'cos', default: 'achieved >= desired'. The goal is a rotation threshold, and
          overshooting it is still success.
        * 'cos' with 'goal_tolerance' set: '|achieved - desired| <= tolerance'. The threshold
          becomes a band, so overshooting a goal is no longer success. See the constructor for
          why -- it is what gives the critic a distance to climb under HER.
        * 'gravity': '||achieved - desired|| <= gravity_goal_eps'. A threshold comparison makes
          no sense on a vector -- and this one runs from +1 (supine) to -1 (prone), so
          'achieved >= desired' would be satisfied at the *start* of a prone->supine episode.
          Equality never happens on continuous sensor readings either, so the goal is a point and
          success is a ball around it.

        The last two are the same rule; they differ only in which radius they read, because the
        two are calibrated on different scales (rho in [0, 1] against the +-1 of a direction
        cosine).
        """
        if self.goal_function == 'gravity':
            return np.linalg.norm(desired - achieved, axis=-1) <= self.gravity_goal_eps
        if self.goal_tolerance is not None:
            return np.linalg.norm(desired - achieved, axis=-1) <= self.goal_tolerance
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
        self._episode_max_achieved = None

        self.goal = self.sample_goal()
        self._prev_achieved_goal = None

        # Every episode starts with every toy novel again.
        if self.look is not None:
            self.look.reset()

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

        A flat Box of shape ('goal_dim',): (1,) for 'cos', (2,) for 'gravity'. Flatness is what
        HerReplayBuffer needs -- it relabels by writing into an array, and SB3 cannot take a
        nested dict as a goal space.
        """
        return spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float64)

    def get_desired_goal_obs(self):
        return self.goal

    def sample_goal(self):
        """ Returns the goal rotation.

        By default this is the fixed rotation that :meth:`.is_success` used to hardcode: 0.95 for
        a full roll, 0.5 under 'success_at_side_lying'. Moving the threshold from the success
        check into the goal is what lets the success check be a pure function of its arguments.

        With 'goal_tolerance' set the fixed full-roll target is 1.0 rather than 0.95, because
        success is then a band around the goal rather than a threshold. The two describe the same
        task -- rho is capped at 1.0 -- but the band is what changes the *relabelled* goals.

        If 'goal_low'/'goal_high' are set, the target is instead drawn uniformly from that range
        on every reset. HER needs this: with a constant desired_goal the policy has no reason to
        condition on the goal input, and the relabelled transitions describe goals that are never
        asked for at evaluation time. Evaluation should always pin the goal to 0.95.

        The 'gravity' goal function ignores all of that: its target is the reference vector
        recorded in the *opposite* starting position, fixed per episode rather than sampled.

        Returns:
            np.array[float]: The target goal, shape ('goal_dim',).
        """
        if self.goal_function == 'gravity':
            # Fixed per starting position. HER still sees plenty of goal variation, because every
            # relabelled goal is an achieved gravity vector from the trajectory -- which is why
            # this goal function trains without the --goal_low/--goal_high range 'cos' needs
            # (14/16 seeds against 3/16; docs/roll_over.md 3.6).
            #
            # The references are recorded at the very end of the constructor. Observation calls
            # before that point reach this function while they do not exist yet, so fall back to
            # the current achieved goal -- it is only used to shape the observation space.
            if not self.reference_goals_created:
                return self.get_achieved_goal()
            if self.starting_position == 'prone':
                return self.supine_reference_goal.copy()
            return self.prone_reference_goal.copy()

        if self.goal_low is not None:
            # A degenerate range means a fixed goal. Return it without drawing: consuming a
            # random number would shift every later draw from the same generator, in particular
            # the initial joint randomisation in 'put_in_starting_position', which changes which
            # episodes you get. That made evaluation results depend on whether the goal was
            # pinned via goal_low/goal_high or left at the default (measured: 94% vs 98% on the
            # same policy and the same seeds).
            if self.goal_low == self.goal_high:
                return np.array([float(self.goal_low)])
            return np.array([self.np_random.uniform(self.goal_low, self.goal_high)])

        if self.goal_tolerance is not None:
            # With the band criterion the target is the *end* of the rotation, not the point at
            # which the threshold is crossed. rho never exceeds 1.0, so '|rho - 1.0| <= 0.05' is
            # exactly the old 'rho >= 0.95' and the task is unchanged -- see the constructor.
            # Under '--side_lying' the value stays 0.5, but note it then means "stop at side
            # lying" rather than "reach at least side lying".
            return np.array([0.5 if self.success_at_side_lying else 1.0])

        return np.array([0.5 if self.success_at_side_lying else 0.95])

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

    
    def get_dot_local_x_to_global_z(self, body_name):
        """ Returns the dot product of the local x axis to the global z axis for the specified body.
        This is just the R[2,0] entry in the rotation matrix.

        Parameters:
            body_name (str): Name of the body to get the rotation for.

        Returns:
            float: The dot product between the body's local x axis and the global z axis.
        """
        return self.data.body(body_name).xmat.reshape(3, 3)[2, 0]


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
    

    def get_achieved_goal_gravity(self):
        """ The gravity direction in each of 'GRAVITY_GOAL_BODIES', x component.

        One entry per body, all on the same scale: +1 supine, -1 prone -- the same scale as
        'get_dot_local_x_to_global_z(body)', but reconstructed from the gyroscope and the
        proprioceptive joint chain instead of from the root free joint that proprioception does
        not report.

        The estimate is a running one, advanced lazily: the first read after a reset seeds it from
        the accelerometer (the one moment MIMo is demonstrably at rest, so specific force IS
        gravity), and every later read integrates the gyroscope forward to the current 'data.time'.
        Keying on the simulation clock rather than on a step counter makes this idempotent -- the
        environment reads the achieved goal several times per step (observation, reward, info) and
        all of them must see the same value.

        Note this makes 'get_achieved_goal' history-dependent, unlike 'cos'. 'compute_reward'
        stays pure -- it only ever sees the stored achieved/desired pair -- so HER relabelling is
        unaffected.
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
        """ The achieved goal, from whichever goal function this run was configured with. """
        if self.goal_function == 'gravity':
            return self.get_achieved_goal_gravity()
        return self.get_achieved_goal_cos()

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
        # the achieved goal of the previous step -- the
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

        # Running maximum over the episode, reset to None in 'reset_model'.
        rho = float(np.asarray(achieved_goal).reshape(-1)[0])
        self._episode_max_achieved = (rho if self._episode_max_achieved is None
                                      else max(self._episode_max_achieved, rho))
        # On the last step of an episode this is the episode's maximum, which is the quantity the
        # evaluation protocol reports. Note that 'side_lying' and 'rolled_over' above describe the
        # *final* step instead, so the two disagree whenever MIMo rolls and then rolls back.
        info['episode_rho_max'] = self._episode_max_achieved

        # Goal-INDEPENDENT reward terms, stored so HER can reuse them verbatim for the virtual
        # transitions it builds. Requires HerReplayBuffer(copy_info_dict=True).
        info['ctrl_cost'] = self.compute_penalization()
        if self.look is not None:
            # Computed here and carried through 'info' for the same reason as the control cost:
            # it depends on the rendered scene and on the habituation state, neither of which
            # 'compute_reward' may touch. It is also goal-independent, so HER may reuse it
            # verbatim on a relabelled transition.
            look, visible = self.look.step()
            info['look_reward'] = look
            info['look_visible'] = float(visible.sum())
            info['look_n_toys'] = float(np.count_nonzero(visible > 0.001))
            # Share of the toys looked at so far this episode. This is the headline metric of the
            # experiment: the playroom is built so that it cannot reach 1.0 without a roll.
            info['look_coverage'] = self.look.coverage
        if self._prev_achieved_goal is not None:
            info['prev_achieved_goal'] = self._prev_achieved_goal

        # 02.09.2026 Recompute the reward now that 'info' is complete.
        #
        # 'MIMoEnv.step' calls 'compute_reward' from inside 'super().step(action)' above, i.e.
        # BEFORE any of the keys written below exist. For 'ctrl_cost' and 'prev_achieved_goal'
        # that is harmless -- '_info_column'/'_info_block' fall back to the live simulation
        # values, which are exactly what those keys hold, so the two agree to the bit. For
        # 'look_reward' there is no live fallback and the default is 0.0, so **the looking reward
        # was computed, logged, and never actually paid**: under '--no_rotation_reward' the reward
        # MIMo received was minus the control cost alone, whose optimum is to lie still. The
        # symptom was 'ep_rew_mean' -202 against 'ep_look_reward' 400, with 'raw_ctrl_cost'
        # falling monotonically as SAC learned to stop moving; under '--nopen' it became
        # 'ep_rew_mean' exactly 0.0 while the looking reward was still being logged.
        #
        # Recomputing here rather than adding the term by hand keeps one definition of the reward,
        # the one HER also calls. For every configuration without a looking reward this is a
        # no-op by the argument above, which is what 'goalenv_check.py' verifies.
        reward = self.compute_reward(self.get_achieved_goal(), self.goal, info)

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

        look = self._info_column(info, 'look_reward', 0.0, n) if self.look_reward else 0.0

        if not self.rotation_reward:
            # rho contributes nothing. What is left is exactly what MIMo can see, minus what it
            # costs him to move -- and rho is still measured, through 'info', so the run can be
            # scored on rolling without rolling ever having been rewarded.
            reward = np.broadcast_to(np.asarray(look, dtype=np.float64), (n,)) - quad_ctrl_cost
            return reward if batched else float(np.asarray(reward).reshape(-1)[0])

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

        reward = reward - quad_ctrl_cost + look
        return reward if batched else float(reward[0])

    def _potential(self, achieved, desired):
        """ Potential of a state, as a pure function of the achieved and desired goal.

        The negative euclidean distance between the two, taken over the goal dimensions. For the
        a scalar goal that is just '-|desired - achieved|'. Both arguments must already be
        shaped (N, goal_dim) -- use ':meth:`._as_goal_batch`'.

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

        Note this deliberately does NOT scale ':attr:`.gravity_goal_eps`': the success radius
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

        PBRS has to reconstruct 'prev_achieved_goal' per transition for HER's relabelled
        batches, shaped (n, goal_dim) rather than flat.
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

