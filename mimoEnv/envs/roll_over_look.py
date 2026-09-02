""" The looking reward for the roll-over playroom.

01.09.2026 The motivation. MIMo's roll-over reward has so far been the rotation itself: rho, the
dot product of the hip and chest local x-axes with global z, either directly or through
potential-based shaping. That is a reward an infant does not have. What an infant does have is a
reason to turn: something worth looking at, off to the side, that he cannot see while lying on his
back. This module supplies that reward, and nothing else -- rho stays in 'info' as the *measured
outcome*, so "did looking produce rolling" is a question the run can answer rather than assume.

**How "MIMo looks at a toy" is measured: a segmentation render from his own eye cameras.**
MuJoCo renders the scene with each geom's id encoded in the pixel colour, so the fraction of the
image belonging to a toy is exactly what MIMo can see of it -- including occlusion by his own arm,
which is the whole point. A purely geometric test (angle between the eye axis and the toy's
position) is free, but it counts a toy hidden behind MIMo's own shoulder as seen, and getting the
body out of the way of the view is precisely the behaviour this reward is meant to select for.

**Interesting means new.** A reward proportional to visible toy area alone is maximised by
finding the best view once and holding still, so the reward is built around *coverage*: how many
different toys MIMo has managed to look at in this episode. Each toy pays a one-off bonus the
first time it is properly in view (:attr:`.novelty_weight`), and a continuous term scaled by
(1 - familiarity) supplies the gradient that gets it there. Familiarity grows while a toy is in
view and, by default, never recovers inside an episode -- a toy already collected is worth nothing
more, so the only way to keep earning is to find another one.

That makes the layout do the real work. The playroom is arranged so that MIMo's reachable set of
toys *while supine* is a strict subset of his reachable set *while prone*: three toys sit towards
his feet where he can see them lying on his back, and seven sit behind his head and low to the
floor, where his own body occludes the sight line until he has turned over. Full coverage is
therefore unreachable without the roll, by geometry rather than by tuning. See the header of
``mimoEnv/assets/roll_over/generate_playroom_scenes.py`` for the ray-cast probe behind that.

**Purity.** The looking reward does not depend on the goal, so like the control cost it is
computed in 'step()' and passed through 'info', where 'compute_reward' reads it back. See "The
purity contract" in CLAUDE.md: anything that reads live simulation state from inside
'compute_reward' turns HER relabelling into a silent no-op.
"""

import numpy as np
import mujoco


TOY_GEOM_PREFIX = "toy_"
""" Geoms whose name starts with this are what MIMo is rewarded for looking at.

A prefix rather than a list, so that editing the playroom in
`mimoEnv/assets/roll_over/generate_playroom_scenes.py` cannot silently leave this module behind.
The stands holding the raised toys up are named `stand_*` and deliberately do not match.

:meta hide-value:
"""

# Resolution of the segmentation render, per eye. Independent of '--vision_resolution': this image
# is never shown to the policy, it is only counted. 32x32 is 1024 samples of the field of view,
# which resolves a 15 cm toy at 0.5 m (~17 degrees across a 60 degree fovy, so ~9 pixels wide) an
# order of magnitude better than the reward needs. Cost is dominated by fixed per-render overhead
# and barely moves with resolution -- measured 54.6 / 57.7 / 44.0 ms at 36 / 64 / 128 px before
# shadows were switched off -- so there is nothing to gain by going lower.
SEGMENTATION_RESOLUTION = 32


class LookReward:
    """ Measures what MIMo is looking at, and turns it into a reward.

    Attributes:
        env: The environment this is attached to.
        cameras (list): Camera names to render from, normally MIMo's eyes.
        toy_geoms (dict): Toy name to geom id, rebuilt by :meth:`.initialize`.
        familiarity (np.ndarray): Per-toy familiarity in [0, 1], reset every episode.
    """

    def __init__(self, env, cameras=("eye_left",), weight=100.0, habituation_steps=50,
                 recovery_steps=0, resolution=SEGMENTATION_RESOLUTION, fovea=0.35,
                 novelty_weight=200.0, seen_threshold=0.01):
        """ Constructor.

        Args:
            env: The environment. Must expose 'model', 'data' and 'mujoco_renderer'.
            cameras: Camera names to render from. One eye is enough -- the two eyes are 2.5 cm
                apart and see essentially the same toys -- and each one costs a render.
            weight (float): Scale of the reward. A 15 cm toy at 0.5 m covers about 8 % of a
                60-degree field of view, and a side-lying view holds a few of them, so the raw
                per-step signal is of order 0.05-0.2 and the default 100 turns that into 5-20 per
                step. For scale: the default control cost is 'pen_factor=0.02' over 92 actuators,
                i.e. at most 1.84 per step.
            habituation_steps (int): Steps of holding a toy in full view that halve what it pays.
            recovery_steps (int): Steps of not seeing a toy that halve its familiarity again.
                **0 disables recovery**, which is the default: within one episode a toy already
                looked at stays spent, so the episode reward is essentially "how many different
                toys did you find". Set it to a positive number to get the older decaying
                habituation back.
            novelty_weight (float): One-off bonus the first time a toy crosses
                `seen_threshold`. Ten toys at the default 200 is 2000 per episode for full
                coverage, against roughly 1500 for the continuous term over 500 steps, so
                coverage is the dominant thing to optimise without the gradient term vanishing.
            seen_threshold (float): Foveal share at which a toy counts as seen. 0.01 is about a
                toy squarely in view rather than clipped by the edge of the field; it is not a
                free parameter, see the note in :meth:`.step`.
            resolution (int): Edge length of the segmentation render.
            fovea (float): Width of the Gaussian that weights a toy by how central it is in the
                field of view, in units of half the image. ``None`` weights every pixel equally.
                See :meth:`._foveal_weights` for why this is not cosmetic.
        """
        self.env = env
        self.cameras = list(cameras)
        self.weight = weight
        self.resolution = resolution
        self.fovea = fovea
        # Per-step multiplicative factors, from the half-lives. 'habituation_rate' applies to the
        # gap to 1, so a toy held in full view for 'habituation_steps' ends at familiarity 0.5.
        self.habituation_rate = 1.0 - 0.5 ** (1.0 / max(habituation_steps, 1))
        self.recovery_rate = 1.0 if recovery_steps <= 0 else 0.5 ** (1.0 / recovery_steps)
        self.novelty_weight = novelty_weight
        self.seen_threshold = seen_threshold
        self.seen = np.zeros(0, dtype=bool)
        self.toy_geoms = {}
        self.familiarity = np.zeros(0)
        self._camera_ids = []
        self._last_visible = np.zeros(0)
        self._renderer = None
        self._weights = self._foveal_weights()
        self.initialize()

    def _foveal_weights(self):
        """ Per-pixel weight, peaked at the centre of the field of view.

        01.09.2026 Not cosmetic -- it is what stops the reward from being maximised by contorting
        the neck. Measured on the landscape sweep with equal weights: from *any* roll angle,
        including flat on his back, MIMo can bring some toy of a 360-degree ring into the corner
        of his eye by driving 'head_tilt' and 'head_swivel' to their limits simultaneously, so
        supine scored 0.18 against prone's 0.31 and the incentive to finish the roll was thin. A
        toy at the edge of a maximally strained view is not what "looking at something" means, and
        an infant's peripheral acuity is poor enough that it is not what it means for him either.

        The result is normalised so that a toy filling the whole image scores 1.0, which keeps the
        number readable as "share of the field of view", now measured foveally.

        Returns:
            np.ndarray: An (H, W) weight array, or ``None`` if :attr:`.fovea` is ``None``.
        """
        if self.fovea is None:
            return None
        axis = np.linspace(-1.0, 1.0, self.resolution)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        w = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * self.fovea ** 2))
        return w / w.sum()

    def initialize(self):
        """ Resolves the toy and camera ids against the current model.

        Called from the constructor and again from
        :meth:`~mimoEnv.envs.roll_over.MIMoRollOverEnv.set_embodiment`, which swaps in a whole new
        model mid-training and invalidates every cached id.
        """
        model = self.env.model
        self.toy_geoms = {}
        for gid in range(model.ngeom):
            name = model.geom(gid).name
            if name.startswith(TOY_GEOM_PREFIX):
                self.toy_geoms[name] = gid
        if not self.toy_geoms:
            raise ValueError(
                f"No geoms named '{TOY_GEOM_PREFIX}*' in the scene. The looking reward needs the "
                f"playroom: pass 'playroom=True' (--playroom), and regenerate the scenes with "
                f"'python mimoEnv/assets/roll_over/generate_playroom_scenes.py' if they are "
                f"missing.")
        self._camera_ids = [model.camera(name).id for name in self.cameras]
        self.familiarity = np.zeros(len(self.toy_geoms))
        self.seen = np.zeros(len(self.toy_geoms), dtype=bool)
        self._last_visible = np.zeros(len(self.toy_geoms))
        # Own renderer rather than the env's. Two reasons, both load-bearing:
        #  - gymnasium 1.0.0's 'OffScreenViewer.render(segmentation=True)' decodes the id colour
        #    as 'rgb[:, :, 1] * 2**8' on a uint8 array, which under numpy 2 raises
        #    "OverflowError: Python integer 256 out of bounds for uint8". That path cannot be
        #    used at all here.
        #  - The env's renderer is shared with the video output and the vision observation, and a
        #    segmentation pass has to flip scene flags on it.
        # 'eval_noise_baseline.py' builds its own renderer for the same reason. Note its channel
        # order: 'mujoco.Renderer' returns (objid, objtype), which is the REVERSE of the
        # gymnasium viewer.
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(model, self.resolution, self.resolution)
        self._renderer.enable_segmentation_rendering()

    @property
    def toy_names(self):
        """ Toy names, in the order used by :attr:`.familiarity` and the visibility vectors. """
        return list(self.toy_geoms)

    def reset(self):
        """ Clears the coverage. Every episode starts with every toy unseen again. """
        self.familiarity = np.zeros(len(self.toy_geoms))
        self.seen = np.zeros(len(self.toy_geoms), dtype=bool)
        self._last_visible = np.zeros(len(self.toy_geoms))

    @property
    def coverage(self):
        """ Fraction of the toys looked at so far this episode. """
        return float(self.seen.mean()) if self.seen.size else 0.0

    def visible_fractions(self):
        """ Share of the field of view each toy currently occupies.

        Renders the scene once per camera with MuJoCo's segmentation encoding, which writes each
        geom's id into the pixel instead of its colour, and counts pixels. Averaged over the
        cameras, so the result is a fraction in [0, 1] whatever the number of eyes.

        Returns:
            np.ndarray: One fraction per toy, ordered as :attr:`.toy_names`.
        """
        geom_ids = np.fromiter(self.toy_geoms.values(), dtype=np.int64,
                               count=len(self.toy_geoms))
        totals = np.zeros(len(self.toy_geoms))
        for camera_id in self._camera_ids:
            self._renderer.update_scene(self.env.data, camera=camera_id)
            # After 'update_scene', which rebuilds the flags from the model.
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            seg = self._renderer.render()
            objid, objtype = seg[:, :, 0], seg[:, :, 1]
            ids = np.where(objtype == mujoco.mjtObj.mjOBJ_GEOM, objid, -1)
            hit = ids[..., None] == geom_ids
            if self._weights is None:
                totals += hit.sum(axis=(0, 1)) / (self.resolution ** 2)
            else:
                totals += np.tensordot(self._weights, hit, axes=([0, 1], [0, 1]))

        return totals / max(len(self._camera_ids), 1)

    def step(self):
        """ One step of the looking reward, advancing the habituation state.

        Returns:
            (float, np.ndarray): The reward for this step, and the per-toy visible fractions.
        """

        visible = self.visible_fractions()

        # Coverage. A toy counts once, the first time it is squarely in view. The threshold is
        # what stops MIMo from collecting a toy by catching it in the far corner of a strained
        # view: measured over the full head range, a toy clipped by the edge of the field scores
        # a few thousandths of foveal share, while one he has actually turned towards scores
        # 0.03-0.35.
        newly = (~self.seen) & (visible >= self.seen_threshold)
        self.seen |= newly

        reward = (self.weight * float(np.sum(visible * (1.0 - self.familiarity)))
                  + self.novelty_weight * float(np.count_nonzero(newly)))
        # Habituate on what is in view, recover on what is not. Both are applied every step, so a
        # toy at the edge of the field of view drifts towards an equilibrium rather than latching.
        self.familiarity += (1.0 - self.familiarity) * self.habituation_rate * np.minimum(
            visible / _SATURATING_FRACTION, 1.0)
        self.familiarity *= self.recovery_rate ** (1.0 - np.minimum(
            visible / _SATURATING_FRACTION, 1.0))
        self._last_visible = visible
        return reward, visible

    def close(self):
        """ Releases the segmentation renderer's GL context. """
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# What counts as "in full view" for the habituation clock. A single 15 cm toy at 0.5 m covers
# about 8 % of a 60-degree field of view and MIMo can never fill his eye with one, so habituation
# is scaled against a reachable fraction instead of against 1.0 -- otherwise the clock would run
# twelve times too slowly and habituation would never engage inside a 500-step episode.
_SATURATING_FRACTION = 0.08
