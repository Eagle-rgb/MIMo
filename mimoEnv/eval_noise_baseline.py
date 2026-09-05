""" Does undirected motor noise roll MIMo over? -- the null baseline for the roll-over task.

    MUJOCO_GL=osmesa python mimoEnv/eval_noise_baseline.py [--episodes=30]

This is the control every learned result is measured against: if a policy that never looks at an
observation rolls, then the task is solved by chance and no training result means anything. The
three noise colours are the ones the exploration literature actually proposes as action priors --
white (beta=0, independent per step), pink (beta=1, 1/f, Eberhard et al., ICLR 2023) and red
(beta=2, Brownian, the Ornstein-Uhlenbeck-like end of the family) -- plus two references:

* 'uniform'  -- ``action_space.sample()``, the sampler used everywhere else in this repo. It is
                white in time but uniform rather than Gaussian in amplitude (std 0.577).
* 'zero'     -- no action at all. The floor: whatever rho does while MIMo just lies there and the
                simulation settles. Without it, "noise reaches rho 0.1" is not interpretable.

26.08.2026 Why this is a separate script and not a flag on illustrations.py: there is nothing to
train here. A noise policy has no parameters, so no checkpoint, no data.yml and no save path --
running it through the training CLI would only mean disabling most of that CLI.

NB `mimoComposer/h1_latent_probe.py` also drives MIMo with pink noise, but for a different purpose:
there the noise fills a CLTT buffer whose *latent space* is the object of study, and the rollouts
are never scored for rolling. Its white-vs-pink comparison is also not amplitude-matched
(uniform[-1,1] has std 0.577, pink at sigma=0.3 has std 0.3), which is fine for buffer coverage and
useless as a baseline. Here every colour is driven at the *same* sigma by the same
`ColoredActionNoise` code path, so the only thing that differs between white, pink and red is the
spectral exponent, and sigma is swept separately.

Protocol: identical to mimoEnv/eval_rollover.py, and imported from it rather than restated, so the
numbers land on the same scale as every trained run -- ISR off, goal pinned to rho=0.95, episodes
never cut short, rho_max read off the simulation. Reward settings are irrelevant (nothing here is
trained or scored by reward), which is why no data.yml is needed.

Memory: one MIMo env is ~3.6 GB RSS. Posture is the *outer* loop and the env is rebuilt only when
the posture changes, so at most one env exists at a time -- never run two of these in parallel.
"""
import argparse
import gc
import json
import shutil
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

import mimoEnv  # noqa: F401  (registers MIMoRollOver-v0)
from mimoEnv.eval_rollover import (
    DEFAULT_EPISODE_STEPS,
    ROLL_THRESHOLD,
    SIDE_LYING_THRESHOLD,
    env_kwargs,
)

import gymnasium as gym


def build_env(starting_position, age_morph, age_physio, render=False, use_muscle=False):
    """The evaluation env of eval_rollover.py.

    env_kwargs({}, ...) is the protocol: isr=False, goal pinned, done_active=False. The empty
    config is deliberate -- its defaults (PBRS off, pen_factor 0.02, dense reward) only shape a
    reward nothing here reads.
    """
    # env_kwargs already knows how to swap in MuscleModel; going through it keeps the noise
    # baseline on exactly the same construction path as a real evaluation.
    kwargs = env_kwargs({'use_muscle': use_muscle}, starting_position, ROLL_THRESHOLD)
    # The measurement pass never renders, and an unused renderer is memory for nothing.
    kwargs['render_mode'] = 'rgb_array' if render else None
    kwargs['age_morph'] = age_morph
    kwargs['age_physio'] = age_physio
    return gym.make('MIMoRollOver-v0', **kwargs).unwrapped


def make_sampler(condition, sigma, env, seed, seq_len):
    """An action source and its per-episode reset hook, as ``(sample, reset)``.

    'condition' is 'zero', 'uniform', or a colour name mapping to a spectral exponent.
    """
    if condition == 'zero':
        zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        return (lambda: zero), (lambda: None)

    if condition == 'uniform':
        # Gymnasium keeps env.np_random and action_space._np_random separate, so reset(seed=...)
        # does NOT seed the sampler. Without this line the whole baseline is unreproducible.
        env.action_space.seed(int(seed))
        return (lambda: env.action_space.sample()), (lambda: None)

    from pink import ColoredActionNoise

    beta = BETAS[condition]
    # Same trap one level down: without an explicit rng, powerlaw_psd_gaussian reaches for a fresh
    # np.random.default_rng() that no --seed touches.
    noise = ColoredActionNoise(beta=beta, sigma=sigma,
                               action_dim=int(env.action_space.shape[0]),
                               seq_len=seq_len,
                               rng=np.random.default_rng(int(seed)))
    low, high = env.action_space.low, env.action_space.high
    # 02.09.2026 The noise is zero-mean, the action box is not necessarily. SpringDamperModel gives
    # 46 actuators in [-1, 1], MuscleModel 92 muscles in [0, 1], and dropping zero-mean noise
    # straight into the second one would clip half of every sample to zero -- the muscles would sit
    # at rest most of the time and sigma would stop meaning what it means for the torque model.
    # Mapping onto the box's centre and half-width instead keeps sigma in units of "fraction of the
    # available range" for both, and is bit-identical for [-1, 1] (centre 0, half-width 1).
    centre = (low + high) / 2.0
    half_range = (high - low) / 2.0
    # Clipping matters at sigma=1: a Gaussian at the edge of the box spends a third of its mass
    # outside it, so sigma=1 is close to a bang-bang policy. That is the point of sweeping sigma.
    return (lambda: np.clip(centre + half_range * noise(), low, high)), noise.reset


BETAS = {'white': 0.0, 'pink': 1.0, 'red': 2.0}


def run_condition(env, condition, sigma, episodes, episode_steps, seed0, seq_len):
    """Roll out one action source and score it exactly as eval_rollover.evaluate does."""
    sample, reset_noise = make_sampler(condition, sigma, env, seed0, seq_len)
    rho_max = np.empty(episodes, dtype=float)
    # rho at the *last* step, next to the episode maximum. Scoring is by rho_max, because that is
    # what eval_rollover.py does and the numbers have to stay comparable -- but a noise policy that
    # tips MIMo past 0.95 and lets him fall back has not done the same thing as a policy that rolls
    # and stays there, and only the gap between these two columns shows it.
    rho_end = np.empty(episodes, dtype=float)
    for episode in range(episodes):
        env.reset(seed=seed0 + episode)
        reset_noise()
        rho = float(env.get_achieved_goal_cos_mean()[0])
        best = rho
        for _ in range(episode_steps):
            env.step(sample())
            rho = float(env.get_achieved_goal_cos_mean()[0])
            best = max(best, rho)
        rho_max[episode] = best
        rho_end[episode] = rho
    return rho_max, rho_end


def upper_bound(successes, n, alpha=0.05):
    """One-sided 95 % Clopper-Pearson upper bound on the success rate.

    With zero successes this is the whole result: "0 of 30" is not a proof on its own, "the roll
    rate is below 9.5 % with 95 % confidence" is a statement. Closed form for k = 0
    (1 - alpha^(1/n)); scipy for the general case, and None if scipy is missing and k > 0.
    """
    if successes == 0:
        return 1.0 - alpha ** (1.0 / n)
    try:
        from scipy.stats import beta as beta_dist
    except ImportError:
        return None
    return float(beta_dist.ppf(1 - alpha, successes + 1, n - successes))


def summarise(label, posture, condition, sigma, rho_max, rho_end):
    rolled = int((rho_max >= ROLL_THRESHOLD).sum())
    side = int((rho_max >= SIDE_LYING_THRESHOLD).sum())
    n = int(rho_max.size)
    return {
        'label': label,
        'posture': posture,
        'condition': condition,
        'sigma': sigma,
        'episodes': n,
        'rolled': rolled,
        'roll_rate': rolled / n,
        'roll_rate_upper95': upper_bound(rolled, n),
        'side_lying': side,
        'side_lying_rate': side / n,
        'rho_mean': float(rho_max.mean()),
        'rho_std': float(rho_max.std()),
        'rho_best': float(rho_max.max()),
        'rho_end_mean': float(rho_end.mean()),
        # Of the episodes that touched 0.95, how many were still there at the last step.
        'held': int(((rho_max >= ROLL_THRESHOLD) & (rho_end >= ROLL_THRESHOLD)).sum()),
        'rho_max': rho_max.round(4).tolist(),
        'rho_end': rho_end.round(4).tolist(),
    }


def _overlay(frame, text_lines):
    """Burn rho into the frame. Without it a video of a noise policy is uninterpretable --
    'did that count as a roll?' is exactly the question the numbers answer and the picture does
    not."""
    import cv2

    frame = np.ascontiguousarray(frame)
    # On an RGBA frame the text has to bring its own alpha, or it is drawn into pixels that stay
    # fully transparent and is invisible in anything that honours the channel.
    dark = (0, 0, 0, 255) if frame.shape[2] == 4 else (0, 0, 0)
    light = (255, 255, 255, 255) if frame.shape[2] == 4 else (255, 255, 255)
    for i, line in enumerate(text_lines):
        origin = (10, 24 + 22 * i)
        # Black underlay first, so the text stays readable over both the bright floor and MIMo.
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, dark, 3, cv2.LINE_AA)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, light, 1, cv2.LINE_AA)
    return frame


def make_frame_source(env, camera, size, transparent=False):
    """How a frame is grabbed, as ``(grab, close)``. RGB, or RGBA when 'transparent'.

    'top' is the camera illustrations.py uses for its frame_1..4 stills: free camera, elevation
    -90, azimuth flipped for supine, and `render_top_down` re-aims `lookat` at the centre of
    MIMo's top-down bounding box on *every* frame. That last part is what keeps him centred while
    he slides around, which the env's own fixed cameras do not.

    render.utils.create_renderer hardcodes 240x240 -- fine for a still in a paper, too coarse for a
    video -- so the renderer is built here with the same call and a configurable size.

    Transparency: MuJoCo's renderer has no alpha channel, so the cutout comes from a second pass in
    segmentation mode over the same scene. That pass returns (geom id, object type) per pixel, and
    the rule for "this is MIMo" is `model.geom_bodyid[geom] != 0` -- body 0 is the worldbody, which
    is exactly the floor and the skybox. Naming the floor geom would have worked too, but breaks on
    any scene that names it differently; the worldbody test does not.
    """
    if camera != 'top':
        if transparent:
            raise SystemExit("--transparent needs --camera=top: the cutout comes from a second "
                             "segmentation pass, and the env's own renderer does not offer one.")
        return (lambda: env.mujoco_renderer.render(render_mode="rgb_array")), (lambda: None)

    import mujoco
    from render.utils import create_top_down_camera, render_top_down

    renderer = mujoco.Renderer(env.model, height=size, width=size)
    cam = create_top_down_camera(env.starting_position)
    if not transparent:
        return (lambda: render_top_down(env.data, renderer, cam)), renderer.close

    geom_bodyid = env.model.geom_bodyid

    def grab():
        # render_top_down first: it sets cam.lookat, so the segmentation pass below sees exactly
        # the same framing without having to recompute the centre.
        rgb = render_top_down(env.data, renderer, cam)
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(env.data, camera=cam)
            segmentation = renderer.render()
        finally:
            renderer.disable_segmentation_rendering()
        objid, objtype = segmentation[:, :, 0], segmentation[:, :, 1]
        is_geom = (objtype == mujoco.mjtObj.mjOBJ_GEOM) & (objid >= 0)
        # np.where over a clipped index: objid is -1 on background pixels and would wrap around.
        on_mimo = is_geom & (geom_bodyid[np.clip(objid, 0, None)] != 0)
        rgba = np.dstack([rgb, np.where(on_mimo, 255, 0).astype(np.uint8)])
        return rgba

    return grab, renderer.close


def render_episode(env, condition, sigma, episode, episode_steps, seed0, seq_len, out_path,
                   overlay=True, fps=None, camera='top', size=500, transparent=False):
    """Re-run one episode of the sweep with the renderer on and write it to 'out_path'.

    Two things make this an exact replay of the measured episode rather than a fresh sample:

    * the env is reseeded per episode (seed0 + episode), so its start state does not depend on
      what ran before it, and
    * the colored-noise process is *not* reseeded per episode -- its rng advances by one sequence
      on every reset() -- so episodes 0..episode-1 have to be replayed to reach the same noise
      stream. They are replayed without rendering, which is why this takes a while for a late
      episode and costs nothing in memory.

    Frames are streamed straight into the writer instead of collected in a list: 500 frames of
    480x480 RGB is ~345 MB held for no reason.

    With 'transparent' the frames carry an alpha channel, which mp4v cannot: they go out as a PNG
    sequence, ffmpeg encodes a VP9 WebM (`yuva420p`, the one widely-played format that keeps
    alpha), and the keyframes are cut from the PNGs rather than read back out of the video, so
    they keep full per-pixel alpha instead of VP9's compressed version of it.
    """
    import cv2

    sample, reset_noise = make_sampler(condition, sigma, env, seed0, seq_len)
    for warmup in range(episode):
        env.reset(seed=seed0 + warmup)
        reset_noise()
        for _ in range(episode_steps):
            env.step(sample())

    env.reset(seed=seed0 + episode)
    reset_noise()
    if fps is None:
        # Real time: one simulated second is one second of video.
        fps = int(round(1.0 / env.dt))

    grab, close_renderer = make_frame_source(env, camera, size, transparent)
    frame = grab()
    height, width = frame.shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    frame_dir = None
    writer = None
    if transparent:
        frame_dir = out_path[:-4] + '_frames'
        shutil.rmtree(frame_dir, ignore_errors=True)
        os.makedirs(frame_dir)
    else:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open a writer for {out_path}.")

    rho = float(env.get_achieved_goal_cos_mean()[0])
    best = rho
    # rho per *frame*, and frame i is the state after step i (frame 0 is the reset state), so this
    # trace indexes straight into the written video. That is what lets the keyframes be picked
    # afterwards without simulating the episode a second time.
    trace = []
    try:
        for step in range(episode_steps + 1):
            if step:
                env.step(sample())
                rho = float(env.get_achieved_goal_cos_mean()[0])
                best = max(best, rho)
                frame = grab()
            if overlay:
                frame = _overlay(frame, [
                    f"{condition} noise  sigma={sigma:g}  {env.starting_position}",
                    f"step {step:3d}/{episode_steps}   rho={rho:.3f}   max={best:.3f}",
                    "ROLLED (rho >= 0.95)" if best >= ROLL_THRESHOLD else "",
                ])
            trace.append(rho)
            out = np.asarray(frame, dtype=np.uint8)
            if transparent:
                cv2.imwrite(os.path.join(frame_dir, f"f{step:05d}.png"),
                            cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))
            else:
                writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
        close_renderer()
    if transparent:
        encode_alpha_video(frame_dir, out_path, fps)
    return best, rho, fps, np.asarray(trace), frame_dir


ALPHA_CODEC = ('qtrle', 'argb', '.mov')
# 26.08.2026 Not WebM/VP9. `-c:v libvpx-vp9 -pix_fmt yuva420p` is the recipe everyone quotes for
# transparent video, and ffmpeg 8 accepts it without a warning -- but decoding the result back to
# rgba returns a fully opaque frame, so the alpha is silently dropped. QuickTime RLE keeps it, is
# lossless, and plays in VLC and every editor; ProRes 4444 ('prores_ks', 'yuva444p10le') is the
# other option if a smaller file matters more than losslessness.


def encode_alpha_video(frame_dir, out_path, fps):
    """PNG sequence -> a video that actually carries the alpha channel.

    Verified rather than trusted: the encode is followed by decoding frame 0 back to rgba and
    checking that transparent pixels are still transparent. Getting this wrong is invisible in the
    file itself -- it just quietly gains a black background -- so it is worth the one extra call.
    """
    import subprocess

    codec, pix_fmt, suffix = ALPHA_CODEC
    video = out_path[:-4] + suffix
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(fps), '-i', os.path.join(frame_dir, 'f%05d.png'),
        '-c:v', codec, '-pix_fmt', pix_fmt, video,
    ], check=True)

    probe = os.path.join(frame_dir, '_alpha_check.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video,
                    '-frames:v', '1', '-pix_fmt', 'rgba', probe], check=True)
    import cv2
    decoded = cv2.imread(probe, cv2.IMREAD_UNCHANGED)
    os.remove(probe)
    if decoded is None or decoded.shape[2] != 4 or not (decoded[:, :, 3] == 0).any():
        raise RuntimeError(f"{video} lost its alpha channel during encoding.")
    return video


def write_keyframes(video_path, trace, count, out_prefix, frame_dir=None):
    """Pull 'count' frames out of an already-written video, from the start up to the roll.

    "Up to the roll" is the point: the interesting part of a lucky red-noise episode is over the
    moment rho crosses 0.95, and the remaining seconds of MIMo lying still would use up most of the
    frames. The cut is the *first* frame at or above the threshold; if the episode never rolled,
    the frame with the highest rho stands in for it.

    Frames are read sequentially rather than seeked: mp4v seeking by frame index is unreliable, and
    reading a few hundred frames costs nothing. Only the wanted ones are kept in memory. When the
    render kept a PNG sequence ('frame_dir', the transparent path), the wanted frames are read from
    it directly -- both because it is cheaper and because it preserves the alpha channel exactly.
    """
    import cv2

    rolled = np.flatnonzero(trace >= ROLL_THRESHOLD)
    last = int(rolled[0]) if rolled.size else int(np.argmax(trace))
    # linspace over [0, last] so the first frame is always the start posture and the last is always
    # the roll itself -- the two frames the figure has to contain.
    wanted = sorted(set(int(i) for i in np.linspace(0, last, count)))

    frames = {}
    if frame_dir:
        for i in wanted:
            frame = cv2.imread(os.path.join(frame_dir, f"f{i:05d}.png"), cv2.IMREAD_UNCHANGED)
            if frame is not None:
                frames[i] = frame
    else:
        capture = cv2.VideoCapture(video_path)
        index = 0
        try:
            while index <= wanted[-1]:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in wanted:
                    frames[index] = frame
                index += 1
        finally:
            capture.release()

    paths = []
    for position, i in enumerate(wanted):
        if i not in frames:
            continue
        path = f"{out_prefix}_key{position + 1}.png"
        cv2.imwrite(path, frames[i])
        paths.append((path, i, float(trace[i])))
    # The strip is the thing that actually goes into a document; the singles are for cropping.
    strip = [frames[i] for i in wanted if i in frames]
    strip_path = None
    if strip:
        strip_path = f"{out_prefix}_strip.png"
        cv2.imwrite(strip_path, np.hstack(strip))
    return paths, strip_path, last, bool(rolled.size)


def parse_render_spec(spec):
    """'prone:red:1.0:31' -> ('prone', 'red', 1.0, 31)."""
    parts = spec.split(':')
    if len(parts) != 4:
        raise SystemExit(f"--render takes posture:colour:sigma:episode, got {spec!r}.")
    posture, condition, sigma, episode = parts
    return posture, condition, float(sigma), int(episode)


HEADER = (f"{'posture':<8} {'condition':<9} {'sigma':>6} {'eps':>4} {'roll':>6} {'<=95%':>7} "
          f"{'held':>5} {'side':>6} {'rho_mean':>9} {'rho_best':>9} {'rho_end':>8}")


def print_row(row):
    bound = row['roll_rate_upper95']
    sigma = '-' if row['sigma'] is None else f"{row['sigma']:.2f}"
    bound = 'n/a' if bound is None else f"{bound:.1%}"
    print(f"{row['posture']:<8} {row['condition']:<9} {sigma:>6} {row['episodes']:>4} "
          f"{row['roll_rate']:>6.1%} {bound:>7} {row['held']:>5} {row['side_lying_rate']:>6.1%} "
          f"{row['rho_mean']:>9.4f} {row['rho_best']:>9.4f} {row['rho_end_mean']:>8.4f}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=30,
                        help="Episodes per (posture, condition, sigma) cell. 30 gives a 9.5 %% "
                             "upper bound on an all-zero roll rate; pooling the sigmas per colour "
                             "tightens it to 3.3 %%.")
    parser.add_argument('--episode_steps', type=int, default=DEFAULT_EPISODE_STEPS,
                        help="Horizon, the same 500 steps every trained run is evaluated over.")
    parser.add_argument('--postures', default='prone,supine')
    parser.add_argument('--colours', default='white,pink,red',
                        help="Comma-separated subset of white (beta=0), pink (1), red (2).")
    parser.add_argument('--sigmas', default='0.3,0.6,1.0',
                        help="Noise scales to sweep. Swept because a single amplitude cannot "
                             "distinguish 'noise does not roll' from 'this amplitude does not'.")
    parser.add_argument('--no_uniform', action='store_true',
                        help="Drop the action_space.sample() reference row.")
    parser.add_argument('--no_zero', action='store_true',
                        help="Drop the do-nothing floor row.")
    parser.add_argument('--seq_len', type=int, default=None,
                        help="Colored-noise sequence length; defaults to the episode horizon, "
                             "which is what the pink-noise paper prescribes.")
    parser.add_argument('--use_muscle', action='store_true',
                        help="Drive MuscleModel (92 muscles in [0, 1]) instead of the "
                             "spring-damper model (46 actuators in [-1, 1]). Not comparable to a "
                             "spring-damper baseline cell for cell: the muscle model also zeroes "
                             "the stiffness of the unactuated spine joints and cuts their damping "
                             "by 20, so the body itself differs.")
    parser.add_argument('--morph_age', type=int, default=9)
    parser.add_argument('--physio_age', type=int, default=9)
    parser.add_argument('--seed', type=int, default=1000,
                        help="Base seed. Episode i uses seed+i, so every condition sees the same "
                             "start states -- the colours differ only in their actions.")
    parser.add_argument('--json', default=None, help="Write the full result, per-episode rho_max "
                                                     "included, here.")
    parser.add_argument('--csv', default=None, help="Write the summary table here.")
    parser.add_argument('--render', action='append', default=None, metavar='POSTURE:COLOUR:SIGMA:EP',
                        help="Instead of measuring, re-render one episode of the sweep as a video, "
                             "e.g. --render=prone:red:1.0:31. Repeatable. The episode index is the "
                             "position in the sweep, so the JSON's per-episode rho_max says which "
                             "ones are worth looking at.")
    parser.add_argument('--render_dir', default='results/noise_baseline/video',
                        help="Where --render writes its mp4s.")
    parser.add_argument('--fps', type=int, default=None,
                        help="Video frame rate; defaults to real time (1/env.dt).")
    parser.add_argument('--no_overlay', action='store_true',
                        help="Do not burn the step counter and rho into the frames.")
    parser.add_argument('--camera', choices=('top', 'env'), default='top',
                        help="'top' is illustrations.py's top-down camera, re-centred on MIMo every "
                             "frame (render.utils.render_top_down) -- he stays in the picture even "
                             "as he slides. 'env' is the environment's own fixed camera.")
    parser.add_argument('--transparent', action='store_true',
                        help="Cut the floor and the skybox out: frames get an alpha channel from a "
                             "segmentation pass, keyframes become RGBA PNGs and the video is "
                             "written as a QuickTime-RLE .mov, since mp4 cannot carry alpha.")
    parser.add_argument('--keep_frames', action='store_true',
                        help="With --transparent, keep the intermediate PNG sequence.")
    parser.add_argument('--render_size', type=int, default=500,
                        help="Square resolution of the --camera=top renderer.")
    parser.add_argument('--keyframes', type=int, default=0, metavar='N',
                        help="With --render, also cut N stills out of the video, evenly spaced "
                             "from the start up to the frame where rho first reaches 0.95 (or to "
                             "the peak, if the episode never rolled), plus a single strip image.")
    args = parser.parse_args()

    postures = [p for p in args.postures.split(',') if p]
    colours = [c for c in args.colours.split(',') if c]
    sigmas = [float(s) for s in args.sigmas.split(',') if s]
    seq_len = args.seq_len or args.episode_steps
    for colour in colours:
        if colour not in BETAS:
            raise SystemExit(f"Unknown colour {colour!r}; choose from {sorted(BETAS)}.")

    if args.render:
        specs = [parse_render_spec(spec) for spec in args.render]
        # Grouped by posture for the same reason the sweep is: one 3.6 GB env at a time.
        for posture in dict.fromkeys(s[0] for s in specs):
            # The top-down camera brings its own renderer, so the env needs none -- and an
            # unused one is memory for nothing.
            env = build_env(posture, args.morph_age, args.physio_age,
                            render=args.camera == 'env', use_muscle=args.use_muscle)
            try:
                for _, condition, sigma, episode in (s for s in specs if s[0] == posture):
                    tag = '_muscle' if args.use_muscle else ''
                    out = os.path.join(args.render_dir,
                                       f"{posture}_{condition}_s{sigma:g}_ep{episode}{tag}.mp4")
                    print(f"Rendering {posture} {condition} sigma={sigma:g} episode {episode} "
                          f"(replaying {episode} episodes first to reach its noise stream)...",
                          flush=True)
                    best, end, fps, trace, frame_dir = render_episode(
                        env, condition, sigma, episode, args.episode_steps, args.seed, seq_len,
                        out, overlay=not args.no_overlay, fps=args.fps,
                        camera=args.camera, size=args.render_size,
                        transparent=args.transparent)
                    verdict = 'ROLLED' if best >= ROLL_THRESHOLD else 'no roll'
                    video = out[:-4] + ALPHA_CODEC[2] if args.transparent else out
                    print(f"  -> {video}  rho_max={best:.4f} rho_end={end:.4f} "
                          f"({verdict}, {fps} fps)", flush=True)
                    if args.keyframes:
                        paths, strip, cut, hit = write_keyframes(
                            out, trace, args.keyframes, out[:-4], frame_dir=frame_dir)
                        where = ("roll at step" if hit else "no roll; highest rho at step")
                        print(f"     {args.keyframes} keyframes over steps 0..{cut} "
                              f"({where} {cut}):", flush=True)
                        for path, step, rho in paths:
                            print(f"       step {step:3d}  rho={rho:.3f}  {path}", flush=True)
                        if strip:
                            print(f"       strip: {strip}", flush=True)
                    if frame_dir and not args.keep_frames:
                        shutil.rmtree(frame_dir, ignore_errors=True)
            finally:
                env.close()
                del env
                gc.collect()
        return

    # (condition, sigma). sigma is None where it has no meaning, which keeps the reference rows in
    # the same table instead of in a footnote.
    cells = []
    if not args.no_zero:
        cells.append(('zero', None))
    if not args.no_uniform:
        cells.append(('uniform', None))
    for colour in colours:
        for sigma in sigmas:
            cells.append((colour, sigma))

    rows = []
    model_name = 'MuscleModel' if args.use_muscle else 'SpringDamperModel'
    print(f"{args.episodes} episodes x {args.episode_steps} steps per cell, "
          f"morph_age={args.morph_age} physio_age={args.physio_age}, {model_name}, "
          f"success = rho_max >= {ROLL_THRESHOLD}\n")
    print(HEADER)
    print('-' * len(HEADER))
    for posture in postures:
        # Outer loop, so exactly one 3.6 GB env is alive at any moment.
        env = build_env(posture, args.morph_age, args.physio_age, use_muscle=args.use_muscle)
        try:
            for condition, sigma in cells:
                label = condition if sigma is None else f"{condition}_s{sigma:g}"
                rho_max, rho_end = run_condition(env, condition, sigma, args.episodes,
                                                 args.episode_steps, args.seed, seq_len)
                row = summarise(label, posture, condition, sigma, rho_max, rho_end)
                rows.append(row)
                print_row(row)
        finally:
            env.close()
            del env
            gc.collect()
        print()

    # Pooling the sigma sweep is what turns three loose 9.5 % bounds into one 3.3 % statement per
    # colour and posture.
    print("pooled over sigma:")
    print(HEADER)
    print('-' * len(HEADER))
    pooled = []
    conditions = list(dict.fromkeys(condition for condition, _ in cells))
    for posture in postures:
        for condition in conditions:
            cell = [r for r in rows
                    if r['posture'] == posture and r['condition'] == condition]
            merged = np.concatenate([np.array(r['rho_max']) for r in cell])
            merged_end = np.concatenate([np.array(r['rho_end']) for r in cell])
            row = summarise(f"{condition}_pooled", posture, condition, None, merged, merged_end)
            pooled.append(row)
            print_row(row)
    print()

    payload = {
        'protocol': {
            'episodes_per_cell': args.episodes,
            'episode_steps': args.episode_steps,
            'roll_threshold': ROLL_THRESHOLD,
            'side_lying_threshold': SIDE_LYING_THRESHOLD,
            'morph_age': args.morph_age,
            'physio_age': args.physio_age,
            'use_muscle': args.use_muscle,
            'seed': args.seed,
            'seq_len': seq_len,
            'isr': False,
            'goal': ROLL_THRESHOLD,
        },
        'cells': rows,
        'pooled': pooled,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {args.json}")
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, 'w') as fh:
            fh.write("posture,label,sigma,episodes,rolled,roll_rate,roll_rate_upper95,held,"
                     "side_lying_rate,rho_mean,rho_std,rho_best,rho_end_mean\n")
            for row in rows + pooled:
                sigma = '' if row['sigma'] is None else f"{row['sigma']:g}"
                bound = '' if row['roll_rate_upper95'] is None else f"{row['roll_rate_upper95']:.4f}"
                fh.write(f"{row['posture']},{row['label']},{sigma},{row['episodes']},"
                         f"{row['rolled']},{row['roll_rate']:.4f},{bound},{row['held']},"
                         f"{row['side_lying_rate']:.4f},{row['rho_mean']:.4f},"
                         f"{row['rho_std']:.4f},{row['rho_best']:.4f},"
                         f"{row['rho_end_mean']:.4f}\n")
        print(f"Wrote {args.csv}")

    total_rolls = sum(r['rolled'] for r in rows)
    total_eps = sum(r['episodes'] for r in rows)
    print(f"\n{total_rolls} rolls in {total_eps} episodes across {len(rows)} conditions.")
    # 26.08.2026 This used to print "no noise policy rolled", which was written before the run and
    # is false: red noise at sigma=1 tips MIMo over in a few per cent of episodes. Reporting that
    # is the honest version of the same conclusion -- the baseline is a chance rate, and it sits an
    # order of magnitude below the 75 % a run has to clear to count as successful.
    hits = [r for r in rows if r['rolled']]
    if not hits:
        print(f"No noise policy rolled MIMo over at any amplitude or spectral exponent "
              f"(95 % upper bound on the pooled roll rate: {upper_bound(0, total_eps):.2%}).")
    else:
        print("Rolls occurred only in: "
              + ", ".join(f"{r['posture']}/{r['label']} {r['rolled']}/{r['episodes']}"
                          f" (held: {r['held']})" for r in hits))
        print("Every other condition is at zero. Read these as the chance rate of an undirected "
              "flail tipping MIMo over, not as learning: nothing here is trained, and the thesis "
              f"calls a run successful above {0.75:.0%} rolls.")


if __name__ == '__main__':
    main()
