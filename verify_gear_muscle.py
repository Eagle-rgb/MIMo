"""Throwaway script: shows how the muscle model treats the XML 'gear', and whether --physio_age
reaches the applied torque.

    MUJOCO_GL=osmesa python verify_gear_muscle.py 1
    MUJOCO_GL=osmesa python verify_gear_muscle.py 9

Run both and compare. One env per process (~3.6 GB). Untracked, delete when done.
"""
import sys

import numpy as np
import mujoco
import gymnasium as gym

import mimoEnv                                    # registers MIMoRollOver-v0
from mimoActuation.muscle import MuscleModel

AGE = int(sys.argv[1]) if len(sys.argv) > 1 else 9
ACT = "act:hip_bend"

# --- 1. What does the XML say? Fresh compile, nothing has touched it yet. -------------
scene = f"mimoEnv/assets/roll_over/prone/scene_act_{AGE}_body_9.xml"
m = mujoco.MjModel.from_xml_path(scene)
i_fresh = [k for k in range(m.nu) if m.actuator(k).name == ACT][0]
gear_xml = m.actuator_gear[i_fresh, 0]
print(f"\nage {AGE}")
print(f"  1) gear in the XML (fresh compile)     : {gear_xml:.4f}")

# --- 2. Build the env with the muscle model and take a single step -------------------
env = gym.make("MIMoRollOver-v0", actuation_model=MuscleModel,
               age_morph=9, age_physio=AGE, isr=False, touch_params=None).unwrapped
env.reset(seed=0)

names = [env.model.actuator(a).name for a in env.mimo_actuators]
i = names.index(ACT)                              # index within mimo_actuators
adr = env.mimo_actuators[i]                       # index within model.actuator_*

print(f"  2) gear after env.reset()              : {env.model.actuator_gear[adr, 0]:.4f}")

action = np.zeros(env.action_space.shape)
action[i] = 1.0                                   # exactly one muscle, fully driven
env.step(action)

print(f"  3) gear after one env.step()           : {env.model.actuator_gear[adr, 0]:.4f}")
print(f"     data.ctrl for this actuator         : {env.data.ctrl[adr]:.4f}")

# --- 3. What actually reaches the dynamics? -----------------------------------------
# qfrc_actuator is the generalised force MuJoCo really applies:
#   actuator_force = gear * ctrl  (clamped to forcerange when forcelimited)
#   qfrc_actuator  = moment^T * actuator_force
dof = env.model.jnt_dofadr[env.model.actuator_trnid[adr, 0]]
print(f"  4) qfrc_actuator (the real torque)     : {env.data.qfrc_actuator[dof]:.6f}   <-- compare")

# Cross-check from the model's own quantities, without any MuJoCo internals:
am = env.actuation_model
print(f"     recomputed moment*force*fmax        : "
      f"{-am.moment_1[i] * am.force_muscles_1[i] * am.fmax[i]:.6f}")
print(f"     with fmax={am.fmax[i]:.3f}  activity={am.activity[i]:.3f}  "
      f"(fmax comes from the 'user' field, not from gear)")

# --- 4. Static: pose pinned, zero velocity, full activation --------------------------
# The step above mixes in the force-velocity curve: a stronger muscle accelerates the joint
# faster, fv drops, and the force comes back down. To see the FMAX effect on its own, fl and
# fv have to be identical at both ages, which pinning the pose and zeroing qvel guarantees.
env.data.qpos[am.mimo_actuated_qpos] = env.model.qpos_spring[am.mimo_actuated_qpos]
env.data.qvel[:] = 0.0
mujoco.mj_forward(env.model, env.data)
n = am.n_actuators
am.target_activity = np.concatenate([np.ones(n), np.zeros(n)])
for _ in range(2000):                             # let the activation lag converge
    am._update_activity()
am._update_virtual_lengths()
am._update_virtual_velocities()
am._update_torque()
am._apply_torque()
env.data.ctrl[env.mimo_actuators] = 1.0
mujoco.mj_forward(env.model, env.data)
print("  5) STATIC, full activation:")
print(f"     fl*fv+fp = {am.force_muscles_1[i]:.4f}   (identical at both ages)")
print(f"     qfrc_actuator                       : {env.data.qfrc_actuator[dof]:.4f}   <-- compare")
print(f"     spring-damper gear*|forcerange|     : "
      f"{-gear_xml * abs(m.actuator_forcerange[i_fresh, 0]):.4f}")
env.close()
