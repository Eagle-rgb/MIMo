"""Goal-response curves: rho_max as a function of the desired_goal fed to the policy.

Data from mimoEnv/eval_rollover.py --policy_goal_sweep, 20 episodes per point, deterministic,
ISR off. The environment's own goal is fixed at 0.95 throughout; only the policy's input varies.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fed = np.arange(0.25, 0.951, 0.05)

seed3 = [0.525, 0.660, 0.700, 0.689, 0.789, 0.782, 0.777, 0.826,
         0.693, 0.112, 0.095, 0.098, 0.085, 0.068, 0.096]
seed3_min = [0.438, 0.507, 0.512, 0.507, 0.724, 0.725, 0.484, 0.690,
             0.108, 0.068, 0.061, 0.059, 0.042, 0.046, 0.047]
seed3_max = [0.650, 0.981, 0.851, 0.791, 0.826, 0.813, 0.818, 0.922,
             0.935, 0.269, 0.129, 0.200, 0.115, 0.113, 0.197]

e3b = [0.285, 0.339, 0.372, 0.438, 0.507, 0.657, 0.787, 0.823,
       0.837, 0.847, 0.858, 0.925, 0.967, 0.981, 0.987]
e3b_min = [0.193, 0.284, 0.301, 0.390, 0.463, 0.545, 0.692, 0.732,
           0.774, 0.801, 0.823, 0.823, 0.912, 0.945, 0.962]
e3b_max = [0.341, 0.394, 0.424, 0.469, 0.580, 0.851, 0.853, 0.916,
           0.859, 0.917, 0.915, 0.998, 0.995, 0.997, 0.999]

fig, ax = plt.subplots(figsize=(8.2, 5.0))

ax.fill_between(fed, e3b_min, e3b_max, color='#2a7f62', alpha=0.15, linewidth=0)
ax.plot(fed, e3b, 'o-', color='#2a7f62', linewidth=2.2, markersize=5,
        label='E3b (Seed 1) — 100 % Roll, keine Klippe')

ax.fill_between(fed, seed3_min, seed3_max, color='#b5442e', alpha=0.15, linewidth=0)
ax.plot(fed, seed3, 'o-', color='#b5442e', linewidth=2.2, markersize=5,
        label='Seed 3 — Plateau bei $\\rho\\approx0{,}83$, Klippe bei 0,65→0,70')

# The identity: below it the policy undershoots what was asked, above it it overshoots.
ax.plot([0.25, 0.95], [0.25, 0.95], ':', color='#888888', linewidth=1.2,
        label='Identität (erreicht = verlangt)')

ax.annotate('Klippe: 0,693 → 0,112\nbei 0,05 Änderung des Eingangs',
            xy=(0.695, 0.125), xytext=(0.74, 0.40), fontsize=9, color='#b5442e',
            arrowprops=dict(arrowstyle='->', color='#b5442e', linewidth=1.2))

ax.annotate('ehrliche Anfrage: 0,95 → 0,096\nbelogen mit 0,50: → 0,792',
            xy=(0.95, 0.096), xytext=(0.615, 0.045), fontsize=9, color='#b5442e')

ax.set_xlabel('$desired\\_goal$, das der Policy eingespeist wird')
ax.set_ylabel('erreichtes $\\rho_{max}$ (aus der Simulation gemessen)')
ax.set_title('Ziel-Antwortkurve: dieselben Gewichte, nur der Zieleingang variiert\n'
             '20 Episoden je Punkt, deterministisch, ISR aus, Umgebungsziel fix 0,95',
             fontsize=10.5)
ax.set_xlim(0.22, 0.98)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25, linewidth=0.6)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig('/home/lphilipp/MyVault/Projects/MIMo-intrinsic/her-goal-response.png', dpi=160)
print("written")
