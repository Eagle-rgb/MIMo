""" Initialisation of a policy's action-mean head. """

import torch as th


def find_action_mean_layer(model):
    """ The final ``nn.Linear`` producing the action mean, for the algorithms that have one.

    PPO/A2C keep it as ``policy.action_net``; SAC keeps it as ``policy.actor.mu``, which is
    wrapped in an ``nn.Sequential`` when ``clip_mean`` is set. TD3/DDPG build ``actor.mu`` as a
    whole MLP ending in a ``Tanh``, and their deterministic policy makes an initial offset mean
    something different, so they are refused rather than guessed at.

    Args:
        model: A stable-baselines3 model.

    Returns:
        torch.nn.Linear: The layer whose bias is the action mean at initialisation.

    Raises:
        TypeError: If the algorithm has no single such layer.
    """
    head = getattr(model.policy, "action_net", None)          # PPO, A2C
    if head is None:
        actor = getattr(model.policy, "actor", None)
        head = getattr(actor, "mu", None) if actor is not None else None
    if isinstance(head, th.nn.Sequential):                     # SAC with clip_mean
        linears = [layer for layer in head if isinstance(layer, th.nn.Linear)]
        head = linears[-1] if len(linears) == 1 else None
    if not isinstance(head, th.nn.Linear):
        raise TypeError(
            f"{type(model).__name__} has no single action-mean layer to offset. "
            f"--action_bias_init supports PPO, A2C and SAC.")
    return head


def set_action_bias(model, value):
    """ Start every action dimension at ``value`` instead of at 0.

    08.09.2026 Why this exists. SB3 initialises ``action_net`` orthogonally with gain 0.01, so a
    fresh policy outputs a mean of ~0 in every dimension. What that *means* differs between the
    two actuation models, and the difference is not something anyone chose:

    * ':class:`~mimoActuation.actuation.SpringDamperModel`' -- the action is the control input,
      so 0 is zero torque: MIMo starts limp.
    * ':class:`~mimoActuation.muscle.MuscleModel`' under the symmetric action space -- activation
      is ``0.5 * (action + 1)``, so 0 is **half activation on all 92 muscles**: MIMo starts as a
      rigid, fully co-contracted statue.

    That is the measured operating point of every muscle run of 06.09.2026: mean activation
    0.55-0.59 across all 92 muscles, of which a third is spent against the muscle's own
    antagonist, and the metabolic penalty shaves only ~8 % off it in 1M steps. Note lowering
    ``log_std_init`` alone makes this *worse*, not better -- it concentrates the distribution on
    the co-contracted mean. The mean is the knob for it.

    Under the muscle model the useful values are negative: -1.0 is fully relaxed but sits on the
    edge of the action box (half of every sample clipped), so something like -0.6 (activation
    0.2) leaves room to explore in both directions.

    On SAC the mean is squashed, so the effective action is ``tanh(value)``; on PPO/A2C it is
    ``value`` directly.

    Args:
        model: A stable-baselines3 model, freshly constructed (not loaded -- a loaded model's
            weights are the point of loading it).
        value (float): The bias to write into every output unit.

    Returns:
        torch.nn.Linear: The layer that was modified.
    """
    head = find_action_mean_layer(model)
    with th.no_grad():
        head.bias.fill_(float(value))
    return head
