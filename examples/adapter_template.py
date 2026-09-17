"""Copy this file to my_adapter.py and implement all methods using your environment.
The template intentionally does not pretend to be a working environment.
"""
from core.interfaces import EnvSpec, Frame, Transition

class Adapter:
    # Replace ALL names/dimensions. Names/order are checked on resume.
    spec = EnvSpec(observation_size=2,state_size=2,agents=1,
                   observation_names=('signal','context'),state_names=('signal','context'),
                   feature_names=('signal','context'),action_names=('choice_a','choice_b'))

    def reset(self, seed=None) -> Frame:
        # Return float32 observations [agents, obs_size], state [state_size].
        raise NotImplementedError('Connect your own environment reset here')

    def step(self, actions) -> Transition:
        # actions: integer indices in spec.action_names order.
        # Return terminal frame BEFORE any reset; do NOT auto-reset internally.
        # terminated=True: actual terminal; truncated=True: time limit.
        # actor_mask=False for actions chosen by an external controller (even
        # if coincidentally equal to the NN action). Otherwise it must be True.
        # submitted_actions records actual input, not collision-adjusted motion.
        # Rewards are supplied by your environment; this library adds no rewards.
        raise NotImplementedError('Connect your own environment step here')

    def features(self, frame, agent):
        # Ordered mapping matching feature_names, public information only.
        # Name relational features interaction.<name> to include in relational audit.
        # Do not insert labels, future actions or hidden simulator information.
        raise NotImplementedError('Provide named numeric features here')

    # OPTIONAL: implement BOTH for exact resume, otherwise omit BOTH.
    # def snapshot(self): return complete_environment_and_controller_state
    # def restore(self, snapshot): restore all histories, RNGs and simulator state


def make_env():
    return Adapter()
