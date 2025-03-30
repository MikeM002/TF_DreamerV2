import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._act_spaces = [env.act_space for env in envs]
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0):
    print(f"DEBUG - Driver called - steps: {steps}, episodes: {episodes}")
    print(f"DEBUG - Driver - Policy object type: {type(policy).__name__}")
    print(f"DEBUG - Driver - Policy._fn type: {type(policy._fn).__name__ if hasattr(policy, '_fn') else 'no _fn attribute'}")
    step, episode = 0, 0
    while step < steps or episode < episodes:
        print(f"DEBUG - Driver loop - Current step: {step}, Current episode: {episode}")
        obs = {
            i: self._envs[i].reset()
            for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
        for i, ob in obs.items():
            self._obs[i] = ob() if callable(ob) else ob
            act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
            tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
            [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
            self._eps[i] = [tran]
        obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
        
        print(f"DEBUG - Driver - Before policy call, obs keys: {list(obs.keys())}")
        print(f"DEBUG - Driver - Before policy call, state type: {type(self._state).__name__ if self._state is not None else 'None'}")
        
        # Get policy actions
        policy_output, self._state = policy(obs, self._state, **self._kwargs)
        
        print(f"DEBUG - Driver - After policy call, output type: {type(policy_output).__name__}")
        if isinstance(policy_output, dict):
            print(f"DEBUG - Driver - Policy returned dict with keys: {list(policy_output.keys())}")
            if 'action' in policy_output:
                action = policy_output['action']
                print(f"DEBUG - Driver - Action type: {type(action).__name__}")
                print(f"DEBUG - Driver - Action shape: {action.shape if hasattr(action, 'shape') else 'no shape'}")
                print(f"DEBUG - Driver - Action values: {action}")
            else:
                print(f"DEBUG - Driver - Warning: No 'action' in policy output")
        else:
            print(f"DEBUG - Driver - Warning: Policy did not return a dict, got: {type(policy_output).__name__}")
        
        # Properly prepare action dictionaries for each environment
        if isinstance(policy_output, dict) and 'action' in policy_output:
            # Normal case - policy returned the expected structure
            actions = [
                {k: np.array(policy_output[k][i]) for k in policy_output}
                for i in range(len(self._envs))]
        else:
            # Something's wrong with the policy output - create safe default actions
            print(f"WARNING: Policy returned unexpected format. Creating default actions.")
            actions = [
                {k: np.zeros(v.shape) for k, v in act_space.items()} 
                for act_space in self._act_spaces
            ]
        
        # Verify actions have correct structure before passing to environments
        for i, (env, action) in enumerate(zip(self._envs, actions)):
            expected_keys = set(self._act_spaces[i].keys())
            actual_keys = set(action.keys()) 
            if expected_keys != actual_keys:
                print(f"WARNING: Action keys mismatch for env {i}. Expected: {expected_keys}, Got: {actual_keys}")
                # Fix the action dictionary if needed
                for k in expected_keys - actual_keys:
                    print(f"Adding missing key '{k}' to action")
                    action[k] = np.zeros(self._act_spaces[i][k].shape)
            
            print(f"DEBUG - Driver - Action for env {i}: keys={list(action.keys())}")
            print(f"DEBUG - Driver - Action for env {i}: {actions[-1] if i < len(actions) else 'not yet created'}")
        
        # Continue with environment stepping
        assert len(actions) == len(self._envs)
        obs = [e.step(a) for e, a in zip(self._envs, actions)]
        obs = [ob() if callable(ob) else ob for ob in obs]
        for i, (act, ob) in enumerate(zip(actions, obs)):
            tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
            print(f"DEBUG - Transition for env {i}: {tran.keys()}")
            [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
            self._eps[i].append(tran)
            step += 1
            if ob['is_last']:
                ep = self._eps[i]
                ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                [fn(ep, **self._kwargs) for fn in self._on_episodes]
                episode += 1
        self._obs = obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value
