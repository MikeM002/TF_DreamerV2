import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np

from . import obj_detector

class GymWrapper:

  def __init__(self, env, obs_key='image', act_key='action'):
    self._env = env
    self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_is_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    if self._obs_is_dict:
      spaces = self._env.observation_space.spaces.copy()
    else:
      spaces = {self._obs_key: self._env.observation_space}
    return {
        **spaces,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    if self._act_is_dict:
      return self._env.action_space.spaces.copy()
    else:
      return {self._act_key: self._env.action_space}

  def step(self, action):
    if not self._act_is_dict:
      action = action[self._act_key]
    obs, reward, done, info = self._env.step(action)
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = float(reward)
    obs['is_first'] = False
    obs['is_last'] = done
    obs['is_terminal'] = info.get('is_terminal', done)
    return obs

  def reset(self):
    obs = self._env.reset()
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = 0.0
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    return obs


class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    os.environ['MUJOCO_GL'] = 'egl'
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if domain == 'manip':
      from dm_control import manipulation
      self._env = manipulation.load(task + '_vision')
    elif domain == 'locom':
      from dm_control.locomotion.examples import basic_rodent_2020
      self._env = getattr(basic_rodent_2020, task)()
    else:
      from dm_control import suite
      self._env = suite.load(domain, task)
    self._action_repeat = action_repeat
    self._size = size
    if camera in (-1, None):
      camera = dict(
          quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
          quadruped_fetch=2, locom_rodent_maze_forage=1,
          locom_rodent_two_touch=1,
      ).get(name, 0)
    self._camera = camera
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      if value.shape == (0,):
        self._ignored_keys.append(key)

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    for key, value in self._env.observation_spec().items():
      if key in self._ignored_keys:
        continue
      if value.dtype == np.float64:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
      elif value.dtype == np.uint8:
        spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
      else:
        raise NotImplementedError(value.dtype)
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return {'action': action}

  def step(self, action):
    assert np.isfinite(action['action']).all(), action['action']
    reward = 0.0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action['action'])
      reward += time_step.reward or 0.0
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    obs = {
        'reward': reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    self._env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._size = size
    self._grayscale = grayscale

  @property
  def obs_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    return {
        'image': gym.spaces.Box(0, 255, shape, np.uint8),
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def step(self, action):
    image, reward, done, info = self._env.step(action['action'])
    if self._grayscale:
      image = image[..., None]
    return {
        'image': image,
        'ram': self._env.env._get_ram(),
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': done,
    }

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    return {
        'image': image,
        'ram': self._env.env._get_ram(),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }

  def close(self):
    return self._env.close()


class Crafter:

  def __init__(self, outdir=None, reward=True, seed=None):
    import crafter
    self._env = crafter.Env(reward=reward, seed=seed)
    self._env = crafter.Recorder(
        self._env, outdir,
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    self._achievements = crafter.constants.achievements.copy()

  @property
  def obs_space(self):
    spaces = {
        'image': self._env.observation_space,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'log_reward': gym.spaces.Box(-np.inf, np.inf, (), np.float32),
    }
    spaces.update({
        f'log_achievement_{k}': gym.spaces.Box(0, 2 ** 31 - 1, (), np.int32)
        for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def step(self, action):
    image, reward, done, info = self._env.step(action['action'])
    obs = {
        'image': image,
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': info['discount'] == 0,
        'log_reward': info['reward'],
    }
    obs.update({
        f'log_achievement_{k}': v
        for k, v in info['achievements'].items()})
    return obs

  def reset(self):
    obs = {
        'image': self._env.reset(),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'log_reward': 0.0,
    }
    obs.update({
        f'log_achievement_{k}': 0
        for k in self._achievements})
    return obs


class Dummy:

  def __init__(self):
    pass

  @property
  def obs_space(self):
    return {
        'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    return {'action': gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

  def step(self, action):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': False,
        'is_last': False,
        'is_terminal': False,
    }

  def reset(self):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert hasattr(env.act_space[key], 'n')
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    shape = (self._env.act_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    if self._key not in action:
        pass
    else:
        action_vector = action[self._key]
        index = np.argmax(action_vector).astype(int)
        reference = np.zeros_like(action_vector)
        reference[index] = 1
        if not np.allclose(reference, action_vector):
            print(f"ERROR - OneHotAction - Invalid one-hot format: vector={action_vector}, index={index}, reference={reference}")
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
        raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.act_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class ResizeImage:

  def __init__(self, env, size=(64, 64)):
    self._env = env
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    if self._keys:
      from PIL import Image
      self._Image = Image

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def reset(self):
    obs = self._env.reset()
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image


class RenderImage:

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._shape = self._env.render().shape

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    obs[self._key] = self._env.render('rgb_array')
    return obs

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._env.render('rgb_array')
    return obs


class Async:

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _CLOSE = 4
  _EXCEPTION = 5

  def __init__(self, constructor, strategy='thread'):
    self._pickled_ctor = cloudpickle.dumps(constructor)
    if strategy == 'process':
      import multiprocessing as mp
      context = mp.get_context('spawn')
    elif strategy == 'thread':
      import multiprocessing.dummy as context
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = context.Pipe()
    self._process = context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    self._receive()  # Ready.
    self._obs_space = None
    self._act_space = None

  def access(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      pass  # The connection was already closed.
    self._process.join(5)

  @property
  def obs_space(self):
    if not self._obs_space:
      self._obs_space = self.access('obs_space')()
    return self._obs_space

  @property
  def act_space(self):
    if not self._act_space:
      self._act_space = self.access('act_space')()
    return self._act_space

  def step(self, action, blocking=False):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=False):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    try:
      ctor = cloudpickle.loads(self._pickled_ctor)
      env = ctor()
      conn.send((self._RESULT, None))  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass  # The connection was already closed.


class ObjectDetectionWrapper:
    """Environment wrapper that adds object detection to observations."""

    def __init__(self, env, template_path=None, detection_threshold=0.7, process_size=None):
        self._env = env
        self.detection_size = (128, 128)  # Default size for detection
        self.process_size = process_size if process_size else (64, 64)
        self.detection_threshold = detection_threshold
        
        print("="*50)
        print(f"OBJECT DETECTION DEBUG INFO:")
        print(f"  Detection threshold: {detection_threshold}")
        print(f"  Detection size: {self.detection_size}")
        print(f"  Process size: {self.process_size}")
        
        # New: Check if template_path is a directory and load all templates
        import os
        import glob
        
        if template_path and os.path.isdir(template_path):
            template_files = glob.glob(os.path.join(template_path, "*.png"))
            print(f"  Template directory: {template_path}")
            print(f"  Found {len(template_files)} template files")
            if len(template_files) < 12:
                print(f"  WARNING: Expected 12 templates but found only {len(template_files)}!")
            
            # Load each template into a separate detector
            self.template_files = template_files
            self.template_path = template_path  # Keep for reference
        else:
            # Support for single template file (backward compatibility)
            template_dir = os.path.dirname(template_path) if template_path else None
            self.template_path = template_path
            self.template_files = [template_path] if template_path else []
            print(f"  Template path: {template_path}")
            print(f"  Template directory: {template_dir}")
            
        # Create detector
        self.detector = obj_detector.create_detector(
            template_path=template_path,
            threshold=detection_threshold,
            detection_size=self.detection_size,
            process_size=self.process_size
        )
            
        # Pre-determine the output channels count (do this only once)
        self._output_channels = self._determine_output_channels()
        
        # Handle both gym-style and DreamerV2-style environments
        if hasattr(self._env, 'act_space'):
            self._act_space = self._env.act_space
        
        if hasattr(self._env, 'obs_space'):
            self._obs_space = self._update_obs_space()
        
        # For gym-style environments
        if hasattr(self._env, 'action_space'):
            self.action_space = self._env.action_space
        
        if hasattr(self._env, 'observation_space'):
            self.observation_space = self._update_observation_space()
        
        # Keep other attributes
        if hasattr(self._env, 'reward_range'):
            self.reward_range = self._env.reward_range
        if hasattr(self._env, 'metadata'):
            self.metadata = self._env.metadata
    
    
    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped environment."""
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self._env, name)
        
    @property
    def act_space(self):
        """Property to match DreamerV2's environment interface."""
        return self._act_space
        
    @property
    def obs_space(self):
        """Property to match DreamerV2's environment interface."""
        return self._obs_space

            
    def _determine_output_channels(self):
        """Determine the number of output channels without full processing."""
        # Create a minimal test image (1x1 pixel is enough to determine channels)
        test_image = np.zeros((1, 1, 1), dtype=np.uint8)
        test_obs = {'image': test_image}
        
        # Set a flag to suppress prints during this test processing
        self._suppress_prints = True
        processed = self._process_obs(test_obs)
        self._suppress_prints = False
        
        # Return the number of channels in the processed output
        return processed['image'].shape[-1]

    def _update_obs_space(self):
        """Update DreamerV2-style observation space using pre-determined channel count."""
        spaces = dict(self._env.obs_space)
        
        print(f"\n==== UPDATING OBS SPACE ====")
        if 'image' in spaces:
            input_shape = spaces['image'].shape
            print(f"Input observation shape: {input_shape}")
            
            if len(input_shape) > 1:  # Make sure it's an image
                # Use the pre-computed channel count determined during initialization
                shape = self.process_size + (self._output_channels,)
                print(f"Using pre-determined channel count: {self._output_channels}")
                
                spaces['image'] = gym.spaces.Box(0, 255, shape, dtype=np.uint8)
                print(f"Updated observation space shape: {spaces['image'].shape}")
        
        return spaces

    def _update_observation_space(self):
        """Update gym-style observation space to match actual output format."""
        import gym
        from gym.spaces import Box
        
        obs_space = self._env.observation_space
        
        # Get input space information
        if isinstance(obs_space, gym.spaces.Dict) and 'image' in obs_space.spaces:
            input_space = obs_space.spaces['image']
            input_shape = input_space.shape
            print(f"Input observation space: Dict with image shape {input_shape}")
        elif not isinstance(obs_space, gym.spaces.Dict):
            input_shape = obs_space.shape
            print(f"Input observation space: Box with shape {input_shape}")
        else:
            print(f"Input observation space: {obs_space} (no image found)")
            return obs_space
            
        # Use the same dynamically determined channel count
        print(f"Using pre-determined output channel count: {self._output_channels}")
        output_shape = self.process_size + (self._output_channels,)
        
        # Create appropriate space type
        if isinstance(obs_space, gym.spaces.Dict):
            spaces = {k: v for k, v in obs_space.spaces.items()}
            spaces['image'] = Box(0, 255, output_shape, dtype=np.uint8)
            result = gym.spaces.Dict(spaces)
            print(f"Updated 'image' in Dict to shape {output_shape}")
        else:
            # Direct Box space
            result = Box(0, 255, output_shape, dtype=np.uint8)
            print(f"Updated Box space to shape {output_shape}")
        
        return result

    def _resize(self, image, size):
        """Resize an image to the given size."""
        from PIL import Image
        
        # Handle None images by returning a blank image
        if image is None:
            return np.zeros((*size, 1), dtype=np.uint8)
        
        # Handle single-channel images
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)  # Remove the channel dimension for PIL compatibility
        
        # Convert to PIL and resize
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize(size, Image.NEAREST)
        resized = np.array(image_pil)
        
        # Restore the channel dimension if it was removed
        if len(image.shape) == 3 and image.shape[2] == 1 and len(resized.shape) == 2:
            resized = resized[..., np.newaxis]
        
        return resized

    def _process_obs(self, obs):
        """Process observation to add object detection mask."""
        import numpy as np
        
        if isinstance(obs, dict) and 'image' in obs:
            # Get the original image
            original_image = obs['image']
            
            # Process the image with object detection
            import time
            start_time = time.time()
            
            # Try to use detector's multi-template capabilities
            if hasattr(self.detector, 'try_templates'):
                # Use the new multi-template method
                result, confidence, mask, combined_output = self.detector.try_templates(original_image)
            else:
                # Fallback to standard processing
                result, confidence, mask, combined_output = self.detector.process_and_resize(original_image)
            
            detection_time = time.time() - start_time
            
            # No need to crop the mask here since we already do it in the detector
            
            # Resize both the original image and the mask
            small_original = self._resize(original_image, self.process_size)
            small_mask = self._resize(mask, self.process_size) if mask is not None else np.zeros((*self.process_size, 1), dtype=np.uint8)
            
            # Ensure both arrays have 3 dimensions before concatenation
            if len(small_original.shape) == 2:
                small_original = small_original[..., None]  # Add channel dimension
                
            if len(small_mask.shape) == 2:
                small_mask = small_mask[..., None]  # Add channel dimension
                
            # Concatenate the resized original image and mask (both now have 3 dimensions)
            combined = np.concatenate([small_original, small_mask], axis=-1)
                
            obs['image'] = combined
            
        return obs

    def reset(self):
        obs = self._env.reset()
        return self._process_obs(obs)

    def step(self, action):
        # Handle both dictionary actions and direct actions
        if isinstance(action, dict) and hasattr(self._env, 'act_space'):
            obs = self._env.step(action)
            # DreamerV2-style environments return just the observation dictionary
            return self._process_obs(obs)
        else:
            # Gym-style environments return observation, reward, done, info
            obs, reward, done, info = self._env.step(action)
            return self._process_obs(obs), reward, done, info


def make(name, **kwargs):
  # Extract object detection parameters first
  use_obj_detection = kwargs.pop('use_obj_detection', False)
  obj_detection_template = kwargs.pop('obj_detection_template', None)
  obj_detection_threshold = kwargs.pop('obj_detection_threshold', 0.7)
  
  # Handle render_size and process_size parameters
  render_size = kwargs.pop('render_size', (64, 64))
  process_size = kwargs.pop('process_size', (64, 64))
  
  # Create the environment based on the suite
  suite, task = name.split('_', 1)
  if suite == 'dmc':
    env = DMC(task, **kwargs)
  elif suite == 'atari':
    env = Atari(task, action_repeat=kwargs.get('action_repeat', 4),
                size=render_size,
                grayscale=kwargs.get('atari_grayscale', True))
  elif suite == 'crafter':
    outdir = kwargs.get('outdir', None)
    reward = kwargs.get('reward', True)
    seed = kwargs.get('seed', None)
    env = Crafter(outdir, reward, seed)
  else:
    raise NotImplementedError(f"Unknown environment suite: {suite}")
  
  # Apply object detection wrapper if needed
  if use_obj_detection or ('pacman' in name.lower()):
    env = ObjectDetectionWrapper(
      env, 
      template_path=obj_detection_template,
      detection_threshold=obj_detection_threshold,
      process_size=process_size
    )
  
  return env


def inspect_image_channels(image, label=""):
    """Utility function to inspect image channels and their characteristics."""
    print(f"\n==== IMAGE CHANNEL INSPECTION: {label} ====")
    print(f"Shape: {image.shape}")
    
    if len(image.shape) < 3:
        print("Not a multi-channel image")
        return
    
    num_channels = image.shape[-1]
    print(f"Number of channels: {num_channels}")
    
    # For each channel, analyze its characteristics
    for i, channel in enumerate(image.transpose(2, 0, 1)):
        min_val = np.min(channel)
        max_val = np.max(channel)
        unique_vals = np.unique(channel)
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        
        print(f"Channel {i}:")
        print(f"  Range: {min_val:.2f} to {max_val:.2f}")
        print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        print(f"  Unique values: {len(unique_vals)}")
        
        # Try to identify the type of channel
        if len(unique_vals) <= 2:
            print("  Likely a binary mask")
        elif std_val < 0.1 * (max_val - min_val):
            print("  Low variation - might be a constant or near-constant channel")
        elif mean_val < 0.2 * max_val:
            print("  Mostly dark - could be a specific feature channel")
        else:
            print("  Normal image data channel")
