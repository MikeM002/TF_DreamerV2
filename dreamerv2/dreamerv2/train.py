import collections #estructuras de datos especializadas (listas, diccionarios, tuplas, etc)
import functools #manejo de funciones
import logging #mensajes de estado, errores, advertencias, etc
import os
import pathlib
import re
import sys #Variables mantenidas por el interprete
import warnings
import time  # Import for timing operations


try:
  import rich.traceback #mejora la salida de los errores
  rich.traceback.install()
except ImportError:
  pass

try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    HAS_RICH = True
    print("Rich library is available","-"*10)
except ImportError:
    HAS_RICH = False
    print("Nooooooooooooooo - Rich library is not available","-"*10)

def print_debug(message, separator=False, metrics=None):
    """Utility function for consistent debug messages with optional metrics."""
    if separator:
        print("=" * 50)
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[DEBUG {timestamp}] {message}")
    if metrics and isinstance(metrics, dict):
        for key, value in metrics.items():
            try:
                val = float(value.numpy()) if hasattr(value, 'numpy') else float(value)
                print(f"  {key}: {val:.6f}")
            except:
                print(f"  {key}: {value}")
    if separator:
        print("=" * 50)

def track_training_progress(data, step_value, config, is_evaluation=False):
    """Helper function to log detailed training progress."""
    mode = "Evaluation" if is_evaluation else "Training"
    progress = (step_value / config.steps) * 100
    remaining = config.steps - step_value
    bar_width = 20
    filled_width = int(bar_width * step_value / config.steps)
    bar = '█' * filled_width + '░' * (bar_width - filled_width)
    print(f"\n{mode} Progress: {step_value}/{config.steps} ({progress:.1f}%)")
    print(f"[{bar}] {remaining} steps remaining")
    print(f"Current time: {time.strftime('%H:%M:%S', time.localtime())}")

#os.environ es un diccionario que contiene todas las variables de entorno del sistema
#TF_CPP_MIN_LOG_LEVEL es una variable de entorno que establece el nivel de registro de TensorFlow (3: solo errores críticos)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#logging.getLogger() devuelve el logger raíz
#setLevel() establece el nivel de registro del logger raíz, en este caso, ERROR, solo se mostrarán mensajes de error
logging.getLogger().setLevel('ERROR')
#Se ignoran los warnings que tengan el mensaje 'box bound precision lowered'
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

#sys.path es una lista de directorios en los que el intérprete buscará los módulos
#pathlib.Path(__file__).parent devuelve el directorio padre del archivo actual
#pathlib.Path(__file__).parent.parent devuelve el directorio padre del directorio padre del archivo actual
#En pocas palabras, se añaden al path los directorios dreamerv2/dreamerv2 y dreamerv2 para que los módulos puedan ser importados
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
#import ruamel.yaml as yaml
#ruamel.yaml es una librería que permite leer y escribir archivos YAML (para cargar la configuración)
from ruamel.yaml import YAML

#El módulo agent.py contiene la clase Agent que define un agente de DreamerV2
import agent
#El módulo common.py contiene funciones y clases comunes que se utilizan en DreamerV2
import common


def ensure_channel_consistency(config, train_env, eval_env):
    """Enhanced version to thoroughly debug channel inconsistency issues."""
    print_debug("CHANNEL CONSISTENCY INVESTIGATION", separator=True)
    
    # 1. Check config settings that affect channels
    print_debug("Configuration settings that affect channels:")
    print_debug(f"  use_obj_detection: {config.get('use_obj_detection', False)}")
    print_debug(f"  atari_grayscale: {config.get('atari_grayscale', False)}")
    
    # 2. Look at the environment types and wrappers
    print_debug("Environment information:")
    train_env_type = type(train_env).__name__
    print_debug(f"  Train env type: {train_env_type}")
    
    # 3. Get observations from both environments
    train_obs = train_env.reset()
    eval_obs = eval_env.reset()
    
    # 4. Analyze image characteristics
    if 'image' in train_obs:
        image = train_obs['image']
        print_debug(f"  Train image shape: {image.shape}")
        print_debug(f"  Train image dtype: {image.dtype}")
        print_debug(f"  Train image range: {image.min()} to {image.max()}")
        
        # 5. If we have multiple channels, inspect them
        if len(image.shape) == 3 and image.shape[-1] > 1:
            for i in range(min(image.shape[-1], 6)):  # Show up to 6 channels
                channel = image[..., i]
                print_debug(f"  Channel {i} range: {channel.min()} to {channel.max()}")
                
                # Check if this looks like a mask (binary or near-binary values)
                unique_vals = np.unique(channel)
                if len(unique_vals) <= 2:
                    print_debug(f"  Channel {i} appears to be binary (mask-like)")
                    
    # 6. Check if eval has same structure
    if 'image' in eval_obs:
        image = eval_obs['image']
        print_debug(f"  Eval image shape: {image.shape}")
    
    # 7. Look for wrappers that might modify channels
    if hasattr(train_env, '_env'):
        wrapper_chain = []
        current_env = train_env
        while hasattr(current_env, '_env'):
            wrapper_chain.append(type(current_env).__name__)
            current_env = current_env._env
        print_debug(f"  Environment wrapper chain: {wrapper_chain}")
    
    # 8. Identify expected channels based on configuration
    expected_channels = None
    if config.get('use_obj_detection', False):
        # Con detección de objetos, usamos el valor configurado (2 canales: grayscale + mask)
        expected_channels = config.channels_expected  # Usar valor de config (2 para grayscale + mask)
    else:
        if config.get('atari_grayscale', False):
            expected_channels = 1  # Grayscale sin detección de objetos
        else:
            expected_channels = 3  # RGB sin detección de objetos
    
    print_debug(f"  Expected channels based on config: {expected_channels}")
    actual_channels = train_obs['image'].shape[-1] if 'image' in train_obs else None
    print_debug(f"  Actual channels in train env: {actual_channels}")
    
    # Detect channels in train environment
    if 'image' in train_obs:
        detected_channels = train_obs['image'].shape[-1]
        print_debug(f"[DEBUG] Detected {detected_channels} channels in environment observations")
        return detected_channels
    else:
        print_debug("[WARNING] No image key found in observations")
        return config.channels_expected  # Default to existing config if no image key


def main():

  print("-"*100,"Version:",40)

  #configs = yaml.safe_load((
      #pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  
  #Se carga la configuración por defecto
  yaml = YAML()
  #configs = yaml.load((pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  configs = yaml.load((pathlib.Path(__file__).parent / 'lite_configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  #Se configura el directorio de logs
  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  #Se configura TensorFlow
  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  #Se configuran los buffers de replay para el entrenamiento y la evaluación
  #Se configura el contador de pasos y el logger para registrar métricas
  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  #Definición de funciones que determinan cuándo se debe entrenar, registrar métricas y grabar videos
  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until // config.action_repeat)

  #Función que crea el entorno de entrenamiento 
  def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      # Create base Atari environment
      env = common.Atari(
          task, config.action_repeat, config.render_size,
          config.atari_grayscale)
      print(f"Creando entorno de Atari: {task}")
      
      # Apply object detection if enabled
      if config.get('use_obj_detection', False):
          print(f"Applying object detection wrapper to {task}")
          template_path = os.path.join(os.path.dirname(__file__), 'templates', 'pacman.png')
          
          print("=" * 50)
          print("DEBUG: TEMPLATE LOADING DETAILS")
          print(f"Absolute template path: {os.path.abspath(template_path)}")
          print(f"Current working directory: {os.getcwd()}")
          print(f"Current file location: {__file__}")
          print(f"Path exists: {os.path.exists(template_path)}")
          
          template_dir = os.path.dirname(template_path)
          print(f"Template directory exists: {os.path.exists(template_dir)}")
          
          if os.path.exists(os.path.dirname(__file__)):
              print(f"Files in {os.path.dirname(__file__)}:")
              for f in os.listdir(os.path.dirname(__file__)):
                  print(f"  {f}")
                  
              templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
              if os.path.exists(templates_dir):
                  print(f"Files in templates directory:")
                  for f in os.listdir(templates_dir):
                      print(f"  {f}")
              else:
                  print("Templates directory doesn't exist. Creating it...")
                  os.makedirs(templates_dir, exist_ok=True)
                  print(f"Templates directory created: {os.path.exists(templates_dir)}")
                  
                  try:
                      import cv2
                      import numpy as np
                      template = np.zeros((32, 32, 3), dtype=np.uint8)
                      cv2.circle(template, (16, 16), 14, (0, 255, 255), -1)
                      cv2.imwrite(template_path, template)
                      print(f"Created template at {template_path}")
                      print(f"Template now exists: {os.path.exists(template_path)}")
                      print(f"Template size: {os.path.getsize(template_path)} bytes")
                  except Exception as e:
                      print(f"Error creating template: {str(e)}")
          
          print("=" * 50)
          
          try:
              import cv2
              template = cv2.imread(template_path, 0)
              if template is None:
                  print(f"ERROR: OpenCV couldn't load the template at {template_path}")
                  template = cv2.imread(template_path, 1)
                  if template is None:
                      print("ERROR: OpenCV couldn't load template in color mode either")
                      print(f"File readable: {os.access(template_path, os.R_OK)}")
                      print(f"File writable: {os.access(template_path, os.W_OK)}")
                  else:
                      print(f"Successfully loaded template in color mode: {template.shape}")
              else:
                  print(f"Successfully loaded template in grayscale: {template.shape}")
          except Exception as e:
              print(f"Exception when loading template with OpenCV: {str(e)}")
          
          env = common.envs.ObjectDetectionWrapper(
              env, 
              detection_threshold=config.get('obj_detection_threshold', 0.7),
              process_size=config.get('process_size', (64, 64)),
              template_path=template_path
          )
          
      env = common.OneHotAction(env)
    elif suite == 'crafter':
      assert config.action_repeat == 1
      outdir = logdir / 'crafter' if mode == 'train' else None
      reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
      env = common.Crafter(outdir, reward)
      env = common.OneHotAction(env)
    else:
      raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env

  #Función que registra las métricas de un episodio
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  print_debug("Creating environments...", separator=True)
  #Se crean los entornos de entrenamiento y evaluación
  num_eval_envs = min(config.envs, config.eval_eps)
  if config.envs_parallel == 'none':
    train_envs = [make_env('train') for _ in range(config.envs)]
    print(f"Entornos de entrenamiento creados: {config.task}")
    eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
    print(f"Entornos de evaluación creados: {config.task}")
  else:
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    train_envs = [make_async_env('train') for _ in range(config.envs)]
    eval_envs = [make_async_env('eval') for _ in range(eval_envs)]

  # Ensure channel consistency
  train_channels = ensure_channel_consistency(config, train_envs[0], eval_envs[0])
  if train_channels != config.channels_expected:
      print_debug(f"[DEBUG] Updating channels_expected from {config.channels_expected} to {train_channels}")
      # Update both the general channels_expected and specific encoder/decoder channels
      config = config.update({
          'channels_expected': train_channels,
          'encoder.cnn_channels': train_channels,
          'decoder.cnn_channels': train_channels
      })
      print_debug(f"[DEBUG] Updated config now has channels_expected = {config.channels_expected}")

  #Se configura el driver de entrenamiento y evaluación, define los callbacks para registrar métricas y guardar datos
  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  #Se pre-entrena el agente con un agente aleatorio para llenar el buffer de replay
  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print_debug(f"Pre-filling replay buffer with {prefill} random steps...", separator=True)
    random_agent = common.RandomAgent(act_space)
    if HAS_RICH:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            prefill_task = progress.add_task("Prefilling replay buffer", total=prefill)
            def update_progress(tran, worker):
                progress.update(prefill_task, completed=step.value)
            train_driver.on_step(update_progress)
            train_driver(random_agent, steps=prefill, episodes=1)
    else:
        train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()
    print_debug("Pre-filling complete")

  # Modify obs_space to ensure the correct number of channels
  modified_obs_space = dict(obs_space)

  # If 'image' exists in the observation space and object detection is enabled
  if 'image' in modified_obs_space and config.use_obj_detection:
      current_shape = modified_obs_space['image'].shape
      current_channels = current_shape[-1]

      print_debug(f"Current obs_space image shape: {current_shape} with {current_channels} channels")
      print_debug(f"Expected channels from config: {config.channels_expected}")

      # Adjust the observation space to match the expected number of channels
      if current_channels != config.channels_expected:
          if hasattr(modified_obs_space['image'], 'low') and hasattr(modified_obs_space['image'], 'high'):
              import gym
              new_shape = current_shape[:-1] + (config.channels_expected,)
              print_debug(f"Creating new observation space with shape: {new_shape}")
              modified_obs_space['image'] = gym.spaces.Box(
                  low=0, high=255, shape=new_shape, dtype=modified_obs_space['image'].dtype)
          else:
              from common import Space
              new_shape = current_shape[:-1] + (config.channels_expected,)
              print_debug(f"Creating new observation space with shape: {new_shape}")
              modified_obs_space['image'] = Space(np.zeros(new_shape, dtype=np.uint8))

  # Create the agent with the modified observation space
  print('Creating agent with corrected observation space.')
  agnt = agent.Agent(config, modified_obs_space, act_space, step)

  #Se crea el agente y se entrena, si se ha guardado un checkpoint previo, se carga
  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  train_agent = common.CarryOverState(agnt.train)

  # Add debug logs before training
  sample_batch = next(train_dataset)
  print("DEBUG - Preparing to train agent with data batch:")
  for key, value in sample_batch.items():
      if hasattr(value, 'shape'):
          print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
          if key == 'image':
              print(f"  image min/max values: {value.numpy().min()}/{value.numpy().max()}")

  train_agent(sample_batch)

  # Add debug logs for environment and model settings
  if config.get('debug_prints', False):
    print("=" * 50)
    print("DEBUG - Environment and Model Settings:")
    print(f"Process size: {config.get('process_size', 'Not specified')}")
    print(f"Render size: {config.render_size}")
    print(f"Encoder CNN depth: {config.encoder.get('cnn_depth')}")
    print(f"Encoder CNN kernels: {config.encoder.get('cnn_kernels')}")
    print(f"Decoder CNN depth: {config.decoder.get('cnn_depth')}")
    print(f"Decoder CNN kernels: {config.decoder.get('cnn_kernels')}")
    try:
      for i, layer in enumerate(agnt.wm.encoder._layers):
        if hasattr(layer, 'filters'):
          print(f"Encoder Layer {i}: Type={layer.__class__.__name__}, Filters={layer.filters}, Kernel={layer.kernel_size}")
      for i, layer in enumerate(agnt.wm.heads['decoder']._layers):
        if hasattr(layer, 'filters'):
          print(f"Decoder Layer {i}: Type={layer.__class__.__name__}, Filters={layer.filters}, Kernel={layer.kernel_size}")
    except:
      print("Could not access model layer details")
    print("=" * 50)

  train_agent(next(train_dataset))

  # After pretrain
  if (logdir / 'variables.pkl').exists():
    print('='*50)
    print('Loading variables from checkpoint')
    agnt.load(logdir / 'variables.pkl')
  else:
    print('='*50)
    print('Pretrain agent.')
    for i in range(config.pretrain):
      if i % max(1, config.pretrain // 5) == 0:
        print(f'Pretrain step {i}/{config.pretrain}')
      train_agent(next(train_dataset))
    print('Pretrain complete!')

  # Add marker for main training start
  print_debug("Starting main training loop...", separator=True)

  train_policy = common.CarryOverState(agnt.policy)
  eval_policy = common.CarryOverState(agnt.policy)

  loop_start_time = time.time()
  last_checkpoint_time = loop_start_time
  step_times = []

  def train_step_with_tracking(tran, worker):
          if should_train(step):
              if step.value % max(1, config.log_every // 10) == 0:
                  track_training_progress(tran, step.value, config)
                  if step_times:
                      avg_step_time = sum(step_times[-10:]) / min(10, len(step_times))
                      steps_remaining = config.steps - step.value
                      est_time_remaining = avg_step_time * steps_remaining
                      print(f"  Est. remaining time: {est_time_remaining/60:.1f} minutes")
              step_start = time.time()
              for _ in range(config.train_steps):
                  data_batch = next(train_dataset)
                  mets_tuple = train_agent(data_batch)
                  # Properly unpack the tuple returned by CarryOverState
                  if isinstance(mets_tuple, tuple) and len(mets_tuple) == 2:
                      mets, _ = mets_tuple  # Extract just the metrics, ignore the state
                  else:
                      mets = mets_tuple  # In case it's not a tuple (fallback)
                  # Now update the metrics
                  [metrics[key].append(value) for key, value in mets.items()]
              step_times.append(time.time() - step_start)
          if should_log(step):
              metric_values = {name: np.array(values, np.float64).mean() 
                              for name, values in metrics.items() if values}
              metrics.clear()
              print_debug(f"Logging metrics at step {step.value}", metrics=metric_values)
              logger.add(agnt.report(next(report_dataset)), prefix='train')
              logger.write(fps=True)

  while step < config.steps:
      block_start_time = time.time()
      block_target = min(step.value + config.eval_every, config.steps)

      # =========== EVALUATION PHASE ===========
      print(f"\n{'=' * 20} EVALUATION PHASE {'=' * 20}")
      print(f"Step {step.value}/{config.steps} - Starting evaluation with {config.eval_eps} episodes")
      
      eval_start = time.time()
      eval_driver(eval_policy, episodes=config.eval_eps)
      eval_duration = time.time() - eval_start
      print(f"Evaluation completed in {eval_duration:.2f}s")

      # =========== TRAINING PHASE ===========
      print(f"\n{'=' * 20} TRAINING PHASE {'=' * 20}")
      print(f"Step {step.value}/{config.steps} - Training until step {block_target}")
      
      train_start = time.time()
      train_driver(train_policy, steps=config.eval_every)
      train_duration = time.time() - train_start
      print(f"Training block completed in {train_duration:.2f}s")

      # =========== CHECKPOINT PHASE ===========
      current_time = time.time()
      checkpoint_interval = config.get('checkpoint_interval_minutes', 15) * 60
      if current_time - last_checkpoint_time > checkpoint_interval:
          print(f"\n{'=' * 20} SAVING CHECKPOINT {'=' * 20}")
          print(f"Saving checkpoint at step {step.value}")
          try:
              agnt.save(logdir / 'variables.pkl')
              last_checkpoint_time = current_time
              print("Checkpoint saved successfully")
          except Exception as e:
              print(f"Error saving checkpoint: {str(e)}")

      # =========== BLOCK SUMMARY ===========
      block_duration = time.time() - block_start_time
      total_duration = time.time() - loop_start_time
      
      print(f"\n{'=' * 20} BLOCK SUMMARY {'=' * 20}")
      print(f"Completed block: {step.value}/{config.steps} steps")
      print(f"Block duration: {block_duration/60:.2f} minutes")
      print(f"Total duration: {total_duration/60:.2f} minutes")
      print(f"Steps per second: {config.eval_every/block_duration:.2f}")
      print(f"Est. completion: {time.strftime('%H:%M:%S', time.localtime(time.time() + (config.steps - step.value) * block_duration / config.eval_every))}")

  # Final checkpoint
  try:
      print_debug("Saving final checkpoint")
      agnt.save(logdir / 'variables.pkl')
      print_debug("Final checkpoint saved")
  except Exception as e:
      print_debug(f"Error saving final checkpoint: {str(e)}")

  print_debug(f"Training complete. Total time: {(time.time() - loop_start_time) / 60:.2f} minutes", separator=True)

  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()
