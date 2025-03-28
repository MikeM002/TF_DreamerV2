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


def main():

  print("-"*100,"Version:",190)

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
    env = common.envs.make(
        config.task, 
        render_size=config.render_size,
        process_size=config.process_size, 
        use_obj_detection=config.use_obj_detection,
        obj_detection_threshold=config.get('obj_detection_threshold', 0.7),
    )
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      env = common.Atari(
          task, config.action_repeat, config.render_size,
          config.atari_grayscale)
      print(f"Creando entorno de Atari: {task}")  # Añade esta línea para verificar el entorno
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

  #Se crea el agente y se entrena, si se ha guardado un checkpoint previo, se carga
  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, obs_space, act_space, step)
  train_agent = common.CarryOverState(agnt.train)

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

  while step < config.steps:
      block_start_time = time.time()

      # Evaluation phase
      print_debug(f"Starting evaluation at step {step.value}/{config.steps}", separator=True)
      try:
          eval_start = time.time()
          eval_driver(eval_policy, episodes=config.eval_eps)
          eval_duration = time.time() - eval_start
          print_debug(f"Evaluation completed in {eval_duration:.2f}s")
      except Exception as e:
          print_debug(f"Error during evaluation: {str(e)}")

      # Training phase
      print_debug(f"Starting training block at step {step.value}/{config.steps}", separator=True)
      block_target = min(step.value + config.eval_every, config.steps)

      try:
          train_start = time.time()

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
                      mets = train_agent(data_batch)
                      [metrics[key].append(value) for key, value in mets.items()]
                  step_times.append(time.time() - step_start)
              if should_log(step):
                  metric_values = {name: np.array(values, np.float64).mean() for name, values in metrics.items() if values}
                  metrics.clear()
                  print_debug(f"Logging metrics at step {step.value}", metrics=metric_values)
                  logger.add(agnt.report(next(report_dataset)), prefix='train')
                  logger.write(fps=True)

          original_train_step = train_driver._on_step
          train_driver.on_step(train_step_with_tracking)
          train_driver(train_policy, steps=config.eval_every)
          train_driver._on_step = original_train_step

          train_duration = time.time() - train_start
          print_debug(f"Training block completed in {train_duration:.2f}s")
      except Exception as e:
          print_debug(f"Error during training: {str(e)}")

      # Save checkpoint every 15 minutes or at the specified interval
      current_time = time.time()
      checkpoint_interval = config.get('checkpoint_interval_minutes', 15) * 60
      if current_time - last_checkpoint_time > checkpoint_interval:
          print_debug(f"Saving checkpoint at step {step.value}")
          try:
              agnt.save(logdir / 'variables.pkl')
              last_checkpoint_time = current_time
              print_debug("Checkpoint saved successfully")
          except Exception as e:
              print_debug(f"Error saving checkpoint: {str(e)}")

      block_duration = time.time() - block_start_time
      total_duration = time.time() - loop_start_time
      print_debug(
          f"Completed block: {step.value}/{config.steps} steps",
          separator=True,
          metrics={
              "block_duration_minutes": block_duration / 60,
              "total_duration_minutes": total_duration / 60,
              "steps_per_second": config.eval_every / block_duration if block_duration > 0 else 0
          }
      )

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
