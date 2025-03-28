import tensorflow as tf
from tensorflow.keras import layers

class ConvEncoder(tf.keras.Model):
  def __init__(self, depth, kernels, keys=['image'], **kw):
    super().__init__()
    self._depth = depth
    self._kernels = kernels
    self._keys = keys
    self._kw = kw
    
  def __call__(self, obs):
    if not isinstance(obs, dict):
      obs = {'image': obs}
    outputs = []
    for key in self._keys:
      if key not in obs:
        continue
      x = obs[key]
      
      # Handle input with extra mask channel (64x64x2 instead of 64x64x3)
      input_shape = x.shape[-3:]  # Get (H, W, C)
      channels = input_shape[-1]
      
      for i, kernel in enumerate(self._kernels):
        depth = 2 ** i * self._depth
        x = self.get('h{}_conv{}'.format(key, i), layers.Conv2D, depth, kernel, 2,
            padding='same', activation=self._kw.get('activation', 'relu'))(x)
      
      x = tf.reshape(x, [tf.shape(x)[0], np.prod(x.shape[1:])])
      outputs.append(x)
    if outputs:
      return tf.concat(outputs, -1)
    else:
      return tf.zeros([tf.shape(obs[list(obs.keys())[0]])[0], 0])
