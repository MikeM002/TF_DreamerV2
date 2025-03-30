def build_image_decoder(config, shape, **kwargs):
    print(f"DEBUG - build_image_decoder - Input shape: {shape}")
    layers = []
    kernel_size = config.get('kernel_size', 3)
    strides = config.get('strides', 2)
    depth = config.get('cnn_channels', shape[-1])
    
    print(f"DEBUG - Using kernel size: {kernel_size}, strides: {strides}")
    print(f"DEBUG - Using cnn_channels from config: {depth}")
    
    for i in range(config.get('cnn_depth', 4)):
        layers.append(tf.keras.layers.Conv2DTranspose(
            depth, kernel_size, strides, padding='same', activation='relu'))
        print(f"DEBUG - Added Conv2DTranspose layer {i} with depth {depth}")
    
    print(f"DEBUG - Final decoder layer will output {depth} channels")
    layers.append(tf.keras.layers.Conv2DTranspose(
        depth, kernel_size, 1, padding='valid'))
    
    return tf.keras.Sequential(layers)