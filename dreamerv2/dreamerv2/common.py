class Decoder:
    def __init__(self, shapes, cnn_depth, cnn_kernels, channels_expected, **kwargs):
        self.shapes = shapes
        self.cnn_depth = cnn_depth
        self.cnn_kernels = cnn_kernels
        self.output_channels = channels_expected  # Use channels_expected for output
        self.kwargs = kwargs

    def __call__(self, features):
        x = features
        for _ in range(self.cnn_depth):
            x = tf.keras.layers.Conv2D(
                filters=self.cnn_kernels,
                kernel_size=3,
                activation='relu'
            )(x)
        x = tf.keras.layers.Conv2D(
            filters=self.output_channels,  # Ensure output matches expected channels
            kernel_size=3,
            activation=None
        )(x)
        return x