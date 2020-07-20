import tensorflow as tf

from trainer.names import GRID_DIM


class ModelSimple:
    def __init__(self, feature_dim, target_dim_dict, filters, kernels):
        self.feature_dim = feature_dim
        self.target_dim_dict = target_dim_dict
        self.filters = filters
        self.kernels = kernels

    def build(self):
        feature, mask = self._hydra_input()
        z = self._encoder(feature)
        heads = self._heads(z)
        output = self._hydra_output(heads, mask)

        model = tf.keras.models.Model(inputs=(feature, mask), outputs=output)
        return model

    def _heads(self, z):
        heads = []
        for p_name, p_dim in self.target_dim_dict.items():
            head = self._decoder(z, p_name)
            head = tf.keras.layers.Conv2D(p_dim, 7, 1, 'same', activation='linear', name=p_name + '_head')(head)
            heads.append(head)
        return heads

    def _encoder(self, x):
        kernel = self.kernels[0]
        for i, filt in enumerate(self.filters):
            x = tf.keras.layers.Conv2D(filt, kernel, 2, 'same', activation='relu', name="encoder" + str(i))(x)

        return x

    def _decoder(self, x, mode):
        kernel = self.kernels[0]
        for i, filt in enumerate(self.filters[::-1]):
            x = tf.keras.layers.Conv2DTranspose(filt, kernel, 2, 'same', activation='relu',
                                                name=mode + "decoder" + str(i))(x)

        return x

    def _hydra_input(self):
        feature = tf.keras.layers.Input(shape=(GRID_DIM.y, GRID_DIM.x, self.feature_dim), name='feature')
        mask = tf.keras.layers.Input(shape=(GRID_DIM.y, GRID_DIM.x, 1), name='mask')

        return feature, mask

    def _hydra_output(self, heads, mask):
        o = tf.keras.layers.Concatenate(axis=-1)(heads)

        return tf.keras.layers.Multiply(name="prediction")([o, mask])


if __name__ == '__main__':
    from trainer.names import PHYSICAL_DIMS, PHYSICAL_DIMS_SCALAR

    model = ModelSimple(16, PHYSICAL_DIMS_SCALAR, [32, 64, 128], [7, 7, 7]).build()
    model.summary()
