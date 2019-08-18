from tensorflow import keras

from trainer.names import *


def get_model(name, feature_dim):
    models = dict(
        hydra_v0=hydra_v0,
        hydra_v1=hydra_v1,
        hydra_v2=hydra_v2,
        hydra_v01=hydra_v01,
        hydra_v001=hydra_v001,
        hydra_unet_v0=hydra_unet_v0,
        hydra_scalar_v0=hydra_scalar_v0,
        hydra_scalar_v1=hydra_scalar_v1,
        hydra_scalar_v2=hydra_scalar_v2,
        simple_cnn=simple_cnn
    )

    if name in models.keys():
        return models[name](feature_dim)
    else:
        raise (Exception("{} not a valid model".format(name)))


def hydra_input(feature_dim):
    feature = keras.layers.Input(shape=(GRID_DIM.y, GRID_DIM.x, feature_dim), name='feature')
    mask = keras.layers.Input(shape=(GRID_DIM.y, GRID_DIM.x, 1), name='mask')

    return feature, mask


def hydra_output(heads, mask):
    # hydra assembler
    o = keras.layers.Concatenate(axis=-1)(heads)

    # mask output
    output = keras.layers.Multiply(name="prediction")([o, mask])

    return output


def hydra_v0(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(128, 7, 2, 'same', activation='relu', name="feature_embedding")(feature)

    def expand(mode, out_dim):
        return keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='linear', name=mode+'_head')(x)

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS.items():
        head = expand(p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def hydra_v1(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(128, 7, 2, 'same', activation='relu', name="feature_embedding_1")(feature)
    x = keras.layers.Conv2D(128, 5, 2, 'same', activation='relu', name="feature_embedding_2")(x)

    def expand(mode, out_dim):
        h = keras.layers.Conv2DTranspose(128, 5, 2, 'same', activation='relu', name=mode+'_1')(x)
        h = keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='linear', name=mode+'_head')(h)

        return h

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS.items():
        head = expand(p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def hydra_v2(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(128, 7, 2, 'same', activation='relu', name="feature_embedding_1")(feature)
    x = keras.layers.Conv2D(128, 5, 2, 'same', activation='relu', name="feature_embedding_2")(x)
    x = keras.layers.Conv2D(128, 3, 2, 'same', activation='relu', name="feature_embedding_3")(x)

    def expand(x_emb, mode, out_dim):
        h = keras.layers.Conv2DTranspose(128, 3, 2, 'same', activation='relu', name=mode+'_2')(x)
        h = keras.layers.Conv2DTranspose(128, 5, 2, 'same', activation='relu', name=mode+'_3')(h)
        h = keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='linear', name=mode+'_head')(h)

        return h

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS.items():
        head = expand(x, p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def hydra_v01(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(64, 7, 2, 'same', activation='relu', name="feature_embedding_1")(feature)
    x = keras.layers.Conv2D(64, 5, 2, 'same', activation='relu', name="feature_embedding_2")(x)

    def expand(mode, out_dim):
        h = keras.layers.Conv2DTranspose(64, 5, 2, 'same', activation='relu', name=mode + '_1')(x)
        h = keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='linear', name=mode + '_head')(h)

        return h

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS.items():
        head = expand(p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def hydra_v001(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(256, 7, 2, 'same', activation='relu', name="feature_embedding_1")(feature)
    x = keras.layers.Conv2D(256, 5, 2, 'same', activation='relu', name="feature_embedding_2")(x)

    def expand(mode, out_dim):
        h = keras.layers.Conv2DTranspose(256, 5, 2, 'same', activation='relu', name=mode + '_1')(x)
        h = keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='linear', name=mode + '_head')(h)

        return h

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS.items():
        head = expand(p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def hydra_scalar_v0(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(128, 7, 2, 'same', activation='relu', name="feature_embedding")(feature)

    def expand(mode, out_dim):
        return keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='linear', name=mode+'_head')(x)

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS_SCALAR.items():
        head = expand(p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def hydra_scalar_v1(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(128, 7, 2, 'same', activation='relu', name="feature_embedding_1")(feature)
    x = keras.layers.Conv2D(256, 5, 2, 'same', activation='relu', name="feature_embedding_2")(x)

    def expand(mode, out_dim):
        h = keras.layers.Conv2DTranspose(128, 5, 2, 'same', activation='relu', name=mode + '_1')(x)
        h = keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='linear', name=mode + '_head')(h)

        return h

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS_SCALAR.items():
        head = expand(p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def hydra_scalar_v2(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(128, 7, 2, 'same', activation='relu', name="feature_embedding_1")(feature)
    x = keras.layers.Conv2D(256, 5, 2, 'same', activation='relu', name="feature_embedding_2")(x)
    x = keras.layers.Conv2D(512, 3, 2, 'same', activation='relu', name="feature_embedding_3")(x)

    def expand(mode, out_dim):
        h = keras.layers.Conv2DTranspose(256, 3, 2, 'same', activation='relu', name=mode+'_1')(x)
        h = keras.layers.Conv2DTranspose(128, 5, 2, 'same', activation='relu', name=mode+'_2')(h)
        h = keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='linear', name=mode+'_head')(h)

        return h

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS_SCALAR.items():
        head = expand(p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def hydra_unet_v0(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x1 = keras.layers.Conv2D(128, 7, 2, 'same', activation='relu', name="d1")(feature)
    x = keras.layers.Conv2D(256, 5, 2, 'same', activation='relu', name="d2")(x1)

    def expand(mode, out_dim):
        h = keras.layers.Conv2DTranspose(256, 3, 1, 'same', activation='relu', name=mode+'_1')(x)
        h = keras.layers.Conv2DTranspose(128, 5, 2, 'same', activation='relu', name=mode+'_2')(h)
        h = keras.layers.Concatenate(name=mode+'_c1')([x1, h])
        h = keras.layers.Conv2DTranspose(out_dim, 7, 2, 'same', activation='relu', name=mode+'_3')(h)
        h = keras.layers.Concatenate(name=mode+'_c2')([feature, h])
        h = keras.layers.Conv2D(out_dim, 7, 1, 'same', activation='linear', name=mode+'_head')(h)

        return h

    # expansion
    heads = []
    for p_name, p_dim in PHYSICAL_DIMS.items():
        head = expand(p_name, p_dim)
        heads.append(head)

    # output
    output = hydra_output(heads, mask)

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


def simple_cnn(feature_dim):
    feature, mask = hydra_input(feature_dim)

    # contraction
    x = keras.layers.Conv2D(128, 7, 2, 'same', activation='relu', name="feature_embedding_1")(feature)
    x = keras.layers.Conv2D(256, 5, 2, 'same', activation='relu', name="feature_embedding_2")(x)
    x = keras.layers.Conv2D(512, 3, 2, 'same', activation='relu', name="feature_embedding_3")(x)

    # expansion
    h = keras.layers.Conv2DTranspose(256, 3, 2, 'same', activation='relu', name='expansion_1')(x)
    h = keras.layers.Conv2DTranspose(128, 5, 2, 'same', activation='relu', name='expansion_2')(h)
    h = keras.layers.Conv2DTranspose(TARGET_DIM, 7, 2, 'same', activation='linear', name='expansion_head')(h)

    # output
    output = keras.layers.Multiply(name="prediction")([h, mask])

    # create model
    model = keras.models.Model(inputs=(feature, mask), outputs=output)

    return model


if __name__ == '__main__':
    from tensorflow.python.keras.utils import plot_model

    plot_model(simple_cnn(15), to_file='model.png')

    # print(cnn_v6(feature_dim=15).summary())
