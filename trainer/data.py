from trainer.utils.util import *


def _make_features():
    return dict(
        x=tf.io.FixedLenFeature([], tf.string),
        y=tf.io.FixedLenFeature([], tf.string),
        mask=tf.io.FixedLenFeature([], tf.string),
        height=tf.io.FixedLenFeature([], tf.int64),
        width=tf.io.FixedLenFeature([], tf.int64),
        depth_x=tf.io.FixedLenFeature([], tf.int64),
        depth_y=tf.io.FixedLenFeature([], tf.int64),
        name=tf.io.FixedLenFeature([], tf.string)
    )


def _parser(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features=_make_features())

    def process_image(img, shape):
        img = tf.io.decode_raw(img, tf.float32)
        return tf.reshape(img, shape)

    shape_x = [features['height'], features['width'], features['depth_x']]
    shape_y = [features['height'], features['width'], features['depth_y']]
    shape_mask = [features['height'], features['width'], 1]

    x = process_image(features['x'], shape_x)
    y = process_image(features['y'], shape_y)
    mask = process_image(features['mask'], shape_mask)

    name = features['name']

    return {'feature': x, 'mask': mask, 'name': name}, {'prediction': y}


def make_dataset(path_tfrecords, batch_size, mode):
    print(path_tfrecords)
    path = glob.glob(os.path.join(path_tfrecords, mode + '_*.tfrecords'))
    print(path)

    dataset = tf.data.TFRecordDataset(path)

    dataset = dataset.map(_parser)
    if mode == 'train':
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)

    return dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = make_dataset('data/processed/boundary_edge_surface_0.25/tfrecords', 2, 'train')

    d, _ = ds.make_one_shot_iterator().get_next()
    d = d['feature']
    with tf.Session() as sess:
        try:
            d = sess.run(d)
            print(d.shape)
            print(d[0, 0, 0,:])
            for i in range(0, d.shape[3]):
                plt.figure()
                plt.imshow(d[13, :, :, i])
            plt.show()
        except tf.errors.OutOfRangeError:
            print('eof')
    # print(b.shape)
    # print(c.shape)
