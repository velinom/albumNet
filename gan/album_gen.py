import os

from gan.utils import *

HEIGHT, WIDTH, CHANNEL = 256, 256, 3
BATCH_SIZE = 64
EPOCH = 5000

version = 'newAlbums'
new_covers_path = './' + version


def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)


def process_data():
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    albums_dir = os.path.join(current_dir, 'pop')
    images = []
    for each in os.listdir(albums_dir):
        images.append(os.path.join(albums_dir, each))
    # print images
    all_images = tf.convert_to_tensor(images, dtype=tf.string)

    images_queue = tf.train.slice_input_producer(
        [all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.resize_images(tf.image.decode_jpeg(content, channels=CHANNEL), size=[HEIGHT, WIDTH])
    image.set_shape([HEIGHT, WIDTH, CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch(
        [image], batch_size=BATCH_SIZE,
        num_threads=8, capacity=200 + 3 * BATCH_SIZE,
        min_after_dequeue=200)
    num_images = len(images)

    return images_batch, num_images


def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32  # channel num
    s4 = 8
    output_dim = CHANNEL
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        drop1 = tf.nn.dropout(act1, keep_prob=0.8)

        # 8*8*256
        conv2 = tf.layers.conv2d_transpose(drop1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        drop2 = tf.nn.dropout(act2, keep_prob=0.8)

        conv3 = tf.layers.conv2d_transpose(drop2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        drop3 = tf.nn.dropout(act3, keep_prob=0.8)

        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(drop3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        drop4 = tf.nn.dropout(act4, keep_prob=0.8)

        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(drop4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        drop5 = tf.nn.dropout(act5, keep_prob=0.8)

        conv6 = tf.layers.conv2d_transpose(drop5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,
        # updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 128, 256, 512, 1024
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn1')
        act1 = lrelu(bn1, n='act1')
        drop1 = tf.nn.dropout(act1, keep_prob=.7)

        conv2 = tf.layers.conv2d(drop1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        drop2 = tf.nn.dropout(act2, keep_prob=.7)

        conv3 = tf.layers.conv2d(drop2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
        drop3 = tf.nn.dropout(act3, keep_prob=.7)

        conv4 = tf.layers.conv2d(drop3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')

        w2 = tf.get_variable('w2', shape=[dim, 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)
        return logits


def train():
    random_dim = 100

    with tf.variable_scope('input'):
        # real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    # wgan
    fake_image = generator(random_input, random_dim, is_train)

    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    # test
    # print(d_vars)
    global_step_g = tf.Variable(0, trainable=False)
    learning_rate_g = 0.07
    learning_rate_g = tf.train.exponential_decay(learning_rate_g, global_step_g, 10000, .3, staircase=True)
    global_step_d = tf.Variable(0, trainable=False)
    learning_rate_d = tf.train.exponential_decay(learning_rate_g, global_step_g, 10000, .3, staircase=True)
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate_d).minimize(d_loss, var_list=d_vars,
                                                                                  global_step=global_step_d)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=learning_rate_g).minimize(g_loss, var_list=g_vars,
                                                                                  global_step=global_step_g)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()

    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "./tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(EPOCH):
        print('i: ',i)
        for j in range(batch_num):
            print('j: ',j)
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                train_image = sess.run(image_batch)
                # wgan clip weights
                sess.run(d_clip)
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
            # Update the generator
            for k in range(g_iters):
                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

        # save check point every 500 epoch
        if i % 500 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' + version + '/' + str(i))
        if i % 50 == 0:
            # save images
            if not os.path.exists(new_covers_path):
                os.makedirs(new_covers_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            save_images(imgtest, [8, 8], new_covers_path + '/epoch' + str(i) + '.jpg')

            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":
    train()













