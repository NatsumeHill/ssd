import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from preprocessing import ssd_vgg_preprocessing
import ssd_vgg_300
import tf_utils
from datasets import pascalvoc_2012
from collections import namedtuple

slim = tf.contrib.slim

# 使用CPU训练时必须修改为“NHWC‘
DATA_FORMAT = 'NCHW'

config = namedtuple('config', ['train_dir',
                               'dataset_dir',
                               'dataset_split_name',
                               'checkpoint_path',
                               'model_name',
                               'checkpoint_model_scope',
                               'checkpoint_exclude_scopes',
                               'trainable_scopes',
                               'save_summaries_secs',
                               'save_interval_secs',
                               'weight_decay',
                               'learning_rate',
                               'learning_rate_decay_factor',
                               'batch_size',
                               'num_classes',
                               'num_readers',
                               'num_preprocessing_threads',
                               'match_threshold',
                               'negative_ratio',
                               'loss_alpha',
                               'num_epochs_per_decay',
                               'log_every_n_steps',
                               'max_number_of_steps',
                               'ignore_missing_vars'
                               ])

# FLAGS = config(
#     train_dir='./log',
#     dataset_dir='./tfrecords',
#     dataset_split_name='train',
#     model_name='ssd_300_vgg',
#     checkpoint_path='./checkpoints/vgg_16.ckpt',
#     checkpoint_model_scope='vgg_16',
#     checkpoint_exclude_scopes='ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box',
#     trainable_scopes='ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box',
#     save_summaries_secs=60,
#     save_interval_secs=600,
#     weight_decay=0.0005,
#     learning_rate=0.001,
#     learning_rate_decay_factor=0.94,
#     batch_size=32,
#     num_classes=21,
#     num_readers=4,
#     num_preprocessing_threads=4,
#     match_threshold=0.5,
#     negative_ratio=3.0,
#     loss_alpha=1.0,
#     num_epochs_per_decay=2.0,
#     log_every_n_steps=10,
#     max_number_of_steps=None,
#     ignore_missing_vars=False
# )
FLAGS = config(
    train_dir='./log',
    dataset_dir='./tfrecords',
    dataset_split_name='train',
    model_name='ssd_300_vgg',
    checkpoint_path='./checkpoints/ssd_300_vgg.ckpt',
    checkpoint_model_scope=None,
    checkpoint_exclude_scopes=None,
    trainable_scopes=None,
    save_summaries_secs=60,
    save_interval_secs=600,
    weight_decay=0.0005,
    learning_rate=0.001,
    learning_rate_decay_factor=0.94,
    batch_size=32,
    num_classes=21,
    num_readers=4,
    num_preprocessing_threads=4,
    match_threshold=0.5,
    negative_ratio=3.0,
    loss_alpha=1.0,
    num_epochs_per_decay=2.0,
    log_every_n_steps=10,
    max_number_of_steps=None,
    ignore_missing_vars=False
)

def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Create global_step.
        global_step = slim.create_global_step()

        # Select the dataset.
        dataset = pascalvoc_2012.get_split('train', FLAGS.dataset_dir)

        # Get the SSD network and its anchors.
        ssd_class = ssd_vgg_300.SSDNet
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        # 计算所有先验框位置和大小[anchor=(x,y,h,w)....]
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.name_scope('pascalvoc_2012_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=20 * FLAGS.batch_size,
                common_queue_min=10 * FLAGS.batch_size,
                shuffle=True)
        # Get for SSD network: image, labels, bboxes.
        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox'])
        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = \
            ssd_vgg_preprocessing.preprocess_image(image, glabels, gbboxes,
                                                   out_shape=ssd_shape,
                                                   data_format=DATA_FORMAT,
                                                   is_training=True)
        # Encode groundtruth labels and bboxes.
        gclasses, glocalisations, gscores = \
            ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
        batch_shape = [1] + [len(ssd_anchors)] * 3

        # Training batches and queue.
        r = tf.train.batch(
            tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)
        b_image, b_gclasses, b_glocalisations, b_gscores = \
            tf_utils.reshape_list(r, batch_shape)

        # Intermediate queueing
        batch_queue = slim.prefetch_queue.prefetch_queue(
            tf_utils.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
            capacity=2)

        # Dequeue batch.
        b_image, b_gclasses, b_glocalisations, b_gscores = \
            tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)

        # Construct SSD network.
        # 读取网络中的默认参数
        arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                      data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = \
                ssd_net.net(b_image, is_training=True)
        # Add loss function.
        ssd_net.losses(logits, localisations,
                       b_gclasses, b_glocalisations, b_gscores,
                       match_threshold=FLAGS.match_threshold,
                       negative_ratio=FLAGS.negative_ratio,
                       alpha=FLAGS.loss_alpha,
                       label_smoothing=0.0)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # =================================================================== #
        # Add summaries.
        # =================================================================== #
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Add summaries for end_points.
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))
        # Add summaries for losses and extra losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        for loss in tf.get_collection('EXTRA_LOSSES'):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        # =================================================================== #
        # Configure the optimization procedure.
        # =================================================================== #
        learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                         dataset.num_samples,
                                                         global_step)
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1.0)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        # Variables to train.
        variables_to_train = tf_utils.get_variables_to_train(FLAGS)

        # and returns a train_tensor and summary_op
        total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
        gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        # 将所有的更新操作组合成一个operation
        update_op = tf.group(*update_ops)
        # 保证所有的更新操作执行后，才获取total_loss
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                          name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)
        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            init_fn=tf_utils.get_init_fn(FLAGS),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            saver=saver,
            save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
    tf.app.run()
