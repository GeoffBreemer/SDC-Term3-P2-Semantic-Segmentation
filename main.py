import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Constants
MODEL_SAVE_PATH = "./models"                        # Filename of the TensorFlow model
EPOCHS = 15                                         # Number of epochs
BATCH_SIZE = 10                                     # Reduce this depending on amount of RAM available
DROPOUT_KEEP_PROB = 0.8
LEARNING_RATE = 0.001                               # Initial learning rate
DATA_PATH = './data'
RUNS_PATH = './runs'
NUM_CLASSES = 2


def save_model(sess, epoch):
    '''Save TensorFlow model variables to disk, the current epoch becomes part of the name'''
    saver = tf.train.Saver()
    save_path = saver.save(sess, MODEL_SAVE_PATH + '/P2-epoch' + str(epoch) + '.ckpt')
    print("Model saved in file: %s" % save_path)


def load_model(sess, epoch=EPOCHS):
    '''Load previously saved TensorFlow model variables of a particular epoch'''
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_SAVE_PATH + '/P2-epoch' + str(epoch) + '.ckpt')


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out = graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out = graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

tests.test_load_vgg(load_vgg, tf)


def layer_1x1_conv(layer, num_classes, layer_name):
    return tf.layers.conv2d(layer, num_classes, 1,
                            strides=(1, 1),
                            name=layer_name+'_1x1_conv',
                            padding='same',
                            # kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def layer_transposed(layer, num_classes, layer_name, kernel=4, strides=(2, 2), padding='same'):
    return tf.layers.conv2d_transpose(layer, num_classes,
                                      kernel,
                                      strides = strides,
                                      padding = padding,
                                      name = layer_name + '_transposed_conv',
                                      # kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def layer_skip_connection(layer1, layer2, num_classes, layer_name):
    return tf.add(layer1, layer2, name = layer_name + '_skip_connection')


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # 1x1 convolution
    vgg_layer7_1x1 = layer_1x1_conv(vgg_layer7_out, num_classes, 'vgg_layer7')

    # Deconvolution
    trans1 = layer_transposed(vgg_layer7_1x1, num_classes, 'transp_conv1')

    # 1x1 convolution
    vgg_layer4_1x1 = layer_1x1_conv(vgg_layer4_out, num_classes, 'vgg_layer4')

    # Skip
    skip1 = layer_skip_connection(trans1, vgg_layer4_1x1, num_classes, 'skip_conn1')

    # Deconvolution
    trans2 = layer_transposed(skip1, num_classes, 'transp_conv2')

    # 1x1 convolution
    vgg_layer3_1x1 = layer_1x1_conv(vgg_layer3_out, num_classes, 'vgg_layer3')

    # Skip
    skip2 = layer_skip_connection(trans2, vgg_layer3_1x1, num_classes, 'skip_conn2')

    # Deconvolution
    last = layer_transposed(skip2, num_classes, 'output', kernel=16, strides=(8, 8))

    return last

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                                                                labels = correct_label))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, optimizer, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input image
    :param correct_label: TF Placeholder for label image
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # log_dir = '/tmp/tf/adl/logs'
    # if tf.gfile.Exists(log_dir):
    #     tf.gfile.DeleteRecursively(log_dir)
    # tf.gfile.MakeDirs(log_dir)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        image_batches = []
        for i in range(epochs):
            batch = 0
            print('Epoch %d' % (i))
            for image, label in get_batches_fn(batch_size):
                image_batches.append((image, label))
                batch += 1

                loss, _ = sess.run([cross_entropy_loss, train_op],
                                   feed_dict={input_image: image,
                                   correct_label: label,
                                   learning_rate: LEARNING_RATE,
                                   keep_prob: DROPOUT_KEEP_PROB})

                print ('Batch %4d cross_entropy_loss %.03f' % (batch, loss))


            save_model(sess, i)

# tests.test_train_nn(train_nn)


def run():
    image_shape = (160, 576)
    print("Load data")
    tests.test_for_kitti_dataset(DATA_PATH)

    # Download pretrained vgg model
    print("Load VGG")
    helper.maybe_download_pretrained_vgg(DATA_PATH)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_PATH, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_PATH, 'data_road/training'), image_shape)

        # 1. Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUM_CLASSES)
        learning_rate = tf.placeholder(tf.float32, name = 'learning-rate')

        correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], NUM_CLASSES),
                                       name='correct-label')

        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, NUM_CLASSES)

        # 2. Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op,
                 cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # 3. Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_PATH, DATA_PATH, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    print("Starting")
    run()
