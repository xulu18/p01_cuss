import tensorflow as tf
import numpy as np
from heart_net import cu
from data_logger import DataLogger
from evaluator import Evaluator
from class_dataset import MyDataSet

logger_path = sys.argv[1]
logger = DataLogger(logger_path)
path_train = sys.argv[2]
reader_train = MyDataSet(path_train, r'.mat', ['img_n'], ['label'])
path_test = sys.argv[3]
reader_test = MyDataSet(path_test, r'.mat', ['img_n'], ['label'])

gpu_options = tf.GPUOptions(visible_device_list=)
config = tf.ConfigProto(gpu_options=gpu_options)
config.log_device_placement = False
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession("", config=config)

x = tf.placeholder("float", shape=[None, 256, 256])
x_in = tf.expand_dims(x, -1)
y_ = tf.placeholder("float", shape=[None, 256, 256, 8])
batch_size_h = tf.placeholder(tf.int32)

g = tf.Graph()
net = cu(x_in, y_, batch_size_h)
a = tf.argmax(net.probs, 3)
b = tf.argmax(y_, 3)
g.finalize()

var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars

saver = tf.train.Saver(var_list=var_list, max_to_keep=50)
start_learning_rate = 4e-4
decay_steps = 1000
decay_rate = 0.9

logger.log_print('learning rate :' + str(start_learning_rate))
logger.log_print('decay step :' + str(decay_steps))
logger.log_print('decay size :' + str(decay_rate))
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(start_learning_rate,
                                           global_step,
                                           decay_steps,
                                           decay_rate,
                                           staircase=True)


def jaccard_coe(output, target, loss_type='d', axis=[1, 2, 3], smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'd':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
    elif loss_type == 'i':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
        dice = (inse + smooth) / (l + r - inse + smooth)
        dice = tf.reduce_mean(dice)
    else:
        raise Exception("Unknow loss_type")
    return dice


def _tf_fspecial_mean(size):
    x_data = np.ones(shape=(size, size))
    # print(x_data)
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
    x = tf.constant(x_data, dtype=tf.float32)
    g = x / tf.reduce_sum(x)
    # print(sess.run(g))
    return g

def ssim2(img1, img2, k1=0.01, k2=0.02, L=1, window_size):

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    window = _tf_fspecial_mean(window_size)
    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides = [1 ,1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

    return 1-tf.reduce_mean(ssim_map)



dice_loss = tf.reduce_sum(
    0.05 * (1 - jaccard_coe(net.probs[:, :, :, 0], y_[:, :, :, 0], loss_type='d', axis=[1, 2], smooth=1e-8)) + \
    1 * (1 - jaccard_coe(net.probs[:, :, :, 1], y_[:, :, :, 1], loss_type='d', axis=[1, 2], smooth=1e-8)) + \
    1 * (1 - jaccard_coe(net.probs[:, :, :, 2], y_[:, :, :, 2], loss_type='d', axis=[1, 2], smooth=1e-8)) + \
    1 * (1 - jaccard_coe(net.probs[:, :, :, 3], y_[:, :, :, 3], loss_type='d', axis=[1, 2], smooth=1e-8)) + \
    1 * (1 - jaccard_coe(net.probs[:, :, :, 4], y_[:, :, :, 4], loss_type='d', axis=[1, 2], smooth=1e-8)) + \
    1 * (1 - jaccard_coe(net.probs[:, :, :, 5], y_[:, :, :, 5], loss_type='d', axis=[1, 2], smooth=1e-8)) + \
    1 * (1 - jaccard_coe(net.probs[:, :, :, 6], y_[:, :, :, 6], loss_type='d', axis=[1, 2], smooth=1e-8)) + \
    1 * (1 - jaccard_coe(net.probs[:, :, :, 7], y_[:, :, :, 7], loss_type='d', axis=[1, 2], smooth=1e-8)))

cross_entropy = dice_loss


train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(net.probs, 3), tf.argmax(y_, 3))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())


# saver.restore(sess,save_path)

def one_hot(datain, index):
    shape = datain.shape
    out = np.zeros(shape=(shape[0], shape[1], shape[2], len(index)))
    for i in range(len(index)):
        out[:, :, :, i][datain == index[i]] = 1
    return out


for i in range(100001):
    batch_size = 1
    batch = reader_train.next_batch(batch_size)
    if i % 50 == 0:

        [loss, probs] = sess.run([cross_entropy, net.out],
                                 feed_dict={
                                     x: batch['img_n'],
                                     y_: one_hot(batch['label'], [0, 1, 2, 3, 4, 5, 6, 7]),
                                     batch_size_h: batch_size})
        #print(g) #,tf.gradients(ssim_loss, net.w)
        # print(il)
        logger.log_print("step %d, loss %g" % (i, loss))
    train_step.run(feed_dict={x: batch['img_n'],
                              y_: one_hot(batch['label'], [0, 1, 2, 3, 4, 5, 6, 7]),
                              batch_size_h: batch_size})
    if i % 1000 == 0:
        dice0 = 0
        dice1 = 0
        dice2 = 0
        dice3 = 0
        dice4 = 0
        dice5 = 0
        dice6 = 0
        dice7 = 0
        pa = 0
        miou = 0
        loss = 0

        for it in range(reader_test._num_files):
            batch_size_test = 1
            btest = reader_test.next_batch(batch_size_test)
            [pred, gt, loss_t,pa_t] = sess.run([a, b,
                                           cross_entropy,accuracy],
                                          feed_dict={x: btest['img_n'],
                                                     y_: one_hot(btest['label'], [0, 1, 2, 3, 4, 5, 6, 7]),
                                                     batch_size_h: batch_size_test})


            evaler = Evaluator(gt, pred)
            dice0_t = evaler.Dice(0)
            dice1_t = evaler.Dice(1)
            dice2_t = evaler.Dice(2)
            dice3_t = evaler.Dice(3)
            dice4_t = evaler.Dice(4)
            dice5_t = evaler.Dice(5)
            dice6_t = evaler.Dice(6)
            dice7_t = evaler.Dice(7)
            miou_t = evaler.IoU(6)
            del evaler
            dice0 = dice0 + dice0_t
            dice1 = dice1 + dice1_t
            dice2 = dice2 + dice2_t
            dice3 = dice3 + dice3_t
            dice4 = dice4 + dice4_t
            dice5 = dice5 + dice5_t
            dice6 = dice6 + dice6_t
            dice7 = dice7 + dice7_t
            pa = pa + pa_t
            miou = miou + miou_t
            loss = loss + loss_t

        dice0 = dice0 / reader_test._num_files
        dice1 = dice1 / reader_test._num_files
        dice2 = dice2 / reader_test._num_files
        dice3 = dice3 / reader_test._num_files
        dice4 = dice4 / reader_test._num_files
        dice5 = dice5 / reader_test._num_files
        dice6 = dice6 / reader_test._num_files
        dice7 = dice7 / reader_test._num_files
        pa = pa / reader_test._num_files
        miou = miou / reader_test._num_files
        loss = loss / reader_test._num_files


        logger.log_print('testing loss:' + str(loss))
        logger.log_print('pa %.4f  miou %.4f'% (pa,miou))
        logger.log_print('Dice of different parts:%.4f |%.4f |%.4f |%.4f |%.4f |%.4f |%.4f |%.4f'
                         % (dice0, dice1, dice2, dice3, dice4, dice5, dice6, dice7))

    if i % 1000 == 0:
        step_final = sess.run(global_step)
        saver.save(sess, logger_path + r'/model/model' + str(step_final))

