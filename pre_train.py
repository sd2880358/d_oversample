import tensorflow as tf
from over_sampling import start_train
from dataset import preprocess_images, divide_dataset, imbalance_sample, Dataset
from model import CVAE, Classifier, F_VAE
from celebA import CelebA
from load_data import split
import tensorflow as tf
from model import CVAE, Classifier, F_VAE
from dataset import preprocess_images, divide_dataset, imbalance_sample
from tensorflow_addons.image import rotate
import time
from tensorflow.linalg import matvec
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import math
import pandas as pd
from loss import compute_loss, confidence_function, top_loss, acc_metrix, indices


def estimate(classifier, x_logit, threshold, label, target):
    conf, l = confidence_function(classifier, x_logit, target=target)
    return np.where((conf>=threshold) & (l==label))

def merge_list(l1, l2):
    in_l1 = set(l1)
    in_l2 = set(l2)
    in_l1_not_in_l2 = in_l1 - in_l2
    return list(l2) + list(in_l1_not_in_l2)


def latent_triversal(model, classifier, x, y, r, n):
    mean, logvar = model.encode(x)
    features = model.reparameterize(mean, logvar).numpy()
    triversal_range = np.linspace(-r, r, n)
    acc = tf.keras.metrics.Mean()
    for dim in range(features.shape[1]):
        for replace in triversal_range:
            features[:, dim] = replace
            z = tf.concat([features, tf.expand_dims(y,1)], axis=1)
            x_logit = model.sample(z)
            conf, l = confidence_function(classifier, x_logit, target=target)
            sample = x_logit.numpy()[np.where((conf >= threshold) & (l == y))]
            if (len(sample)==0):
                acc(0)
            else:
                acc(len(sample)/len(y))
    return acc.result()

def start_train(epochs, c_epochs, model, classifier, method,
                train_set, test_set, date, filePath):
    sim_optimizer = tf.keras.optimizers.Adam(1e-4)
    cls_optimizer = tf.keras.optimizers.Adam(1e-4)
    def train_step(model, classifier, x, y, epoch, sim_optimizer,
                   cls_optimizer):
            with tf.GradientTape() as sim_tape, tf.GradientTape() as cls_tape:
                ori_loss, _, encode_loss = compute_loss(model, classifier, x, y, method=method)
            sim_gradients = sim_tape.gradient(ori_loss, model.trainable_variables)
            sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
            if (epoch < c_epochs):
                cls_gradients = cls_tape.gradient(encode_loss, classifier.trainable_variables)
                cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
    checkpoint_path = "./checkpoints/{}/{}".format(date, filePath)
    ckpt = tf.train.Checkpoint(sim_clr=model,
                               clssifier=classifier,
                               optimizer=sim_optimizer,
                               cls_optimizer=cls_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)

    result_dir = "./score/{}/{}".format(date, filePath)
    if os.path.isfile(result_dir+'/result.csv'):
        e = pd.read_csv(result_dir+'/result.csv').index[-1]
    else:
        e = 0
    for epoch in range(epochs):
        e += 1
        start_time = time.time()
        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            train_step(model, classifier, x, y, epoch, sim_optimizer, cls_optimizer)

        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%5 == 0:

            end_time = time.time()
            elbo_loss = tf.keras.metrics.Mean()
            pre_train_g_mean = tf.keras.metrics.Mean()
            pre_train_acsa = tf.keras.metrics.Mean()
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            ori_loss, h, _ = compute_loss(model, classifier, test_set[0], test_set[1])
            pre_acsa, pre_g_mean, pre_tpr, pre_confMat, pre_acc = indices(h.numpy().argmax(-1), test_set[1])
            total_loss = ori_loss

            pre_train_g_mean(pre_g_mean)
            pre_train_acsa(pre_acsa)
            elbo_loss(total_loss)
            elbo = -elbo_loss.result()
            pre_train_g_mean_acc = pre_train_g_mean.result()
            pre_train_acsa_acc = pre_train_acsa.result()


            result = {
                "elbo": elbo,
                "pre_g_mean": pre_train_g_mean_acc,
                'pre_acsa': pre_train_acsa_acc
            }
            df = pd.DataFrame(result, index=[e], dtype=np.float32)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            if not os.path.isfile(result_dir+'/result.csv'):
                df.to_csv(result_dir+'/result.csv')
            else:  # else it exists so append without writing the header
                df.to_csv(result_dir+'/result.csv', mode='a', header=False)

            print('*' * 20)
            print('Epoch: {}, elbo: {}, \n'
                  ' pre_g_means: {}, pre_acsa: {}, \n,'
                  'time elapse for current epoch: {}'
                  .format(epoch+1, elbo,pre_train_g_mean_acc,
                          pre_train_acsa_acc,
                          end_time - start_time))
            print('*' * 20)
    #compute_and_save_inception_score(model, file_path)



if __name__ == '__main__':
    os.environ["CUDA_DECICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=8168)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    target = 'margin'
    threshold = 0.95
    date = '8_18'
    data_name = 'mnist'
    file_path = 'pre_train_mnist_super_loss'
    dataset = Dataset(data_name, batch_size=32)
    epochs = 200
    c_epochs = 70
    method = 'super_loss'
    (train_set, train_labels), (test_set, test_labels) = dataset.load_data()
    sim_clr = F_VAE(data=data_name, shape=dataset.shape, latent_dim=dataset.latent_dims, model='cnn', num_cls=dataset.num_cls)
    classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls)

    start_train(epochs, c_epochs, sim_clr, classifier, method,
                [train_set, train_labels],
                [test_set, test_labels], date, file_path)




