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
from loss import classifier_loss, confidence_function, top_loss, acc_metrix, indices, super_loss, compute_loss
import copy


def estimate(classifier, x_logit, threshold, label, n, method='top'):
    _, sigma = super_loss(classifier, x_logit, label, out_put=2, on_train=False)
    if (method == 'top'):
        top_n = x_logit.numpy()[tf.where((tf.less(tf.argsort(sigma, direction='DESCENDING'), n))
                                    & (tf.greater_equal(sigma, threshold))).numpy()]
    else:
        valid = x_logit.numpy()[tf.where(tf.greater_equal(sigma, threshold)).numpy()]
        return tf.Variable(tf.random.shuffle(valid)[:n])
    return tf.squeeze(tf.Variable(top_n), 1)

def transfer_to_data(d, l, nums, dataset):
    data = np.zeros([nums, dataset.shape[0], dataset.shape[1], dataset.shape[2]])
    label = np.zeros([nums,])
    s = 0
    for i in range(len(d)):
        assert (d[i].shape[0] == l[i].shape[0])
        data[s:s+d[i].shape[0], :, :, :] = d[i]
        label[s:s+l[i].shape[0],] = l[i]
        s += d[i].shape[0]
    return data, label

def high_performance(model, classifier, cls, x, oversample, y, oversample_label, method):
    o_optimizer = tf.keras.optimizers.Adam(2e-4, 0.5)
    with tf.GradientTape() as m_one_tape, tf.GradientTape() as sim_tape:
        gen_loss, _, m_one_loss = compute_loss(model, classifier, x,
                                    y, method=method)
    '''
    sim_gradients = sim_tape.gradient(gen_loss, model.trainable_variables)
    sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
    '''
    m_one_gradients = m_one_tape.gradient(m_one_loss, classifier.trainable_variables)
    o_optimizer.apply_gradients(zip(m_one_gradients, classifier.trainable_variables))
    m_one_pre = classifier.call(x)
    m_one_acc = np.sum(m_one_pre.numpy().argmax(-1) == y)
    if (oversample.shape[0] > 0):
        t_cls = copy.copy(classifier)
        with tf.GradientTape() as m_two_tape, tf.GradientTape() as sim_tape:
            gen_loss, _, m_two_loss = compute_loss(model, t_cls, oversample,
                                                   oversample_label, method=method)
        '''
        sim_gradients = sim_tape.gradient(gen_loss, model.trainable_variables)
        sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
        '''
        m_two_gradients = m_two_tape.gradient(m_two_loss, t_cls.trainable_variables)
        o_optimizer.apply_gradients(zip(m_two_gradients, t_cls.trainable_variables))
        m_two_pre = t_cls.call(x)
        m_two_acc = np.sum(m_two_pre.numpy().argmax(-1) == y)
        if (m_two_acc >= m_one_acc):
            classifier = t_cls
        _, sigma = super_loss(classifier, oversample, oversample_label, out_put=2, on_train=False)
        #margin = 0.01*(m_two_acc-m_one_acc) * tf.abs(classifier.threshold[cls] - np.mean(sigma))
        #classifier._accumulate_threshold(cls, margin)
    return classifier


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

def start_train(epochs, n, threshold_list, method, model, classifier, dataset,
                train_set, test_set, sample_size, date, filePath):
    optimizer_list = []
    checkpoints_list = []
    classifier_list= []
    result_dir_list = []
    '''
    if (model.data == 'mnist'):
        file = np.load('../dataset/mnist_oversample_latent.npz')
        latent = file['latent'].squeeze(2)
        latent_len = latent.shape[0]
        mnist_train_len = np.load('../dataset/mnist_dataset.npz')['train_images'].shape[0]
        block = np.ceil(mnist_train_len/32)
        batch_size = int(np.ceil(latent_len/block))
        latent = (tf.data.Dataset.from_tensor_slices(latent)
                    .shuffle(len(latent), seed=1).batch(batch_size))
    elif(model.data == 'fashion_mnist'):
        file = np.load('../dataset/fashion_mnist_features')
        latent = file['fashion_mnist_features']
        latent_len = latent.shape[0]
        fashion_mnist_len = np.load('../dataset/fashion_mnist_dataset.npz')['train_images'].shape[0]
        block = np.ceil(fashion_mnist_len)
        batch_size = int(np.ceil(latent_len/block))
        latent = (tf.data.Dataset.from_tensor_slices(latent)
                  .shuffle(len(latent), seed=1).batch(batch_size))
    '''
    for i in range(len(threshold_list)):
        optimizer_list.append(tf.keras.optimizers.Adam(1e-4))
        result_dir = "./score/{}/{}/{}".format(date, filePath, i)
        result_dir_list.append(result_dir)
        checkpoint_path = "./checkpoints/{}/{}/{}".format(date, filePath, i)
        o_classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls,
                                  threshold=threshold_list[i])
        ckpt = tf.train.Checkpoint(sim_clr=model,
                                   clssifier=classifier,
                                   o_classifier=o_classifier,
                                   )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        classifier_list.append(o_classifier)
        checkpoints_list.append(ckpt_manager)
    def train_step(model, classifier, classifier_list, x,  y, oversample=False, metrix_list=None, features=None):
        if (oversample):
            if(features == None):
                mean, logvar = model.encode(x)
                features = model.reparameterize(mean, logvar)
            for i in range(len(classifier_list)):
                # get the accuracy during training
                label_on_train = classifier_list[i].call(x).numpy().argmax(-1)
                metrix_list[i]['train_acc'].append(np.sum(label_on_train==y.numpy())/len(y.numpy()))
                test = np.linspace(-4, 4, sample_size)
                features_set = np.zeros([sample_size, features.shape[1] * features.shape[0], 7])
                for features_idx in range(features.shape[1]):
                    for j in range(len(test)):
                        tmp = features.numpy().copy()
                        tmp[:, features_idx] = test[j]
                        features_set[j, features_idx*features.shape[0]:(features_idx+1)*features.shape[0], :] = tmp
                features_set = tf.Variable(
                    features_set.reshape(features.shape[0]*features.shape[1]*sample_size, 7))
                for cls in range(5, model.num_cls):
                    # oversampling
                            sample_label = np.array(([cls] * features_set.shape[0]))
                            z = tf.concat([features_set, np.expand_dims(sample_label, 1)], axis=1)
                            x_logit = model.sample(z)
                            threshold = classifier_list[i].threshold
                            m_sample = estimate(classifier, x_logit,
                                               threshold[cls], sample_label, n=n)
                            sample_y = sample_label[:m_sample.shape[0]]
                            s_sample = estimate(classifier_list[i], x_logit,
                                               threshold[cls], sample_label, n=n)
                            o_sample_y = sample_label[:s_sample.shape[0]]
                            #total_sample_idx = merge_list(s_index[0], m_index[0])
                            metrix_list[i]['valid_sample'].append([len(sample_y),
                                                           len(o_sample_y)])
                            metrix_list[i]['valid_sample_data'].append(m_sample)
                            metrix_list[i]['valid_sample_label'].append(sample_y)
                            metrix_list[i]['total_sample'] = metrix_list[i]['total_sample'] + list(sample_label)
                            metrix_list[i]['total_valid_sample'] = metrix_list[i]['total_valid_sample'] + list(sample_y)
                            classifier_list[i] = high_performance(model, classifier_list[i], cls, x,
                                                          m_sample, y, sample_y, method=method)
            return metrix_list

    for epoch in range(epochs):
        metrix_list = []
        for _ in threshold_list:
            metrix = {}
            metrix['valid_sample'] = []
            metrix['total_sample'] = []
            metrix['total_valid_sample'] = []
            metrix['train_acc'] = []
            metrix['valid_sample_data'] = []
            metrix['valid_sample_label'] = []
            metrix_list.append(metrix)

        start_time = time.time()
        #if (model.data == 'celebA' or model.data == "large_celebA"):
        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            metrix_list = train_step(model, classifier, classifier_list,
            x, y, oversample=True, metrix_list=metrix_list)

        '''
        elif (model.data == 'mnist'):
            for x,z,y in tf.data.Dataset.zip((train_set[0], latent, train_set[1])):
                metrix_list = train_step(model, classifier, classifier_list,
                x, y, features=z, oversample=True, metrix_list=metrix_list)

        '''
            #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%1 == 0:

            print('*' * 20)
            end_time = time.time()
            print("Epoch: {}, time elapse for current epoch: {}".format(epoch + 1, end_time - start_time))
            h, _ = classifier_loss(classifier, test_set[0], test_set[1], method=method)
            pre_acsa, pre_g_mean, pre_tpr, pre_confMat, pre_acc = indices(h.numpy().argmax(-1), test_set[1])
            for i in range(len(threshold_list)):
                o_h, _ = classifier_loss(classifier_list[i], test_set[0], test_set[1], method=method)
                oAsca, oGMean, o_tpr, o_confMat, o_acc = indices(o_h.numpy().argmax(-1), test_set[1])
                pre_train_g_mean_acc = pre_g_mean
                pre_train_acsa_acc = pre_acsa
                o_acsa_acc = oAsca
                o_g_mean_acc = oGMean
                train_acc = np.mean(np.array(metrix_list[i]['train_acc']))
                valid_sample = np.array(metrix_list[i]['valid_sample'])
                total_sample = np.array(metrix_list[i]['total_sample'])
                pass_pre_train_classifier = np.sum(valid_sample[:, 0])/len(total_sample.flatten())
                pass_o_classifier = np.sum(valid_sample[:, 1])/len(total_sample.flatten())
                total_valid_sample = np.array(metrix_list[i]['total_valid_sample'])

                ckpt_save_path = checkpoints_list[i].save()
                print('Saving checkpoint at {}'.format(ckpt_save_path))

                result_dir = result_dir_list[i]

                valid_sample_data, valid_sample_label = transfer_to_data(metrix_list[i]['valid_sample_data'],
                                                                         metrix_list[i]['valid_sample_label'],
                                                                         np.sum(valid_sample[:, 0]),
                                                                                dataset)


                result = {
                    "pre_g_mean": pre_train_g_mean_acc,
                    'pre_acsa': pre_train_acsa_acc,
                    'o_g_mean': o_g_mean_acc,
                    'o_acsa': o_acsa_acc,
                    'acc_in_training': train_acc,
                    'pass_pre_train_classifier': pass_pre_train_classifier,
                    'pass_o_classifier': pass_o_classifier
                }
                current_threshold = classifier_list[i].threshold.numpy()
                for cls in range(model.num_cls):
                    cls_acc = 'acc_in_cls{}'.format(cls)
                    name = 'valid_ratio_in_cls{}'.format(cls)
                    valid_sample_name = 'valid_sample_in_cls{}'.format(cls)
                    result[cls_acc] = o_tpr[cls]
                    threshold_id = 'threshold_in_cls{}'.format(cls)
                    result[threshold_id] = current_threshold[cls]
                    valid_sample_num = np.sum(total_valid_sample == cls)
                    total_gen_num = np.sum(total_sample.flatten() == cls)
                    result[valid_sample_name] = valid_sample_num
                    if (valid_sample_num == 0):
                        result[name] = 0

                    else:
                        result[name] = valid_sample_num / total_gen_num
                if os.path.isfile(result_dir + '/result.csv'):
                    e = pd.read_csv(result_dir + '/result.csv').index[-1] + 1
                else:
                    e = epoch + 1
                df = pd.DataFrame(result, index=[e], dtype=np.float32)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                if not os.path.isfile(result_dir+'/result.csv'):
                    df.to_csv(result_dir+'/result.csv')
                    if not os.path.isfile(result_dir +'dataset.npz'):
                        np.savez(result_dir + '/dataset', dataset=valid_sample_data, labels=valid_sample_label)
                else:  # else it exists so append without writing the header
                    df.to_csv(result_dir+'/result.csv', mode='a', header=False)


                print('threshld:{} , \n'
                      ' o_g_means:{},  o_acsa:{}, \n'
                      .format(classifier_list[i].threshold,
                              o_g_mean_acc, o_acsa_acc,
                              ))
                print("-" * 20)
            print('*' * 20)
    #compute_and_save_inception_score(model, file_path)



if __name__ == '__main__':
    target = 'margin'
    threshold = 0.85
    shape = [28, 28, 1]
    mbs = tf.losses.MeanAbsoluteError()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    (mnist_images, mnist_labels), (test_images, testset_labels) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images, shape=shape)
    test_images = preprocess_images(test_images, shape=shape)
    irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
    majority_images = mnist_images[np.where(mnist_labels==0)][:irs[0]]
    majority_labels = [0] * irs[0]
    train_images, train_labels = imbalance_sample(mnist_images, mnist_labels, irs)
    num_examples_to_generate = 16
    epochs = 200
    batch_size = 32
    sim_clr = F_VAE(model='cnn')
    classifier = Classifier(shape=[28, 28, 1], model='cnn')

    train_images = (tf.data.Dataset.from_tensor_slices(train_images)
            .shuffle(len(train_images), seed=1).batch(batch_size))

    train_labels = (tf.data.Dataset.from_tensor_slices(train_labels)
                    .shuffle(len(train_labels), seed=1).batch(batch_size))

    majority_images = (tf.data.Dataset.from_tensor_slices(majority_images)
            .shuffle(len(majority_images), seed=2).batch(batch_size))

    majority_labels = (tf.data.Dataset.from_tensor_slices(majority_labels)
            .shuffle(len(majority_labels), seed=2).batch(batch_size))

    test_images = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(len(test_images), seed=1).batch(batch_size))

    testset_labels = (tf.data.Dataset.from_tensor_slices(testset_labels)
                    .shuffle(len(testset_labels), seed=1).batch(batch_size))


    date = '7_23'
    file_path = 'mnist_test21'
    start_train(epochs, target, threshold, sim_clr, classifier, [train_images, train_labels], [majority_images, majority_labels],
                [test_images, testset_labels], date, file_path)



cls = ['blonde','black','bald','brown','gray']