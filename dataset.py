import numpy as np
from tensorflow_addons.image import rotate
import pandas as pd
import tensorflow as tf

def preprocess_images(images, shape):
  images = images.reshape((images.shape[0], shape[0], shape[1], shape[2])) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

def divide_dataset(train_data, train_labels, sample_size):
  labels = pd.DataFrame({'labels': train_labels})
  dataset = []
  for i in range(0, 10):
    idx = labels[labels.labels == i].iloc[:sample_size].index
    train_images = train_data[idx]
    dataset.append(train_images)
  return np.array(dataset).reshape(10 * sample_size, 28, 28, 1)

def rotate_dataset(image, label, rotate_set):
  s = rotate_set[0]
  e = rotate_set[1]
  dataset = image
  labelset = label
  for degree in range(s+10, e+10, 10):
    d = np.radians(degree)
    r_image = rotate(image, d)
    dataset = np.concatenate([r_image, dataset])
    labelset = np.concatenate([labelset, label])
  return dataset, labelset


def imbalance_sample(data, labels, irs):
  dataset = np.zeros([sum(irs), data.shape[1], data.shape[2], data.shape[3]]).astype('float32')
  label_set = np.zeros([sum(irs)], dtype=int)
  s = 0
  for i in range(len(irs)):
    tmp_sample = data[np.where(labels==i)]
    max_index = tmp_sample.shape[0]
    sample_index = np.random.randint(0, max_index, irs[i])
    dataset[s:s + irs[i], :, :, :] = tmp_sample[sample_index]
    label_set[s:s + irs[i]] = [i] * irs[i]
    s += irs[i]
  return dataset, label_set




class Dataset():

  def __init__(self, dataset, batch_size=32):
    self.batch_size = batch_size
    self.dataset = dataset
    self.switcher = {
      'mnist': np.load('./dataset/mnist_dataset1.npz'),
      'celebA': np.load('./dataset/celebA_dataset.npz'),
      'large_celebA': np.load('./dataset/celebA_large_dataset.npz'),
      #'fashion_mnist': np.load('../dataset/fashion_mnist.npz')
    }

    if (dataset == 'mnist'):
      self.shape = (28, 28, 1)
      self.num_cls = 10
      self.latent_dims = 8
      self.irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    elif (dataset == 'fashionMNIST'):
      self.shape = (28, 28, 1)
      self.num_cls = 10
      self.latent_dims = 16
      self.irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    elif (dataset == 'celebA' or dataset == 'large_celebA'):
      self.shape = (64, 64, 3)
      self.num_cls = 5
      self.latent_dims = 256
      self.irs = [15000, 1500, 750, 300, 150]

  def load_data(self, normalize=True):
    datasets = self.switcher[self.dataset]
    (train_images, train_labels)  = datasets['train_images'], datasets['train_labels']
    (test_images, test_labels) = datasets['test_images'], datasets['test_labels']

    if (normalize):
      train_images = (tf.data.Dataset.from_tensor_slices(train_images)
                      .shuffle(len(train_images), seed=1).batch(self.batch_size))

      train_labels = (tf.data.Dataset.from_tensor_slices(train_labels)
                      .shuffle(len(train_labels), seed=1).batch(self.batch_size))
    return (train_images, train_labels), (test_images, test_labels)

if __name__ == '__main__':
  (train_set, train_labels), (test_set, test_labels) = tf.keras.datasets.mnist.load_data()
  train_set = preprocess_images(train_set, [28, 28, 1])
  test_set = preprocess_images(test_set, [28, 28, 1])
  shape = (28, 28, 1)
  num_cls = 10
  latent_dims = 16
  irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
  train_images, train_labels = imbalance_sample(train_set, train_labels, irs)
  test_irs = [100] * len(irs)
  test_images, test_labels = imbalance_sample(test_set, test_labels, test_irs)
  np.savez('./dataset/mnist_dataset1', train_images=train_images, train_labels=train_labels,
          test_images=test_images, test_labels=test_labels)



