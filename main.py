import tensorflow as tf
from over_sampling import start_train
from dataset import Dataset
from model import Classifier, F_VAE
import os

if __name__ == '__main__':
    ''' gpu setup
    os.environ["CUDA_DECICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=8168)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    '''
    n = 5
    sample_size = 21
    batch_size = 32
    threshold = [0.998, 0.997, 0.966, 0.957, 0.967, 0.95, 0.951, 0.95, 0.95 , 0.95]
    #threshold = [0.96, 0.927, 0.899, 0.739, 0.744]
    threshold_list = [threshold]
    date = '8_24'
    epochs = 50
    for i in range(1, 11):
        data_name = 'mnist'
        file_path = 'mnist_top_n{}'.format(i)
        dataset = Dataset(data_name, batch_size=batch_size)
        method = 'lsq'
        (train_set, train_labels), (test_set, test_labels) = dataset.load_data()
        model = F_VAE(data=data_name, shape=dataset.shape, latent_dim=dataset.latent_dims, model='cnn', num_cls=dataset.num_cls)
        classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls)

        checkpoint = tf.train.Checkpoint(sim_clr=model, clssifier=classifier)
        checkpoint.restore("./checkpoints/8_18/pre_train_mnist_super_loss/ckpt-84")

        start_train(epochs, n, threshold_list, method, model, classifier, dataset,
                    [train_set, train_labels],
                    [test_set, test_labels], sample_size, date, file_path)

