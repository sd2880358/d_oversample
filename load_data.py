import os
import numpy as np
import pandas as pd

class CelebA():
    '''Wraps the celebA dataset, allowing an easy way to:
         - Select the features of interest,
         - Split the dataset into 'training', 'test' or 'validation' partition.
    '''

    def __init__(self, main_folder='../CelebA/', selected_features=None, drop_features=[]):
        self.main_folder = main_folder
        self.images_folder = os.path.join(main_folder, 'img_align_celeba/img_align_celeba/')
        self.attributes_path = os.path.join(main_folder, 'list_attr_celeba.csv')
        self.partition_path = os.path.join(main_folder, 'list_eval_partition.csv')
        self.selected_features = selected_features
        self.features_name = []
        self.__prepare(drop_features)

    def __prepare(self, drop_features):
        '''do some preprocessing before using the data: e.g. feature selection'''
        # attributes:
        if self.selected_features is None:
            self.attributes = pd.read_csv(self.attributes_path)
            self.num_features = 40
        else:
            self.num_features = len(self.selected_features)
            self.selected_features = self.selected_features.copy()
            self.selected_features.append('image_id')
            self.attributes = pd.read_csv(self.attributes_path)[self.selected_features]

        for feature in drop_features:
            if feature in self.attributes:
                self.attributes = self.attributes.drop(feature, axis=1)
                self.num_features -= 1

        self.attributes.replace(to_replace=-1, value=0, inplace=True)

        self.features_name = list(self.attributes.columns)[:-1]

        # load ideal partitioning:
        self.partition = pd.read_csv(self.partition_path)

    def split(self, name='training', drop_zero=False):
        '''Returns the ['training', 'validation', 'test'] split of the dataset'''
        # select partition split:
        if name is 'training':
            to_drop = self.partition.where(lambda x: x != 0).dropna()
        elif name is 'validation':
            to_drop = self.partition.where(lambda x: x != 1).dropna()
        elif name is 'test':  # test
            to_drop = self.partition.where(lambda x: x != 2).dropna()
        else:
            raise ValueError('CelebA.split() => `name` must be one of [training, validation, test]')

        partition = self.partition.drop(index=to_drop.index)

        # join attributes with selected partition:
        joint = partition.join(self.attributes.drop('image_id', axis=1), how='inner').drop('partition', axis=1)

        if drop_zero is True:
            # select rows with all zeros values
            return joint.loc[(joint[self.features_name] == 1).any(axis=1)]
        elif 0 <= drop_zero <= 1:
            zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
            zero = zero.sample(frac=drop_zero)
            return joint.drop(index=zero.index)

        return joint

def load_celeba(path):
    data = np.load(os.path.join(path, "data.npy"))
    data = data.astype(float)
    data = data / 255.0 #0~1
    return data.astype('float32')

def split(path):

    celebA = CelebA(
        main_folder = path,
        drop_features=[
        'Attractive',
        'Pale_Skin',
        'Blurry',
    ])

    dataset = load_celeba(path)
    features = pd.read_csv(os.path.join(path, "label_set.csv"), index_col='index')

    train_set = celebA.split()
    train_set = train_set[train_set.index.isin(features.index)]

    test_set = celebA.split('test')
    test_set = test_set[test_set.index.isin(features.index)]
    labels = features['hair_style']

    train_labels = labels[train_set.index]
    test_labels = labels[test_set.index]
    return (dataset[train_set.index], train_labels.to_numpy()), (dataset[test_set.index], test_labels.to_numpy())



if __name__ == "__main__":
    test = load_celeba("../CelebA/")
    print(len(test))