import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def clean_label(true_labels):
    true_labels = np.array(true_labels)
    if np.min(true_labels) > 0:
        true_labels -= np.min(true_labels)
    return true_labels

def get_labels(dataset):
    if dataset == 'cub':
        with open("./data/CUB_200_2011/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        true_labels = clean_label(true_labels)
        train_test_split = pd.read_csv(os.path.join('./data/', 'CUB_200_2011', 'train_test_split.txt'),
                                    sep=' ', names=['img_id', 'is_training_img'])
        train_test_split = train_test_split['is_training_img'].values
        train_indices = np.where(train_test_split == 1)
        test_indices = np.where(train_test_split == 0)
        train_labels, test_labels = true_labels[train_indices], true_labels[test_indices]

        return train_labels, test_labels