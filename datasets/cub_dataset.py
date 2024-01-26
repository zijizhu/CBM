import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class CUBDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str='train', transforms=None) -> None:
        super().__init__()
        if dataset_dir.endswith('CUB_200_2011'):
            self.dataset_dir = dataset_dir
        else:
            self.dataset_dir = os.path.join(dataset_dir, 'CUB_200_2011')

        with open(os.path.join(self.dataset_dir, 'classes.txt')) as fp:
            lines = fp.read().split('\n')
            self.id2label = [l.split(' ')[1] for l in lines if len(l) > 0]  # ['001.Black_footed_Albatross', '002.Laysan_Albatross', ...]
        
        filename_df = pd.read_csv(os.path.join(self.dataset_dir, 'images.txt'),
                                  delimiter=' ',index_col=0, names=['filename'])
        split_df = pd.read_csv(os.path.join(self.dataset_dir, 'train_test_split.txt'),
                             delimiter=' ', index_col=0, names=['is_train'])
        label_df = pd.read_csv(os.path.join(self.dataset_dir, 'image_class_labels.txt'),
                             delimiter=' ', index_col=0, names=['label'])

        joined = filename_df.join(split_df).join(label_df)
        joined['label'] = joined['label'] - 1
        if split == 'train':
            split_idxs = joined['is_train'] == 1
        else:
            split_idxs = joined['is_train'] == 0
        self.ann = joined[split_idxs].drop(columns=['is_train']).reset_index(drop=True)

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = T.ToTensor()

    @property
    def num_lables(self):
        return len(self.id2labels)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        fn, label = self.ann.iloc[idx]
        img = Image.open(os.path.join(self.dataset_dir, 'images', fn))
        img = self.transforms(img)
        return img, torch.tensor(label)
