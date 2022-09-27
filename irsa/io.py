import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class InfraredDataset(Dataset):
    '''
    categories is the list of different alphabets (folders)
    root_dir is the root directory leading to the alphabet files, could be /images_background or /images_evaluation
    set_size is the size of the train set and the validation set combined
    transform is any image transformations
    '''
    def __init__(self, categories, root_dir, set_size, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.transform = transform
        self.set_size = set_size

    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):
        spec1 = None
        spec2 = None
        label = None

        if idx % 2 == 0: # select the same label for both images
            category = np.random.choice(categories)
            character = np.random.choice(category[1])
            specDir = root_dir + category[0] + '/' + character
            spec1Name = np.random.choice(os.listdir(specDir))
            spec2Name = np.random.choice(os.listdir(specDir))
            spec1 = np.load(specDir + '/' + spec1Name)
            spec2 = np.load(specDir + '/' + spec2Name)
            label = 1.0

        else: # select a different character for both images
            category1, category2 = np.random.choice(categories), np.random.choice(categories)
            category1, category2 = np.random.choice(categories), np.random.choice(categories)
            character1, character2 = np.random.choice(category1[1]), np.random.choice(category2[1])
            specDir1, specDir2 = root_dir + category1[0] + '/' + character1, root_dir + category2[0] + '/' + character2
            spec1Name = np.random.choice(os.listdir(specDir1))
            spec2Name = np.random.choice(os.listdir(specDir2))
            while spec1Name == spec2Name:
                spec2Name = np.random.choice(os.listdir(specDir2))
            label = 0.0
            spec1 = np.load(specDir1 + '/' + spec1Name)
            spec2 = np.load(specDir2 + '/' + spec2Name)

        if self.transform:
            spec1 = self.transform(spec1)
            spec2 = self.transform(spec2)

        return spec1, spec2, torch.from_numpy(np.array([label], dtype=np.float32))
