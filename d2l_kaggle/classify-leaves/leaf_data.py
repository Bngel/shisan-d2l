import os.path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.nn.functional import one_hot


class LeafDataSet(Dataset):

    def __init__(self, image_dir, image_csv_path, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.img_data = pd.read_csv(image_csv_path)
        label_arr = np.array(self.img_data.iloc[:, 1])
        label_dict = { key: value for (value, key) in enumerate(set(label_arr)) }
        int_labels = [label_dict[label] for label in label_arr]
        self.image_labels = one_hot(torch.tensor(int_labels), num_classes=len(label_dict))
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        image_name = self.img_data.iloc[index, 0]
        image = Image.open(os.path.join(self.image_dir, image_name))
        label = torch.tensor(self.image_labels[index], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


def leaf_data_loader(batch_size, resize=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size=resize),
        transforms.ToTensor()
    ])
    train_dataset = LeafDataSet('./data', './data/train.csv', transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
