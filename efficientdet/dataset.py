import os
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import glob
import os
from PIL import Image

class EyeDataset(Dataset):
    def __init__(self, img_list, anno_txt_path, transform):
        self.transform = transform
        self.imgs = img_list
        self.image_dir = os.path.dirname(self.imgs[0])
        self.annos = {}
        self.idx_to_name = []

        for img_path in self.imgs:
            file_name = os.path.basename(img_path)
            self.annos[file_name] = []

        not_exist_count = 0
        with open(anno_txt_path, "r") as fp:
            for line in fp:
                line = line.strip()
                words = line.split()
                file_name = words[0]
                label = words[1:]

                if (file_name not in self.annos):
                    not_exist_count += 1
                    continue

                self.idx_to_name.append(file_name)
                # read out seq is: x1, y1, x2. y2. label
                for i in range(len(label) // 5):
                    x1 = int(label[i * 5])
                    y1 = int(label[i * 5 + 1])
                    x2 = int(label[i * 5 + 2])
                    y2 = int(label[i * 5 + 3])
                    category = int(label[i * 5 + 4])
                    self.annos[file_name].append([x1, y1, x2, y2, category])
                self.annos[file_name] = np.array(self.annos[file_name], dtype="float64")
        print(f"Not exist img num: {not_exist_count}")
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        file_name = self.idx_to_name[idx]
        img = cv2.imread(f"{self.image_dir}/{file_name}")  # order: h, w, c
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        annot = self.annos[file_name]
        sample = {'img': img, 'annot': annot}
        sample = self.transform(sample)
        return sample


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class randomScale(object):
    def __call__(self, sample, scale_x=0.5):
        if (np.random.rand() < scale_x):
            image, annots = sample['img'], sample['annot']
            h, w, c = image.shape
            scale = random.uniform(0.8,1.2)
            image = cv2.resize(image, (int(w * scale), h))  # order of resize is width, height
            annots[:, 0] *= scale
            annots[:, 2] *= scale
            sample = {'img': image, 'annot': annots}
        return sample


class randomBlur(object):
    def __call__(self, sample, blur_x=0.5):
        if (np.random.rand() < blur_x):
            image, annots = sample['img'], sample['annot']
            image = cv2.blur(image,(5,5))
            sample = {'img': image, 'annot': annots}
        return sample


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
