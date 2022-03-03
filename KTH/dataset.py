#
# boostcamp AI Tech
# Educational Mask Dataset
#

import os
import random
from enum import Enum
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2

    @classmethod
    def from_str(cls, value: str) -> int:
        if value == 'mask1' or value == 'mask2' or value == 'mask3' or value == 'mask4' or value == 'mask5':
            return cls.MASK
        elif value == 'incorrect_mask':
            return cls.INCORRECT
        elif value == 'normal':
            return cls.NORMAL
        else:
            raise ValueError(f"Mask value is {value}, which is errorneous")


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        if value == 'male':
            return cls.MALE
        elif value == 'female':
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', but is {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, but is {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class ProfileClassEqualSplitTrainMaskDataset(Dataset):
    def __init__(self, data_dir: str = '/',
                 mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),
                 transform = None, val_ratio: float = 0.2, classes: int = 18) -> None:
        super().__init__()

        self.image_paths = []
        self.image_labels = []
        self.mean = mean
        self.std = std
        self.transform = transform
        self.indices = {
            'train': [],
            'val': []
        }

        self.setup(os.path.join(data_dir, 'train/images'), val_ratio, classes)
        self.calc_statistics()

    @staticmethod
    def split_profile(profiles_len: int, val_ratio: float) -> Dict:
        assert profiles_len % 5 == 0, ValueError(f"Each profile should have five mask wearing images")
        profiles_len = profiles_len // 5

        val_len = int(profiles_len * val_ratio)

        val_indices = set(random.sample(range(profiles_len), k=val_len))
        train_indices = set(range(profiles_len)) - val_indices

        return {
            'train': train_indices,
            'val': val_indices
        }

    def setup(self, root: str, val_ratio: float, classes: int) -> None:
        for _ in range(classes):
            self.image_paths.append([])

        profiles = os.listdir(root)
        for profile in profiles:
            if profile.startswith('.'):
                continue

            _, gender, _, age = profile.split('_')
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)

            img_folder = os.path.join(root, profile)
            for file_name_ext in os.listdir(img_folder):
                file_name, _ = os.path.splitext(file_name_ext)
                if file_name not in ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']:
                    continue

                mask_label = MaskLabels.from_str(file_name)
                label = self.encode_multi_class(mask_label, gender_label, age_label)

                img_path = os.path.join(root, profile, file_name_ext)
                self.image_paths[label].append(img_path)

        # Number of image paths of images with class label [None, 0, 0 ~ 1, 0 ~ 2, ..., 0 ~ 16]
        label_len_sum = [0]
        for label in range(1, classes):
            label_len_sum.append(label_len_sum[label - 1] + len(self.image_paths[label - 1]))

        for label in range(classes // 3):
            split_profiles = self.split_profile(len(self.image_paths[label]), val_ratio)
            for phase, profile_indices in split_profiles.items():
                for profile_index in profile_indices:
                    for i in range(5):
                        self.indices[phase].append(label_len_sum[label] + profile_index * 5 + i) # Label 0 ~ 5
                    self.indices[phase].append(label_len_sum[label + 6] + profile_index) # Label 6 ~ 11
                    self.indices[phase].append(label_len_sum[label + 12] + profile_index) # Label 12 ~ 17

        for label in range(classes):
            self.image_labels.extend([label] * len(self.image_paths[label]))

        self.image_paths = [path for path_label in self.image_paths for path in path_label] # Flatten

    # For baseline compatibility
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    # For baseline compatibility
    def set_transform(self, transform) -> None:
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        image = self.read_image(index)
        label = self.get_label(index)

        if self.transform:
            return self.transform(image), label
        else:
            return image, label

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_label(self, index: int) -> int:
        return self.image_labels[index]

    def read_image(self, index: int) -> Image.Image:
        return Image.open(self.image_paths[index])

    @staticmethod
    def encode_multi_class(mask_label: int, gender_label: int, age_label: int) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    # For baseline compatibility
    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    # For baseline compatibility
    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for _, indices in self.indices.items()]


class EvalMaskDataset(Dataset):
    def __init__(self, data_dir: str = '/', transform = None) -> None:
        super().__init__()

        img_list = pd.read_csv(os.path.join(data_dir, 'eval/info.csv'))
        self.image_paths = [os.path.join(data_dir, 'eval/images', img_id) for img_id in img_list.ImageID]
        self.transform = transform

    # For baseline compatibility
    def set_transform(self, transform) -> None:
        self.transform = transform

    def __getitem__(self, index: int) -> Image.Image:
        image = self.read_image(index)

        if self.transform:
            return self.transform(image)
        else:
            return image

    def __len__(self) -> int:
        return len(self.image_paths)

    def read_image(self, index: int) -> Image.Image:
        return Image.open(self.image_paths[index])

class TestDataset(Dataset):
    '''
    For baseline compatibility
    '''
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)) -> None:
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index: int) -> Image.Image:
        image = self.read_image(index)

        if self.transform:
            return self.transform(image)
        else:
            return image

    def __len__(self) -> int:
        return len(self.img_paths)

    def read_image(self, index: int) -> Image.Image:
        return Image.open(self.img_paths[index])