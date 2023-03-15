import os
import numpy as np
import scipy.io as scio
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from scipy.signal import hilbert


class PicmusPaths(Dataset):
    def __init__(self, root, paths, labels=None):
        super().__init__()

        self.root = root
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

    def __len__(self):
        return self._length

    def preprocess(self, path):
        rf_path = os.path.join(self.root, 'rf_data', path)
        us_path = os.path.join(self.root, 'us_image', path)

        matdata = scio.loadmat(us_path)
        us_image = matdata['us_image']
        matdata = scio.loadmat(rf_path)
        rf_data = matdata['rf_data']

        # ===================US Image=================

        dynamic_range = 60
        vrange = [-dynamic_range, 0]
        env = us_image
        im = 20 * np.log10(env/env.max())
        im = np.clip(im, vrange[0], vrange[1])

        def normImg(x): return 255. * (x-x.min()) / (x.max()-x.min())
        im = np.uint8(normImg(im))

        im = (im/127.5 - 1.0).astype(np.float32)

        example = {}
        example['us_image'] = im

        # ===================RF DATA=================
        data_len = 1024

        rf_env = np.abs(hilbert(rf_data, axis=0))
        D = int(np.floor(rf_env.shape[0] / data_len))
        rf_env = rf_env[slice(0, data_len * D, D), :]

        env = rf_env
        dB_Range = 50
        env = env - np.min(env)
        env = env / np.max(env)
        env = env + 0.00001
        log_env = 20 * np.log10(env)
        log_env = 255/dB_Range*(log_env+dB_Range)
        [N, M] = log_env.shape
        D = int(np.floor(N/1024))
        # env_disp = 255 * log_env[1:N:D, :] / np.max(log_env)
        env_disp = 255 * log_env / np.max(log_env)
        env_disp = env_disp.astype(np.uint8)
        img = env_disp
        img = np.rot90(img, 1)

        img = (img/127.5 - 1.0).astype(np.float32)

        example['rf_data'] = img

        return example

    def __getitem__(self, index):

        example = self.preprocess(self.labels["file_path_"][index])

        return example



class PicmusTrainDataset(Dataset):
    def __init__(self, root, train_list_file, us_transforms=None, rf_transforms=None):
        super().__init__()

        with open(train_list_file, "r") as f:
            paths = f.read().splitlines()

        self.data = PicmusPaths(root=root, paths=paths)

        if us_transforms:
            self.us_transforms = transforms.Compose(us_transforms)
        if rf_transforms:
            self.rf_transforms = transforms.Compose(rf_transforms)

        self._length = len(paths)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        example["us_image"] = self.us_transforms(self.data[index]["us_image"])
        example["rf_data"] = self.rf_transforms(self.data[index]["rf_data"])

        return example
    

class PicmusValDataset(Dataset):
    def __init__(self, root, test_list_file, us_transforms=None, rf_transforms=None):
        super().__init__()

        with open(test_list_file, "r") as f:
            paths = f.read().splitlines()

        self.data = PicmusPaths(root=root, paths=paths)

        if us_transforms:
            self.us_transforms = transforms.Compose(us_transforms)
        if rf_transforms:
            self.rf_transforms = transforms.Compose(rf_transforms)

        self._length = len(paths)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        example["us_image"] = self.us_transforms(self.data[index]["us_image"])
        example["rf_data"] = self.rf_transforms(self.data[index]["rf_data"])

        return example
