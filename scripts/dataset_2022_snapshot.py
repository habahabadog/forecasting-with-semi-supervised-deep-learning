import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import random
from datetime import datetime, timedelta


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)

batch_size = 4
hig_mean = np.array([295.71266308, -1.35061465, -0.47545855, 24.42778042]).reshape(4, 1, 1)
hig_std = np.array([4.25024591, 2.50847865, 2.44562078, 67.33232656]).reshape(4, 1, 1)


class MyDataset(Dataset):
    def __init__(self, file_dir, file_list, mean, std):
        self.file_dir = file_dir
        self.file_list = file_list
        self.mean = mean
        self.std = std

        file_times = [datetime.strptime(f[:-4], "%Y%m%d%H") for f in self.file_list]
        self.stacked_list = [
            [self.file_list[i], self.file_list[i + 1], self.file_list[i + 2]]
            for i in range(len(file_times) - 2)
            if file_times[i + 1] == file_times[i] + timedelta(hours=1)
            and file_times[i + 2] == file_times[i] + timedelta(hours=2)
        ]

    def __getitem__(self, index):
        items_to_fetch = self.stacked_list[index]

        low_datas = []
        high_datas = []

        for file_name in items_to_fetch:
            data = np.load(os.path.join(self.file_dir, file_name)).astype(np.float32)

            low_data = (data[:, :-1:5, :-1:5] - self.mean) / self.std
            high_data = (data[:, :-1, :-1] - self.mean) / self.std

            low_datas.append(torch.from_numpy(low_data).float())
            high_datas.append(torch.from_numpy(high_data).float())

        stacked_low_datas = torch.stack(low_datas, dim=0).reshape(-1, *low_datas[0].shape[1:])
        stacked_high_datas = torch.stack(high_datas, dim=0).reshape(-1, *high_datas[0].shape[1:])

        return (
            stacked_low_datas,
            stacked_low_datas[4:8, :, :],
            stacked_high_datas[4:8, :, :],
            stacked_high_datas,
        )

    def __len__(self):
        return len(self.stacked_list)


# 路径配置
file_dir = '../art/artnpy'
common_files = sorted(os.listdir(file_dir))

# 文件列表划分
train_files = sorted([f for f in common_files if f < '20221000'])
new_files = sorted([f for f in common_files if f > '20221000'])

# 数据集划分
train_size = len(train_files) * 9 // 10
test_size = len(new_files)

train_dataset = MyDataset(file_dir=file_dir, file_list=train_files[:train_size], mean=hig_mean, std=hig_std)
test_dataset = MyDataset(file_dir=file_dir, file_list=train_files[train_size:], mean=hig_mean, std=hig_std)
new_dataset = MyDataset(file_dir=file_dir, file_list=new_files, mean=hig_mean, std=hig_std)

# DataLoader
train_load = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_load = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
new_load = DataLoader(new_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)