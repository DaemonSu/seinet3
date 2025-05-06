import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------------------
#  1️⃣ 读取 IQ 数据文件
# ---------------------------
DATA_PATH = "G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/32ftbak"  # 文件夹路径，存放 16 个设备的 IQ 数据文件
SAMPLE_SIZE = 7000  # 每条数据包含 7000 个 IQ 采样点
STRIDE = 7000       # 步长（可调）
TRAIN_DEVICES = 10
VAL_DEVICES = 3
OPENSET_DEVICES = 3

SAVE_PATH = "F:/seidata/IQdata"  # 训练数据保存目录


def load_iq_data(file_path):
    """
    加载 IQ 数据，并确保数据格式正确
    """
    # path = "F:/seidata/IQdata/train2/device_05_0551.npy"
    iq_data =  np.fromfile(file_path, dtype=np.complex64)
    if np.iscomplexobj(iq_data):  # 检测是否是复数格式

        # print("前10个IQ数据:", iq_data[:10])
        print("I:", iq_data[:800].real)
        print("Q:", iq_data[:800].imag)

        image= torch.from_numpy(iq_data.imag)
 # 拆分 I/Q 组成 (N,2)
        q_abs = torch.abs(image)
        q_fft = torch.fft.fft(image).abs()
        # iq_data = np.vstack((iq_data.imag,q_abs,q_fft))
        iq_data = np.column_stack((iq_data.imag, q_abs, q_fft))



    return iq_data

# ---------------------------
#  2️⃣ 处理数据 & 划分数据集
# ---------------------------
class SEIDataset(Dataset):
    def __init__(self, iq_data, labels):
        self.iq_data = torch.tensor(iq_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.iq_data)

    def __getitem__(self, idx):
        return self.iq_data[idx], self.labels[idx]


def process_data():
    iq_samples = []  # 存放 IQ 数据
    labels = []  # 存放设备类别
    device_files = sorted(os.listdir(DATA_PATH))  # 读取 16 个设备文件

    # 确保设备数据排序
    assert len(device_files) == 16, "数据集应包含 16个设备文件"

    # 设备编号分配
    train_files = device_files[:TRAIN_DEVICES]
    val_files = device_files[TRAIN_DEVICES:TRAIN_DEVICES + VAL_DEVICES]
    openset_files = device_files[TRAIN_DEVICES + VAL_DEVICES:]

    def process_device(files, save_folder, label_offset=0):
        """
        处理多个设备的 IQ 数据：
        - 按 (200, 2) 进行切片
        - 每个数据片段存储为独立的 `.npy` 文件
        """
        save_path = os.path.join(SAVE_PATH, save_folder)
        os.makedirs(save_path, exist_ok=True)

        for device_idx, file in enumerate(files):
            iq_data = load_iq_data(os.path.join(DATA_PATH, file))
            total_points = iq_data.shape[0]  # IQ 点数
            device_id = device_idx + label_offset

            # # 计算可分割的样本数 N
            num_samples = (total_points - SAMPLE_SIZE) // STRIDE + 1

            for i in range(num_samples):
                start = i * STRIDE
                segment = iq_data[start: start + SAMPLE_SIZE]

                # 生成文件名：device_01_0001.npy
                filename = f"device_{device_id:02d}_{i:04d}.npy"
                np.save(os.path.join(save_path, filename), segment)

            print(f"✅ 设备 {device_id} 处理完成: {num_samples} 片段")

    # 划分训练、验证、开集测试数据
    process_device(train_files,"train", label_offset=0)
    process_device(val_files,"val", label_offset=TRAIN_DEVICES)
    process_device(openset_files,"openset", label_offset=TRAIN_DEVICES + VAL_DEVICES)

    # 创建 PyTorch Dataset
    # train_dataset = SEIDataset(train_x, train_y)
    # val_dataset = SEIDataset(val_x, val_y)
    # openset_dataset = SEIDataset(openset_x, openset_y)

    # return train_dataset, val_dataset, openset_dataset


# ---------------------------
#  3️⃣ 创建 DataLoader
# ---------------------------
def get_dataloaders(batch_size=32):
    train_dataset, val_dataset, openset_dataset = process_data()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    openset_loader = DataLoader(openset_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, openset_loader


# ---------------------------
#  4️⃣ 运行数据预处理
# ---------------------------
if __name__ == "__main__":
    process_data()
    # get_dataloaders()
    # print(f"训练集样本数: {len(train_loader.dataset)}")
    # print(f"验证集样本数: {len(val_loader.dataset)}")
    # print(f"开集测试样本数: {len(openset_loader.dataset)}")
