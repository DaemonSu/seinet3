import os
import shutil
import random


def move_random_files(source_folder, destination_folder, num_files_to_move):
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 获取源文件夹中的所有文件
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # 确保要移动的文件数不超过可用文件数
    num_files_to_move = min(num_files_to_move, len(all_files))

    # 随机选择指定数量的文件
    selected_files = random.sample(all_files, num_files_to_move)

    # 移动文件
    for file in selected_files:
        src_path = os.path.join(source_folder, file)
        dest_path = os.path.join(destination_folder, file)
        shutil.move(src_path, dest_path)
        print(f"Moved: {file} -> {destination_folder}")


# 示例用法
source_folder = "F:/seidata/26ft-exp/openset"  # 源文件夹路径
destination_folder = "F:/seidata/26ft-exp/val2"  # 目标文件夹路径
num_files_to_move = 2400  # 需要移动的文件数量

move_random_files(source_folder, destination_folder, num_files_to_move)
