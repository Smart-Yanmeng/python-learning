import os
import shutil
import random

# 保证随机可复现
random.seed(0)


def mk_dir(file_path):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        shutil.rmtree(file_path)
    os.makedirs(file_path)


def split_data(file_path, new_file_path, train_rate, val_rate, test_rate):
    class_names = []

    for cla in os.listdir(file_path):
        class_names.append(cla)

    for cla in class_names:
        mk_dir(new_file_path + '/' + 'train' + '/' + cla)
        mk_dir(new_file_path + '/' + 'val' + '/' + cla)
        mk_dir(new_file_path + '/' + 'test' + '/' + cla)

    for cla in class_names:
        eachclass_image = []
        for image in os.listdir(os.path.join(file_path, cla)):
            eachclass_image.append(image)
        total = len(eachclass_image)
        random.shuffle(eachclass_image)
        train_images = eachclass_image[0:int(train_rate * total)]
        val_images = eachclass_image[int(train_rate * total):int((train_rate + val_rate) * total)]
        test_images = eachclass_image[int((train_rate + val_rate) * total):]

        for image in train_images:
            old_path = file_path + '/' + cla + '/' + image
            new_path = new_file_path + '/' + 'train' + '/' + cla + '/' + image
            shutil.copy(old_path, new_path)

        for image in val_images:
            old_path = file_path + '/' + cla + '/' + image
            new_path = new_file_path + '/' + 'val' + '/' + cla + '/' + image
            shutil.copy(old_path, new_path)

        for image in test_images:
            old_path = file_path + '/' + cla + '/' + image
            new_path = new_file_path + '/' + 'test' + '/' + cla + '/' + image
            shutil.copy(old_path, new_path)


if __name__ == '__main__':
    file_path = f"./data"
    new_file_path = f"./dataset"
    split_data(file_path, new_file_path, train_rate=0.6, val_rate=0.1, test_rate=0.3)
