'''自己加的测试'''

from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
# from data import StandardDataset
# from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
# from data.concat_dataset import ConcatDataset
# from data.JigsawLoader import JigsawNewDataset, JigsawTestNewDataset
from PIL import Image
from random import sample, random


dataset_list = ['art_painting', 'cartoon', 'photo']
for dname in dataset_list:
    name_train, labels_train = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_train_kfold.txt' % dname))
    print(type(name_train))

def get_image(names, index):
    framename = 'PACS' + '/' + names[index]
    img = Image.open(framename).convert('RGB')
    print(img)
    print(img.shape)
    os.system("pause")

for i in range(len(name_train)):
    get_image(name_train, i)