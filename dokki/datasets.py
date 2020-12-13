import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import logging
from typing import Tuple
from transformes import VOCTransform
from databuilders import VOCJsonBuilder


def load_dataset_from_pascal_voc_jar(path,split):
    output_tmp=os.environ["DATASET_TMP"]
    VOCJsonBuilder(path, output_tmp).build()
    json_path = output_tmp
    return VOCDataset(json_path, split)

class VOCDataset(Dataset):

    def __init__(self, data_folder, split, keep_difficulties=False):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.data_folder = data_folder
        self.keep_difficulties = keep_difficulties
        self.tranform = VOCTransform()

        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images_paths = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images_paths) == len(self.objects)
        logging.info("Total de %d imagens carregadas",len(self.images_paths))

    def __getitem__(self, i):
        image = Image.open(self.images_paths[i], mode="r").convert("RGB")
        tboxes, tlabels, tdifficulties = self.__get_boxes_labels_and_difficulties(i)
        image, timage,tboxes, tlabels, tdifficulties = self.tranform.transform(image, tboxes, tlabels, tdifficulties)
        return image, timage,tboxes, tlabels, tdifficulties

    def __get_boxes_labels_and_difficulties(self, i) ->  Tuple[torch.FloatTensor,torch.LongTensor, torch.ByteTensor]:
        objects = self.objects[i]
        tboxes = torch.FloatTensor(objects["boxes"])
        tlabels = torch.LongTensor(objects["labels"])
        tdifficulties = torch.BoolTensor(objects["difficulties"])
        if not self.keep_difficulties:
            keep_only_easy = ~tdifficulties
            tboxes = tboxes[keep_only_easy]
            tlabels = tlabels[keep_only_easy]
            tdifficulties = tdifficulties[keep_only_easy]
        assert tboxes.shape[0] == tlabels.shape[0] and tlabels.shape[0]== tdifficulties.shape[0]
        return tboxes, tlabels, tdifficulties

    def __len__(self):
        return len(self.images_paths)

    def transform_image(self, image):
        return self.tranform.tensor_from_image(image)

    def collate_fn(self, batch):
        images = list()
        timages = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            timages.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            difficulties.append(b[4])

        timages = torch.stack(timages, dim=0)

        return images,timages, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    load_dataset_from_pascal_voc_jar('/home/gugaime/Documentos/Datasets/VOCtrainval_06-Nov-2007.tar', 2007)
    # dataset = VOCDataset("/tmp/VOC", "TRAIN")
    # image,timage, boxes, labels, difficulties = dataset[0]
    # logging.info("-Image:%s",timage.shape)
    # logging.info("-Boxed:%s",boxes.shape)
    # logging.info("-Labels:%s",labels.shape)
    # images, timage, boxes, labels, difficulties = dataset[0]
    # logging.info("-Trader:%s",timage.shape)

