import sys

from datasets import VOCDataset, load_dataset_from_pascal_voc_jar

from model import SSD300, MultiBoxLoss, AverageMeter
from PIL import Image, ImageDraw

from drawer import show_image, show_image_and_boxes

import brains
import os

import logging
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch import nn
import torch

os.environ["DATASET_TMP"]="/tmp/VOC/"

logging.basicConfig(level=logging.INFO)

JAR_PATH="/home/gugaime/Documentos/Datasets/VOCtrainval_06-Nov-2007.tar"
CHECKPOINT_TAR = './checkpoint_ssd300.pth.tar'

if __name__ == "__main__":
    brain = brains.VOCBrain(JAR_PATH, CHECKPOINT_TAR)
    brain.load()
    image = brain.get_dataset_image(5)
    show_image(image)
    image, boxes_batch, labels_batch, scores_batch = brain.eval_image(image)
    show_image_and_boxes(image, boxes_batch, labels_batch, scores_batch, brain.labels)

    image, boxes_batch, labels_batch, scores_batch = brain.eval("/home/gugaime/Imagens/person_dog.jpeg")
    show_image_and_boxes(image, boxes_batch, labels_batch, scores_batch, brain.labels)
