import sys

from datasets import VOCDataset, load_dataset_from_pascal_voc_jar
from drawer import show_image_and_boxes, show_image
import logging
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch import nn
import torch
import brains
import os

os.environ["DATASET_TMP"]="/tmp/VOC/"

logging.basicConfig(level=logging.INFO)

JAR_PATH="/home/gugaime/Documentos/Datasets/VOCtrainval_06-Nov-2007.tar"

if __name__ == "__main__":
    brain = brains.VOCBrain(JAR_PATH, "/home/gugaime/checkpoint_ssd300.pth.tar")
    brain.load()
    brain.train()
