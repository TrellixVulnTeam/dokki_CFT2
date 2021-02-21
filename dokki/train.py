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

#os.environ["DATASET_TMP"]="/tmp/VOC/"
os.environ["DATASET_TMP"]="/Users/gugaime/Documents/Datasets/output"

logging.basicConfig(level=logging.INFO)

JAR_PATH="/Users/gugaime/Documents/Datasets/icdar.task1train.zip"

if __name__ == "__main__":
    #brain = brains.VOCBrain(JAR_PATH, "/home/gugaime/checkpoint_ssd300.pth.tar")
    brain = brains.VOCBrain(JAR_PATH)
    brain.load()
    brain.train()
