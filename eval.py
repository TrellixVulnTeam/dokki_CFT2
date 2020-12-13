import sys

from dokki import VOCBrain, VOCDataset, load_dataset_from_pascal_voc_jar, show_image_and_boxes, show_image
import logging
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch import nn
import torch

logging.basicConfig(level=logging.INFO)

JAR_PATH="/home/gugaime/Documentos/Datasets/VOCtrainval_06-Nov-2007.tar"
CHECKPOINT_TAR = '/home/gugaime/Downloads/ssd/checkpoint_ssd300.pth.tar'

if __name__ == "__main__":
    brain = VOCBrain(JAR_PATH, CHECKPOINT_TAR)
    brain.load()
    image = brain.get_dataset_image(5)
    show_image(image)
    image, boxes_batch, labels_batch, scores_batch = brain.eval("/home/gugaime/Imagens/dog.jpeg")
    show_image_and_boxes(image, boxes_batch, labels_batch, scores_batch, brain.labels)
