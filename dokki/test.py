from datasets import VOCDataset, load_dataset_from_dokki_jar, load_dataset_from_icdar_jar
import os
import logging
import drawer

os.environ["DATASET_TMP"]='C:/Users/gugaime/Documents/Datasets/output'
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset = load_dataset_from_icdar_jar('C:/Users/gugaime/Documents/Datasets/icdar.task1train.zip', "TRAIN")
    dataset = VOCDataset('C:/Users/gugaime/Documents/Datasets/output', "TRAIN")
    image,timage, boxes, labels, difficulties = dataset[2]
    logging.info("-Image:%s",timage.shape)
    logging.info("-Boxed:%s",boxes.shape)
    logging.info("-Labels:%s",labels.shape)
    drawer.show_image(image)

    label_map={}
    label_map[0] = 'x'
    label_map[1] = 'y'

    drawer.show_image_and_boxes(image, boxes, labels, difficulties, label_map)

