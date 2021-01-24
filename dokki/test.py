from datasets import VOCDataset, load_dataset_from_dokki_jar
import os
import logging
import drawer

os.environ["DATASET_TMP"]="/tmp/notafiscalpaulista"
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset = load_dataset_from_dokki_jar('/tmp/notafiscalpaulista.tar.xz', "TRAIN")
    dataset = VOCDataset("/tmp/notafiscalpaulista", "TRAIN")
    image,timage, boxes, labels, difficulties = dataset[0]
    logging.info("-Image:%s",timage.shape)
    logging.info("-Boxed:%s",boxes.shape)
    logging.info("-Labels:%s",labels.shape)
    drawer.show_image(image)

    label_map={}
    label_map[0] = 'x'
    label_map[1] = 'y'

    drawer.show_image_and_boxes(image, boxes, labels, difficulties, label_map)

