import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def show_image(image: Image):
    plt.imshow(image)
    plt.show()

def draw_rectangule(img, rectangle, label):
    draw = ImageDraw.Draw(img)
    draw.rectangle(rectangle, outline="green")
    label_position = (rectangle[0],rectangle[1])
    draw.text(label_position, label)

def show_image_and_boxes(image, boxes_batch, labels_batch, scores_batch, labels):
    for j, tbox in enumerate(boxes_batch):
        label_id = labels_batch[j].item()
        image_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        bboex= (tbox*image_dims).tolist()[0]
        score = str(scores_batch[j].item())
        draw_rectangule(image, bboex, labels[label_id-1]+"-"+score)
    plt.imshow(image)
    plt.show()