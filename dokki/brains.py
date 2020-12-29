import sys
import time
import torch
from torch import nn
from datasets import VOCDataset, load_dataset_from_pascal_voc_jar

from model import SSD300, MultiBoxLoss, AverageMeter
from PIL import Image, ImageDraw

from drawer import show_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VOCBrain():

    PRINT_FREQ = 1

    def __init__(self, dataset_jar_path, chekpoint_tar_path=None, batch_size=4, workers=2):
        self.jar_path=dataset_jar_path
        self.chekpoint_tar_path=chekpoint_tar_path
        self.batch_size=batch_size
        self.workers=workers
        self.labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'background')
        self.n_classes = len(self.labels)
        self.dataset = None
        self.loader = None
        self.pre_trained_model = None
        self.biases = list()
        self.not_biases = list()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr = 1e-3  # learning rate
        self.decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
        self.decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
        self.momentum = 0.9  # momentum
        self.weight_decay = 5e-4
        self.start_epoch = 0
        self.epochs = 20

    def load(self):
        self.dataset = load_dataset_from_pascal_voc_jar(self.jar_path, "TRAIN")
        self.loader = self.__loader_from_dataset(self.dataset, self.batch_size, self.workers)
        if self.chekpoint_tar_path:
            self.start_epoch, self.model, self.optimizer  = self.__load_checkpoint(self.chekpoint_tar_path)
            print('\nLoaded checkpoint from epoch %d.\n' % self.start_epoch)
        else:
            self.model = SSD300(n_classes=self.n_classes)
            for param_name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        self.biases.append(param)
                    else:
                        self.not_biases.append(param)
            self.optimizer = torch.optim.SGD(params=[{'params': self.biases, 'lr': 2 * self.lr}, {'params': self.not_biases}],
                                    lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.model = self.model.to(device)
        self.criterion = MultiBoxLoss(priors_cxcy=self.model.priors_cxcy).to(device)


    def get_dataset_image(self, i):
        image,  _, _, _, _= self.dataset[i]
        return image

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            if epoch in self.decay_lr_at:
                self.__adjust_learning_rate(self.decay_lr_to)
            self.model.train()
            batch_time = AverageMeter()  # forward prop. + back prop. time
            data_time = AverageMeter()  # data loading time
            losses = AverageMeter()
            start = time.time()

            for i, (image, timages, tboxes, tlabels, _) in enumerate(self.loader):

                data_time.update(time.time() - start)

                timages = timages.to(device)  # (batch_size (N), 3, 300, 300)
                tboxes = [b.to(device) for b in tboxes]
                tlabels = [l.to(device) for l in tlabels]

                predicted_locs, predicted_scores = self.model(timages)
                loss = self.criterion(predicted_locs, predicted_scores, tboxes, tlabels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), timages.size(0))
                batch_time.update(time.time() - start)
                start = time.time()

                if i % VOCBrain.PRINT_FREQ==0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(self.loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))

                if i==2:
                    break

            self.__save_checkpoint(epoch, self.model, self.optimizer)

    def eval(self, image_path):
        with torch.no_grad():
            image = Image.open(image_path)
            t_image, _ = self.dataset.transform_image(image)
            mini_batch_image = t_image.unsqueeze(0)
            predicted_locs, predicted_scores = self.model(mini_batch_image)
            boxes_batch, labels_batch, scores_batch = self.model.detect_objects(predicted_locs, predicted_scores,
                                                                                    min_score=0.2, max_overlap=0.45,
                                                                                    top_k=200)
            return image, boxes_batch[0], labels_batch[0], scores_batch[0]

    def __save_checkpoint(self, epoch, model, optimizer):
        state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
        filename = 'checkpoint_ssd300.pth.tar'
        torch.save(state, filename)

    def __adjust_learning_rate(self, scale):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale
        print("DECAYING learning rate.\n The new LR is %f\n" % (self.optimizer.param_groups[1]['lr'],))

    def __load_checkpoint(self, checkpoint:str) -> nn.Module:
        sys.path.insert(0, './dokki')
        checkpoint = torch.load(checkpoint,  map_location=torch.device(device))
        return checkpoint['epoch'] + 1,checkpoint['model'],checkpoint['optimizer']

    def __loader_from_dataset(self, dataset: VOCDataset, batch: int, workers: int):
        return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=dataset.collate_fn, num_workers=workers, pin_memory=True)


