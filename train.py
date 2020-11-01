import logging
import torch
from datasets import ICDARDataset
import matplotlib.pyplot as plt
import time
from model import SSD300, MultiBoxLoss

data_folder="./dataset"
batch_size = 1
workers = 0
start_epoch = 0  # start at this epoch
epochs = 300  # number of epochs to run without early-stopping
print_freq = 10
label_map = {'background': 0, 'text': 1}
n_classes = len(label_map)
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay

def main():

    train_dataset = ICDARDataset(data_folder, split='train')
    #images, boxes, labels = train_dataset[0]
    #print(images.shape, boxes.shape, labels.shape)
    #plt.imshow(  images.permute(1, 2, 0)  )
    #plt.show()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    model = SSD300(n_classes=n_classes)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)

    biases = list()
    not_biases = list()
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(start_epoch, epochs):
        print("********** epoch {}".format(epoch))
        train_loss = train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()  # training mode enables dropout

    start = time.time()

    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):
        print(time.time() - start)

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()


        # Update model
        optimizer.step()

        print(loss.item(), images.size(0))
        print(time.time() - start)

        start = time.time()
        print("epoch {} i {}".format(epoch,i))

main()