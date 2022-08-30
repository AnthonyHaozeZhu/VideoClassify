# -*- coding: UTF-8 -*-
"""
@Project ：Classified 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/8/24 18:19
"""

import argparse
import \
    os.path

import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from data import *
from model import *


def train(epoch):
    train_tqdm = tqdm(train_data, desc="Training Epoch " + str(epoch))
    for index, (data, label) in enumerate(train_tqdm):
        optimizer.zero_grad()
        output = model(data.to(opt.device))
        loss = criterion(output, label.to(opt.device))
        loss.backward()
        optimizer.step()
        train_tqdm.set_postfix({"loss": "%.3g" % loss.item()})


def validate(epoch):
    model.eval()
    val_loss, correct = 0, 0
    test_tqdm = tqdm(test_data, desc="Validating Epoch " + str(epoch))
    for index, (data, label) in enumerate(test_tqdm):
        if data.shape[0] != label.shape[0]:
            continue
        output = model(data.to(opt.device))
        val_loss += criterion(output, label.to(opt.device)).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum()
    val_loss /= len(test_data)
    acc = 100. * correct.to(torch.float32) / len(test_data.dataset)
    logger.info("***** Epoch{}, Eval results *****".format(epoch))
    logger.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_data.dataset), acc))


def main():
    for epoch in range(opt.epochs):
        train(epoch)
        validate(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='./data', type=str)
    parser.add_argument("--sub_folder", default='hmdb51_org', type=str)
    parser.add_argument("--sub_folder_jpg", default='hmdb51_org_jpg')
    parser.add_argument("--pre_load", default=True, type=bool)
    parser.add_argument("--n_frames", default=16, type=int, help="The number of frames per video")
    parser.add_argument("--num_class", default=52, type=int, help="The number of the classes")
    parser.add_argument("--frame_height", default=112, type=int)
    parser.add_argument("--frame_width", default=112, type=int)
    parser.add_argument("--device", default='mps', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--hidden_size", default=100, type=int, help="The hidden size of Lstm")
    parser.add_argument("--num_layers", default=1, type=int, help="The number of the hidden states of Lstm")
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--logdir", default='./log', type=str)
    parser.add_argument("--epochs", default=100, type=int)

    opt = parser.parse_args()

    transformer = transforms.Compose([
        transforms.Resize((opt.frame_height, opt.frame_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
    ])

    frame_dir = os.path.join(opt.data_path, opt.sub_folder_jpg)
    movie_dir = os.path.join(opt.data_path, opt.sub_folder)

    if not opt.pre_load:
        if not exit(os.path.join(opt.data_path, opt.sub_folder_jpg)):
            os.mkdir(os.path.join(opt.data_path, opt.sub_folder_jpg))
        pre_load(movie_dir, frame_dir, opt.n_frames)

    train_loader = FramesLoader(frame_dir, transform=transformer, opt=opt)
    test_loader = FramesLoader(frame_dir, transform=transformer, opt=opt, is_train=False)

    train_data = DataLoader(dataset=train_loader, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_data = DataLoader(dataset=test_loader, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    model = Resnet18Rnn(opt).to(opt.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)
    logger = init_logger(opt)

    main()






