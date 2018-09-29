import os
import argparse
import json
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import models
from sklearn.metrics import accuracy_score, confusion_matrix


def setup_logger(level=logging.DEBUG, filename=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def train(config):
    logger = logging.getLogger(__name__)

    logger.info("%s", config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = np.loadtxt(config.dataset_path, delimiter=',')
    np.random.shuffle(dataset[:config.train_test_split])
    np.random.shuffle(dataset[config.train_test_split:])

    trainset = models.DataSet(dataset[:config.train_test_split], config.tensor_view)
    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=0)

    testset = models.DataSet(dataset[config.train_test_split:], config.tensor_view)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=0)

    net = models.DenseNet(device=device, in_channels=config.in_channels, number_layers=config.layers, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)

    cel = torch.nn.CrossEntropyLoss()
    sgd = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    pkl_path = os.path.join(config.path, "{}.pkl".format(config.path))
    if os.path.exists(pkl_path):
        state_dict = torch.load(pkl_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file %s.", config.pkl_path)
        except RuntimeError:
            logger.error("Loading state from file %s failed.", config.pkl_path)

    for epoch in range(config.epoch_number):
        logger.info("Epoch number: %d", epoch + 1)

        logger.info("Trainset size: %d", len(trainset))

        # train
        running_loss = 0.0

        for i, (feature, label) in enumerate(trainloader):
            feature, label = feature.to(net.device), label.to(net.device)
            sgd.zero_grad()
            feature = net(feature)
            loss = cel(feature, label)
            loss.backward()
            sgd.step()

            running_loss += loss.item()

            logger.debug("[%d, %d] %7.4f", epoch + 1, i + 1, loss.item())

        torch.save(net.state_dict(), pkl_path)

        # test
        if (epoch + 1) % config.test_frequency == 0:
            logger.info("Testset size: %d", len(testset))

            labels_true = []
            labels_predicted = []

            for j, (feature, label) in enumerate(testloader):
                feature, label = feature.to(net.device), label.item()
                _, predicted_label = net(feature).max(dim=1)

                labels_true.append(label)
                labels_predicted.append(predicted_label)

                logger.debug("%5d: %d, %d", j + 1, label, predicted_label)

            cm = confusion_matrix(labels_true, labels_predicted, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

            logger.info("Accuracy: %7.4f", accuracy_score(labels_true, labels_predicted))
            logger.info("Confusion Matrix: \n%s\n", cm)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, help="Config file path.", required=True)
    parser.add_argument('-e', '--epoch', type=int, help="Train epoch number.", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise RuntimeError("Config path not found!")

    with open("{}/config.json".format(args.config)) as config_file:
        config = models.Config(**json.load(config_file))

    if args.epoch:
        config.epoch_number = args.epoch

    setup_logger(level=logging.DEBUG, filename=os.path.join(config.path, "{}.log".format(config.path)))
    train(config)


if __name__ == '__main__':
    main()
