import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


def MnistLabel(class_num):
    raw_dataset = datasets.FashionMNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    class_tot = [0] * 10
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 10 * class_num:
                break
    return TensorDataset(
        torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels))
    )


def MnistUnlabel():
    raw_dataset = datasets.FashionMNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    return raw_dataset


def MnistTest():
    return datasets.FashionMNIST(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )


if __name__ == "__main__":
    print(dir(MnistTest()))