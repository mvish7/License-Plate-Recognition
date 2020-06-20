import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.filters import threshold_otsu
from PIL import Image
import torch.nn as nn


class EMNIST_Dataset(Dataset):

    def __init__(self, dataset_path):

        labels_set = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35
        ]

        data_dir = dataset_path
        images = []
        labels = []

        img_dirs = os.listdir(data_dir)
        for i, tmp_dir in enumerate(img_dirs):
            curr_dir_imgs = os.listdir(os.path.join(data_dir, tmp_dir))[0:25]

            for j, img in enumerate(curr_dir_imgs):
                img_path = os.path.join(data_dir, tmp_dir, curr_dir_imgs[j])
                img_data = Image.open(img_path).convert('L')
                img_data = np.array(img_data.resize((28, 28)))
                binary_img = img_data < threshold_otsu(img_data)

                # flattening to create 1d array for our models

                images.append(binary_img)
                labels.append(labels_set[i])

        self.images = np.array(images)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        sample = {'image': image, 'label': label}
        transformed_sample = self.transform(sample)

        return transformed_sample['image'], transformed_sample['label']

    def transform(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': torch.tensor(image, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.float32)}


class Net(nn.Module):
    def __init__(self, num_inputs=784, num_outputs=36, num_hidden1=1024, num_hidden2=1024, num_hidden3=512,
                 num_hidden4=256, num_hidden5=128, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_hidden3 = num_hidden3
        self.num_hidden4 = num_hidden4
        self.num_hidden5 = num_hidden5
        self.is_training = is_training

        self.linear1 = nn.Linear(self.num_inputs, self.num_hidden1)
        self.linear2 = nn.Linear(self.num_hidden1, self.num_hidden2)
        self.linear3 = nn.Linear(self.num_hidden2, self.num_hidden3)
        self.linear4 = nn.Linear(self.num_hidden3, self.num_hidden4)
        self.linear5 = nn.Linear(self.num_hidden4, self.num_hidden5)
        self.linear6 = nn.Linear(self.num_hidden5, self.num_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, X):
        X = X.reshape(-1, 784)

        H1 = self.relu(self.linear1(X))

        if self.is_training:
            H1 = self.dropout1(H1)

        H2 = self.relu(self.linear2(H1))

        if self.is_training:
            H2 = self.dropout1(H2)

        H3 = self.relu(self.linear3(H2))

        if self.is_training:
            H3 = self.dropout1(H3)

        H4 = self.relu(self.linear4(H3))

        if self.is_training:
            H4 = self.dropout2(H4)

        H5 = self.relu(self.linear5(H4))

        if self.is_training:
            H5 = self.dropout2(H5)

        pred = self.linear6(H5)

        return pred

    def infer(self, X):
        pred = self.forward(X)

        prob = nn.functional.softmax(pred, dim=1)

        return prob.topk(1)


def evaluate_accuracy(data_iter, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""

    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0]
    return acc_sum.item()/n


def train(net, train_iter, criterion, num_epochs, device, lr=None):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=4e-3)
    for epoch in range(num_epochs):
        n, start = 0, time.time()
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device=device, dtype=torch.long)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]

        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'\
            % (epoch + 1, train_l_sum/n, train_acc_sum/n, time.time() - start))


def main():

    emnist_dataset = EMNIST_Dataset('C:\\Users\\mvish\\PycharmProjects\\AutomaticLicensePlateRecognition\\data')
    emnist_dataloader = DataLoader(emnist_dataset, batch_size=128, shuffle=True)

    lr = 0.030
    epochs = 10

    model = Net()
    loss_fun = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model, emnist_dataloader, loss_fun, epochs, device, lr=lr)

    torch.save(model.state_dict(), 'C:\\Users\\mvish\\PycharmProjects\\AutomaticLicensePlateRecognition\\model\\torch\\torch_classifier.pth')


if __name__ == "__main__":
    main()
