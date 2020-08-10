import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 81),
        )
        """
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 81),
        )
        """

    def forward(self, x):
        score = self.conv(x)
        return score


class Board(Dataset):

    def __init__(self):
        self.X = np.load('../generated_games/features-40k.npy').astype(np.float32)
        Y = np.load('../generated_games/labels-40k.npy').astype(np.int64)
        Y = np.argmax(Y, axis=1)
        self.Y = Y
        self.size = self.X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Train:

    def __init__(self, batch_size=64, learning_rate=0.0003, epochs=60):
        data = Board()
        self.epochs = epochs
        self.train_data = Subset(data, list(range(40000)))
        self.vali_data = Subset(data, list(range(40000, len(data))))
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.vali_loader = DataLoader(self.vali_data, batch_size=batch_size, shuffle=False)
        self.model = Model()
        self.model = nn.DataParallel(self.model).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        total_loss = torch.Tensor([0])
        score, y = None, None
        for x, y in self.train_loader:
            x = x.to(device)
            x = x.view(-1, 1, 9, 9)
            y = y.to(device)
            score = self.model(x)
            loss = self.loss_fn(score, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss
        #print(score[0])
        #print(y[0])
        total_loss /= len(self.train_data)
        return total_loss

    def validate(self):
        self.model.eval()
        vali_loss = torch.Tensor([0])
        total, correct = 0, 0
        score = 0
        for x, y in self.vali_loader:
            x = x.to(device)
            x = x.view(-1, 1, 9, 9)
            y = y.to(device)
            score = self.model(x)
            loss = self.loss_fn(score, y)
            vali_loss += loss
            _, predicted = torch.max(score.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        vali_loss /= len(self.vali_data)
        print("test acc: {}".format(100*correct/total))
        print(score[0])
        return vali_loss

    def run(self):
        for epoch in range(self.epochs):
            train_loss = self.train()
            vali_loss = self.validate()
            torch.save({"model_state_dict": self.model.module.state_dict()}, "go_model.pt")
            print("EPOCH {}: train loss: {:05f} vali loss: {:05f}".format(epoch, float(train_loss[0]), float(vali_loss[0])))


if __name__ == "__main__":
    train = Train()
    train.run()

