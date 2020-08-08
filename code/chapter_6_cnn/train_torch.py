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
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 81),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        score = self.conv(x)
        return score


class Board(Dataset):

    def __init__(self):
        self.X = np.load('../generated_games/features-40k.npy').astype(np.float32)
        self.Y = np.load('../generated_games/features-40k.npy').astype(np.float32)
        self.size = self.X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class Train:

    def __init__(self, batch_size=16, learning_rate=0.0005, epochs=100):
        data = Board()
        self.epochs = epochs
        train_data = Subset(data, list(range(40000)))
        test_data = Subset(data, list(range(40000, len(data))))
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        self.model = Model().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        total_loss = torch.Tensor([0])
        for x, y in self.train_loader:
            x = x.to(device)
            y = y.to(device)
            score = self.model(x)
            loss = self.loss_fn(score, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss
        total_loss /= len(self.train_loader)
        return total_loss

    def validate(self):
        self.model.eval()
        vali_loss = torch.Tensor([0])
        for x, y in self.vali_loader:
            x = x.to(device)
            y = y.to(device)
            score = self.model(x)
            loss = self.loss_fn(score, y)
            vali_loss += loss
        vali_loss /= len(self.vali_loader)
        return vali_loss

    def run(self):
        for epoch in range(self.epochs):
            train_loss = self.train()
            vali_loss = self.validate()
            print(epoch, train_loss, vali_loss)


if __name__ == "__main__":
    train = Train()
    train.run()


