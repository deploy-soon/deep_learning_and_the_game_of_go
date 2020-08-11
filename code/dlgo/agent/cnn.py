"""Policy gradient learning."""
import numpy as np
import torch
from torch import nn

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.gotypes import Point
from dlgo import goboard

device = 'cuda:0'

__all__ = [
    'CNNBot',
]

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Flatten(),
        )

    def forward(self, x):
        score = self.conv(x)
        return score


class CNNBot(Agent):
    """An agent that uses a deep policy network to select moves."""
    def __init__(self):
        Agent.__init__(self)
        self.model = nn.DataParallel(Model()).to(device)
        checkpoint = torch.load("go_model.pt")
        self.model.module.load_state_dict(checkpoint["model_state_dict"])

    def predict(self, game_state):
        encoded_state = self.model(game_state)
        input_tensor = np.array([encoded_state])
        return self._model.predict(input_tensor)[0]

    def encode_game_state(self, game_state):
        current_board = np.zeros((9, 9))
        for r in range(9):
            for c in range(9):
                p = Point(row=r+1, col=c+1)
                stone = game_state.board.get(p)
                if stone is not None:
                    current_board[r][c] = 1 if stone is game_state.next_player else -1
        return current_board.astype(np.float32)

    def select_move(self, game_state):
        current_board = self.encode_game_state(game_state)
        current_board = torch.from_numpy(current_board).to(device)
        current_board = current_board.view(-1, 1, 9, 9)
        score = self.model(current_board)
        score = list(enumerate(score[0].cpu().data.numpy()))
        score = sorted(score, key=lambda v:v[1], reverse=True)
        for pos, _ in score:
            point = Point(pos // 9 + 1, pos % 9 + 1)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board,
                                        point,
                                        game_state.next_player):
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

