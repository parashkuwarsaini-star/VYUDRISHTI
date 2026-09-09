import os
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

class ScalableDQN(nn.Module):
    def __init__(self, input_dim: int = 170, output_dim: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiHorizonActivityGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 20
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        logits = self.head(last_hidden)
        probs = torch.sigmoid(logits)
        reshaped_probs = probs.view(-1, 5, 4)
        return reshaped_probs, logits


def load_dqn_model(model_path: str) -> ScalableDQN:
    model = ScalableDQN(input_dim=170, output_dim=5)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_gru_model(model_path: str) -> Tuple[MultiHorizonActivityGRU, Dict[str, Any]]:
    ckpt = torch.load(model_path, map_location='cpu')
    metadata = {
        'input_size': ckpt.get('input_size', 10),
        'hidden_size': ckpt.get('hidden_size', 64),
        'n_bands': ckpt.get('n_bands', 5),
        'horizons': ckpt.get('horizons', [1, 3, 5, 10]),
        'history_length': ckpt.get('history_length', 10)
    }
    model = MultiHorizonActivityGRU(
        input_size=metadata['input_size'],
        hidden_size=metadata['hidden_size'],
        num_layers=2,
        output_size=len(metadata['horizons']) * metadata['n_bands']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, metadata
