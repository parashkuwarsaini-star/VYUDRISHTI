# Model Specification: V73 DQN & Multi-Horizon GRU

**DRDO / SIH Problem Statement 26055**  
**Document:** Formal Model Architecture, Weight Inventory, and Inference Specifications

---

## 1. Model Inventory Overview

The system utilizes two synchronized neural network models:

| Attribute | Scheduler Policy (DQN) | Temporal Forecaster (GRU) |
| :--- | :--- | :--- |
| **Model Filename** | `best_dqn_scalable_v73.pt` | `multi_horizon_activity_gru.pt` |
| **Primary Task** | Frequency Band Selection (Control) | Future Emitter Activity Forecasting |
| **Input Shape** | `(batch_size, 170)` | `(batch_size, 10, 10)` |
| **Output Shape** | `(batch_size, 5)` (Q-values) | `(batch_size, 20)` -> reshaped to `(5, 4)` |
| **Parameter Count** | 85,253 parameters | 44,628 parameters |
| **File Size** | 344,913 bytes | 185,237 bytes |
| **Checkpoint Type** | `collections.OrderedDict` (State Dict) | `dict` (Metadata + `model_state_dict`) |
| **PyTorch Version** | Compatible with PyTorch 2.x (CPU/CUDA) | Compatible with PyTorch 2.x (CPU/CUDA) |
| **Loading Status** | Verified (Loads Cleanly) | Verified (Loads Cleanly) |

---

## 2. V73 Scalable Causal DQN Specification

### 2.1 File Characteristics & Checksum
- **Relative Path:** `models/best_dqn_scalable_v73.pt`
- **SHA256:** `7f4d7d1e2fb2e7d0a21fa31aa0eeea1eb44d8258525b6a71cb0a842b15764d93`
- **Data Structure:** PyTorch state dictionary (`collections.OrderedDict`)

### 2.2 Network Architecture
The network is a 4-layer fully connected deep Q-network ($170 \to 256 \to 128 \to 64 \to 5$) with ReLU non-linearities:

```
Input State (170-dim)
       │
       ▼
Linear Layer 0: [256, 170] + Bias [256]
       │
       ▼
  ReLU Activation
       │
       ▼
Linear Layer 2: [128, 256] + Bias [128]
       │
       ▼
  ReLU Activation
       │
       ▼
Linear Layer 4: [64, 128] + Bias [64]
       │
       ▼
  ReLU Activation
       │
       ▼
Linear Layer 6: [5, 64] + Bias [5]
       │
       ▼
Q-Values for 5 Frequency Bands: Q(s, a_0) ... Q(s, a_4)
```

### 2.3 Layer Parameter Breakdown

| Layer Key | Tensor Type | Shape | Parameters | Value Range (Min / Max) |
| :--- | :--- | :--- | :--- | :--- |
| `network.0.weight` | Weight matrix | `[256, 170]` | 43,520 | `[-1.7225, +1.3135]` |
| `network.0.bias` | Bias vector | `[256]` | 256 | `[-0.1715, +0.1198]` |
| `network.2.weight` | Weight matrix | `[128, 256]` | 32,768 | `[-1.6837, +0.8396]` |
| `network.2.bias` | Bias vector | `[128]` | 128 | `[-0.4274, +0.5964]` |
| `network.4.weight` | Weight matrix | `[64, 128]` | 8,192 | `[-1.3615, +0.7557]` |
| `network.4.bias` | Bias vector | `[64]` | 64 | `[-0.8100, +1.4849]` |
| `network.6.weight` | Weight matrix | `[5, 64]` | 320 | `[-0.9206, +0.7450]` |
| `network.6.bias` | Bias vector | `[5]` | 5 | `[+0.7756, +0.9788]` |
| **Total** | | | **85,253** | |

### 2.4 Reference PyTorch Implementation
```python
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

# Load state dict
dqn_model = ScalableDQN(170, 5)
state_dict = torch.load('models/best_dqn_scalable_v73.pt', map_location='cpu')
dqn_model.load_state_dict(state_dict)
dqn_model.eval()
```

### 2.5 Action Space & Decision Rule
- **Action Space:** Discrete $\mathcal{A} = \{0, 1, 2, 3, 4\}$, where action $a_t = b$ commands the receiver to scan frequency band $b$ at time $t$.
- **Action Selection Rule:** Greedy policy $a_t = \arg\max_{b \in \{0..4\}} Q(s_t, b)$.

---

## 3. Multi-Horizon Temporal Activity GRU Specification

### 3.1 File Characteristics & Checksum
- **Relative Path:** `models/multi_horizon_activity_gru.pt`
- **SHA256:** `5bf89f9e42281b5a5b172a3e0bfe1bca96eeb88383a19bc8a8eb6a7c385c7cfa`
- **Data Structure:** Python dictionary containing metadata and weights.

### 3.2 Checkpoint Metadata
Inspection of the checkpoint dictionary reveals the following hyperparameters:
- `input_size`: `10`
- `hidden_size`: `64`
- `n_bands`: `5`
- `horizons`: `[1, 3, 5, 10]`
- `history_length`: `10`
- `model_state_dict`: Contains all tensor weights and biases.

### 3.3 Network Architecture
The architecture comprises a 2-layer Gated Recurrent Unit (GRU) followed by a 2-layer Multi-Layer Perceptron (MLP) prediction head:

```
Observation History Tensor: [batch, 10, 10]
                   │
                   ▼
2-Layer GRU (input_size=10, hidden_size=64, batch_first=True)
  - Layer 0: 64 hidden units
  - Layer 1: 64 hidden units
                   │
                   ▼
Last Time-Step Hidden State: h_T [batch, 64]
                   │
                   ▼
Head Linear 0: [64, 64] + Bias [64]
                   │
                   ▼
            ReLU Activation
                   │
                   ▼
Head Linear 2: [20, 64] + Bias [20]
                   │
                   ▼
Sigmoid Activation -> Probabilities in [0, 1]
                   │
                   ▼
Reshape to [batch, 5, 4] -> (Bands 0..4 x Horizons H+1, H+3, H+5, H+10)
```

### 3.4 Layer Parameter Breakdown

| Parameter Name | Shape | Parameters | Value Range (Min / Max) | Description |
| :--- | :--- | :--- | :--- | :--- |
| `gru.weight_ih_l0` | `[192, 10]` | 1,920 | `[-0.9247, +0.7917]` | Layer 0 Input-Hidden GRU gates |
| `gru.weight_hh_l0` | `[192, 64]` | 12,288 | `[-0.7708, +0.7709]` | Layer 0 Hidden-Hidden GRU gates |
| `gru.bias_ih_l0` | `[192]` | 192 | `[-0.4647, +0.1337]` | Layer 0 Input bias |
| `gru.bias_hh_l0` | `[192]` | 192 | `[-0.4450, +0.1662]` | Layer 0 Hidden bias |
| `gru.weight_ih_l1` | `[192, 64]` | 12,288 | `[-0.8075, +0.9062]` | Layer 1 Input-Hidden GRU gates |
| `gru.weight_hh_l1` | `[192, 64]` | 12,288 | `[-0.7636, +0.6452]` | Layer 1 Hidden-Hidden GRU gates |
| `gru.bias_ih_l1` | `[192]` | 192 | `[-0.3586, +0.3329]` | Layer 1 Input bias |
| `gru.bias_hh_l1` | `[192]` | 192 | `[-0.3765, +0.2608]` | Layer 1 Hidden bias |
| `head.0.weight` | `[64, 64]` | 4,096 | `[-0.4660, +0.6185]` | MLP Projection Layer weight |
| `head.0.bias` | `[64]` | 64 | `[-0.2382, +0.2017]` | MLP Projection Layer bias |
| `head.2.weight` | `[20, 64]` | 1,280 | `[-0.3417, +0.4930]` | Output Projection Layer weight |
| `head.2.bias` | `[20]` | 20 | `[-0.2051, +0.1733]` | Output Projection Layer bias |
| **Total** | | **44,628** | | |

### 3.5 Input Feature Encoding (Interleaved per Band)
At each timestep $\tau \in [t-9, t]$ of the 10-step history window, the 10 input features are formed by pairing observation and scan status for each frequency band:

$$\mathbf{x}_\tau = [o_{0, \tau}, s_{0, \tau}, o_{1, \tau}, s_{1, \tau}, o_{2, \tau}, s_{2, \tau}, o_{3, \tau}, s_{3, \tau}, o_{4, \tau}, s_{4, \tau}]^T$$

Where for each band $b \in \{0, 1, 2, 3, 4\}$:
- $o_{b, \tau} \in \{+1.0, 0.0, -1.0\}$:
  - $+1.0$: HIT (emitter signal detected)
  - $-1.0$: FALSE ALARM (signal detection triggered on inactive band)
  - $0.0$: MISS, CLEAN inactive scan, or unscanned band
- $s_{b, \tau} \in \{0.0, 1.0\}$:
  - $1.0$: Frequency band $b$ was scanned by receiver at timestep $\tau$
  - $0.0$: Band $b$ was not scanned

### 3.6 Output Tensor Structure
The model head outputs a 20-dimensional vector passed through a Sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$. When reshaped to `(5, 4)`:
- Row $b \in \{0, 1, 2, 3, 4\}$ corresponds to Frequency Band $b$.
- Column $0$: Horizon $H+1$ activity probability
- Column $1$: Horizon $H+3$ activity probability
- Column $2$: Horizon $H+5$ activity probability
- Column $3$: Horizon $H+10$ activity probability

### 3.7 Verification Against Demo Rollout
Running the GRU over the 500 timesteps of `checkpoint77_demo_rollout.npz` with this exact input encoding produces predictions whose $H+1$ column matches the stored `prediction_h1` array with a maximum numerical discrepancy of **$1.2 \times 10^{-7}$** (within single-precision float epsilon), confirming exact architectural and feature alignment.

### 3.8 Reference PyTorch Implementation
```python
import torch
import torch.nn as nn

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 10, 10)
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :] # (batch_size, 64)
        logits = self.head(last_hidden) # (batch_size, 20)
        probs = torch.sigmoid(logits)    # (batch_size, 20)
        return probs.view(-1, 5, 4)      # (batch_size, 5 bands, 4 horizons)

# Load checkpoint
ckpt = torch.load('models/multi_horizon_activity_gru.pt', map_location='cpu')
gru_model = MultiHorizonActivityGRU(
    input_size=ckpt['input_size'],
    hidden_size=ckpt['hidden_size'],
    num_layers=2,
    output_size=len(ckpt['horizons']) * ckpt['n_bands']
)
gru_model.load_state_dict(ckpt['model_state_dict'])
gru_model.eval()
```\n