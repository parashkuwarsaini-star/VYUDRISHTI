# System Architecture: Causal Cognitive EW Receiver

**DRDO / SIH Problem Statement 26055**  
**Document:** End-to-End Algorithmic Architecture, Causal Information Filtration, and Subsystem Interfacing

---

## 1. High-Level Architectural Flow

```
                      +-----------------------------+
                      |    RF Spectrum Reality      |
                      |   Ground Truth s*_t in R^5   |
                      |    Hidden SNR in R^5        |
                      +--------------+--------------+
                                     |
                                     v
                       [ Receiver Front-End ]
                   Tunes to selected band a_t in {0..4}
                                     |
                                     v
                      +-----------------------------+
                      |   Causal Sensor Detector    |
                      |   o_t in {+1, 0, -1}        |
                      |   gamma_t (Measured SNR)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Rolling Observation Buffer  |
                      |   H_t in R^(10 x 10)        |
                      | Interleaved [obs, scan]     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Multi-Horizon Temporal GRU  |
                      | 2-Layer GRU (hidden=64)     |
                      |   MLP Head (20 logits)      |
                      | Sigmoid -> P_(t+h) in R^20  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Causal Scheduler State    |
                      |        s_t in R^170         |
                      | 100-dim History + 70-dim    |
                      | Band Metrics & Forecasts    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       V73 Scalable DQN      |
                      |  4-Layer MLP: 170->256->    |
                      |        128->64->5           |
                      | Q(s_t, a) for a in {0..4}   |
                      +--------------+--------------+
                                     |
                                     v
                       a_(t+1) = argmax_a Q(s_t, a)
                                     |
                         Command Receiver Tuning
                                     +----------------> [Next Step]
```

---

## 2. Mathematical Definition of Causal Filtration

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be the probability space governing the RF spectrum. At discrete time $t \in \{0, 1, \dots, T-1\}$, the true RF environment state is:
$$\mathbf{y}_t = [y_{0, t}, y_{1, t}, y_{2, t}, y_{3, t}, y_{4, t}]^T \in \{0, 1\}^5$$

The receiver selects action $a_t \in \{0, 1, 2, 3, 4\}$. The receiver's observation at time $t$ is:
$$o_{a_t, t} = g(y_{a_t, t}, \text{SNR}_{a_t, t}, \eta_t)$$
where $\eta_t$ is receiver thermal noise. For all unscanned bands $b \neq a_t$, $o_{b, t} = \varnothing$ (unobserved).

The **causal information filtration** available to the scheduler at time $t$ is:
$$\mathcal{F}_t = \sigma \left( \left\{ a_\tau, o_{a_\tau, \tau}, \text{SNR}_{a_\tau, \tau} \right\}_{\tau=0}^{t-1} \right)$$

**Strict Causality Axiom:**
$$\mathbb{E}[a_t \mid \mathcal{F}_t, \mathbf{y}_{t}, \mathbf{y}_{t+1}, \dots] = \mathbb{E}[a_t \mid \mathcal{F}_t]$$
No future ground truth or unobserved channel state enters the policy $\pi(a_t \mid s_t)$.

---

## 3. Subsystem Breakdown

### Subsystem 1: Observation & History Management
- Maintains a circular FIFO buffer $\mathbf{H}_t \in \mathbb{R}^{10 \times 10}$.
- For each step $\tau \in [t-9, t]$ and each band $b \in \{0..4\}$:
  - Index $2b$: Detection outcome $o_{b, \tau} \in \{+1.0, 0.0, -1.0\}$.
  - Index $2b+1$: Scan indicator $s_{b, \tau} \in \{0.0, 1.0\}$.

### Subsystem 2: Multi-Horizon Temporal Forecaster
- **Input:** Buffer $\mathbf{H}_t \in \mathbb{R}^{1 \times 10 \times 10}$.
- **Recurrent Core:** 2-layer GRU with 64 hidden units per layer.
- **Output Projection:** 2-layer MLP producing 20 values, transformed via Sigmoid into activity probabilities:
  $$\hat{\mathbf{P}}_t = [\hat{p}_{b, t+h}] \in [0, 1]^{5 \times 4}, \quad h \in \{1, 3, 5, 10\}$$

### Subsystem 3: 170-Dimensional Causal Scheduler State
- **Dimensions 0 to 99 (100 dims):** Flattened rolling observation and scan history buffer $\mathbf{H}_t$ ($10 \times 10 = 100$).
- **Dimensions 100 to 169 (70 dims):** 5 frequency band blocks (14 features per band):
  - Each band block contains empirical channel occupancy metrics, recent activity markers, and the 4 multi-horizon GRU predictions ($H+1, H+3, H+5, H+10$).

### Subsystem 4: V73 Causal Deep Q-Network Policy
- Evaluates the state-action value function $Q(s_t, a)$ across all 5 bands.
- Selects the next frequency band via greedy exploitation:
  $$a_{t+1} = \arg\max_{a \in \{0..4\}} Q(s_t, a)$$
- Tuning command is dispatched to receiver front-end.\n