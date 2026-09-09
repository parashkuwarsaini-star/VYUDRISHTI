# Project Overview: Smart Scan Strategy for Electronic Warfare

**DRDO / SIH Problem Statement 26055**  
**Project Title:** Smart Scan Strategy for Electronic Warfare (Cognitive Spectrum Monitoring)  
**System Designation:** V73 Causal Deep Q-Network (DQN) with Multi-Horizon Temporal GRU Scheduler

---

## 1. Executive Summary

In Electronic Warfare (EW) and Electronic Support Measures (ESM), tactical receivers face a fundamental operational constraint: **instantaneous receiver bandwidth is strictly narrower than the total operational RF spectrum**. Modern emitter environments feature agile, frequency-hopping, pulse-burst radar and communication threats across multiple wideband channels. A conventional ESM receiver cannot monitor all channels simultaneously and must dynamically schedule which frequency band to tune into at each discrete timestep.

This project delivers a **Causal, Multi-Horizon Reinforcement Learning Receiver Scheduler**:
1. A **Multi-Horizon Temporal GRU** tracks causal receiver observation history (Hits, Misses, False Alarms, Scanned Bands) across 5 frequency bands and forecasts future emitter activity at horizons $H+1$, $H+3$, $H+5$, and $H+10$.
2. A **V73 Causal Scalable DQN** synthesizes receiver history, empirical band metrics, and temporal GRU predictions into a 170-dimensional causal scheduler state to select the optimal frequency band to scan next.
3. The system enforces a **strict causality boundary**: the scheduler and prediction models operate solely on information available up to timestep $t$, strictly preventing any leakage of future emitter ground truth, unobserved SNRs, or oracle labels.

---

## 2. Core Operational Workflow

The system operates in a closed-loop causal cycle:

```
                  +----------------------------------------+
                  |          RF Spectrum Reality           |
                  | (Hidden Ground Truth & Channel Effects)|
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |         Limited-Band Receiver          |
                  |   (Tunes to 1 of 5 Bands per Step)     |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |    Receiver Detection & Telemetry      |
                  |    HIT (+1) / MISS (0) / FA (-1)       |
                  |        Measured In-Band SNR            |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |     Rolling Causal History Buffer      |
                  |  10 Timesteps x (Obs, Scan) per Band   |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |       Multi-Horizon Temporal GRU       |
                  |   Forecasts Activity: H+1, H+3, H+5,   |
                  |     H+10 across all 5 Bands (20 Dim)   |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |   170-Dim Causal Scheduler State       |
                  |  (History Buffer + Temporal Forecasts) |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |        V73 Scalable Causal DQN         |
                  |  Evaluates Q(s, a) for Bands 0..4      |
                  |        Action = argmax Q(s, a)         |
                  +-------------------+--------------------+
                                      |
                                      | Command Receiver to Next Band
                                      +------------> [Repeat]
```

---

## 3. Strict Causality Requirement

Electronic warfare systems deployed on physical hardware must operate strictly **causally**:
- At time $t$, the receiver only knows what it has scanned and detected up to time $t-1$ (and the immediate scan outcome of step $t$).
- **No Ground Truth Leakage**: Unscanned frequency bands remain unobserved. The scheduler does not know whether an emitter was transmitting on an unscanned channel unless it was tuned to that channel.
- **No Oracle Knowledge**: The DQN and GRU models never receive future ground-truth emitter masks, future burst arrival times, or future SNRs.
- Future ground-truth data in this project is strictly confined to offline simulation and evaluation benchmarking to calculate true Probability of Detection ($P_d$) and Interception Delay.

---

## 4. Project Repository Structure

The project assets have been verified and organized into the following layout:

```
sih_Smart_Scan_Final/
├── models/
│   ├── best_dqn_scalable_v73.pt           # Final V73 Causal DQN model weights (344 KB)
│   └── multi_horizon_activity_gru.pt      # Causal Multi-Horizon Activity GRU (185 KB)
├── demo/
│   └── checkpoint77_demo_rollout.npz      # 500-step verified evaluation rollout (20 KB)
├── results/
│   ├── checkpoint75_final_controlled_results.csv    # 240 controlled test episodes (54 KB)
│   ├── checkpoint75_final_controlled_summary.csv    # Benchmark summary table (2 KB)
│   ├── checkpoint76_prediction_ablation_results.csv # 80-episode ablation results (13 KB)
│   ├── checkpoint76_prediction_ablation_delta.csv   # GRU ablation performance deltas (1 KB)
│   ├── checkpoint78_final_metrics.csv               # Formal benchmark publication table (2 KB)
│   └── checkpoint78_v73_comparisons.csv             # Pairwise algorithm comparisons (1 KB)
├── docs/
│   ├── PROJECT_OVERVIEW.md                # This document
│   ├── MODEL_SPECIFICATION.md             # DQN & GRU network architecture & tensor specs
│   ├── DATA_SPECIFICATION.md              # Demonstration rollout data arrays & semantics
│   ├── RESULTS_SPECIFICATION.md           # Controlled benchmark results & ablation data
│   ├── DEMO_SPECIFICATION.md              # Demonstration pipeline & dashboard functional spec
│   └── ARCHITECTURE.md                    # End-to-end mathematical & algorithmic architecture
└── PROJECT_STATUS.md                      # Audit summary, verified components & next steps
```

---

## 5. Verified File Inventory & Integrity

All supplied files have been cryptographically verified:

| File Path | Size (Bytes) | SHA256 Checksum (Prefix) | Role |
| :--- | :--- | :--- | :--- |
| `models/best_dqn_scalable_v73.pt` | 344,913 | `7f4d7d1e2fb2e7d0` | Primary Scan Scheduling Policy |
| `models/multi_horizon_activity_gru.pt` | 185,237 | `5bf89f9e42281b5a` | Causal Multi-Horizon Activity Forecaster |
| `demo/checkpoint77_demo_rollout.npz` | 19,762 | `dceef0ec03315fc4` | Validated 500-step demonstration episode |
| `results/checkpoint75_final_controlled_results.csv` | 53,952 | `6f92717b1def2b98` | Per-episode performance across 40 seeds |
| `results/checkpoint75_final_controlled_summary.csv` | 2,135 | `7a3443c933082405` | Summary statistics across 6 algorithms |
| `results/checkpoint76_prediction_ablation_results.csv` | 12,628 | `e431b0108a7985fc` | Paired ablation evaluations (With vs Without GRU) |
| `results/checkpoint76_prediction_ablation_delta.csv` | 525 | `785697d8bb52d768` | Performance gains attributed directly to GRU |
| `results/checkpoint78_final_metrics.csv` | 2,486 | `ac8a49567639df63` | Consolidated publication benchmark metrics |
| `results/checkpoint78_v73_comparisons.csv` | 617 | `2b95e6a298ee65a5` | Pairwise comparative performance deltas |\n