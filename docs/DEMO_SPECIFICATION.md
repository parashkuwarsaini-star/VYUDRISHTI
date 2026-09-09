# Demonstration Specification: Interactive Dashboard & Playback Pipeline

**DRDO / SIH Problem Statement 26055**  
**Document:** Functional and Visual Requirements for the SIH Interactive EW Demonstration Dashboard

---

## 1. Purpose of the Demonstration Dashboard

The dashboard serves as the interactive evaluator and presentation interface for the DRDO/SIH Problem Statement 26055. It must:
1. Visually demonstrate **cognitive spectrum surveillance** in a 5-channel RF environment using a single limited-band receiver.
2. Provide **real-time step-by-step causal playback** over the 500-step validated rollout (`checkpoint77_demo_rollout.npz`).
3. Visually distinguish between **Hidden Spectrum Ground Truth** (RF environment reality) and **Causal Receiver Perception** (what the system is allowed to observe).
4. Render the **Multi-Horizon Temporal GRU Forecasts** ($H+1, H+3, H+5, H+10$) as an interactive predictive radar HUD.
5. Present the verified **Controlled Benchmarks** and **Ablation Delays** in publication-grade tables and charts.

---

## 2. Demonstration Data Flow

```
[ checkpoint77_demo_rollout.npz ]
            │
            ├── Step Counter (t = 0 .. 499)
            │
            ├──► RF Spectrum Reality View (Ground Truth Activity & Hidden SNR)
            │
            ├──► Receiver Scan Decision (Current Band Action a_t in 0..4)
            │
            ├──► Receiver Detection Event (HIT / MISS / FALSE ALARM + Measured SNR)
            │
            ├──► Causal History Buffer (Rolling 10-step sequence)
            │           │
            │           ▼
            │     [ multi_horizon_activity_gru.pt ] (Inference or Verified Cache)
            │           │
            │           ▼
            │     Multi-Horizon Activity Forecast (H+1, H+3, H+5, H+10)
            │
            └──► Cumulative Metric Telemetry (Pd, Pfa, Fairness, Reward)
```

---

## 3. Core Visual Panels & User Controls

### 3.1 Interactive Playback Controls
- **Timeline Slider:** Scrub across timesteps $t \in [0, 499]$.
- **Transport Buttons:** Play (at 1x, 2x, 5x speed), Pause, Step Forward, Step Backward, Reset.
- **Current Step Indicator:** Displays current time step $t$ out of 500.

### 3.2 Panel 1: Multi-Band RF Waterfall & Ground Truth
- 5 horizontal tracks corresponding to Frequency Bands 0, 1, 2, 3, 4.
- Displays true emitter transmissions as filled blocks over time.
- Overlays the **Receiver Scan Cursor** showing which single band was tuned at step $t$.
- Highlights unscanned bands to visually demonstrate receiver blindness to unscanned channels.

### 3.3 Panel 2: Receiver Intercept Telemetry & Signal Events
- Displays detection badge for current timestep:
  - **HIT (Green):** Emitter present and intercepted ($+1$).
  - **MISS (Yellow/Orange):** Emitter present on scanned band, but SNR too weak ($0$).
  - **CLEAN (Slate Gray):** Channel inactive, no false trigger ($0$).
  - **FALSE ALARM (Red):** Inactive band falsely triggered detector ($-1$).
- Displays **Measured In-Band SNR** gauge (dB) and instant step reward.

### 3.4 Panel 3: Multi-Horizon Temporal Predictive Radar HUD
- Heatmap / bar chart showing predicted activity probabilities across all 5 bands for:
  - **$H+1$** (Immediate next step)
  - **$H+3$** (Short-range burst forecast)
  - **$H+5$** (Mid-range tactical horizon)
  - **$H+10$** (Long-range horizon)
- Visually highlights which band is forecast to have the highest imminent threat activity.

### 3.5 Panel 4: Live Cumulative Operational Metrics
- Dynamic live counters updating as $t$ advances:
  - **Overall $P_d$:** Running detection rate across total spectrum opportunities.
  - **Scanned $P_d$:** Running detection efficiency on scanned channels.
  - **$P_{fa}$:** Running false alarm rate.
  - **Jain's Fairness Index:** Real-time scan balance across bands.
  - **Cumulative Reward:** RL reward trajectory.

### 3.6 Panel 5: Controlled Benchmark & Comparison Suite
- Interactive presentation of verified data from `checkpoint78_final_metrics.csv` and `checkpoint78_v73_comparisons.csv`.
- Side-by-side radar / bar charts comparing V73 against V70, V65, Old DQN, Round Robin, and Random.

### 3.7 Panel 6: Multi-Horizon GRU Ablation Showcase
- Interactive comparison of `checkpoint76_prediction_ablation_delta.csv`.
- Direct visual evidence proving that adding the GRU improves Reward (+9.5%), Detection Hits (+4.4%), and Fairness (+4.1%).

---

## 4. Demonstration Causal Boundary Rules

1. **Strict Separation:** The dashboard must visually maintain two distinct zones:
   - *Environment Inspector (Ground Truth):* Clearly labeled as internal simulation / oracle benchmark data.
   - *Receiver Perception (Causal Agent):* Shows only what the receiver has scanned and detected up to current step $t$.
2. **Deterministic Playback:** The dashboard will read directly from the verified 500-step rollout array, guaranteeing zero simulation drift or non-reproducibility during live judge presentations.\n