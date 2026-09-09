# Project Status: Smart Scan Strategy for Electronic Warfare

**DRDO / SIH Problem Statement 26055**  
**System Designation:** V73 Causal Deep Q-Network (DQN) + Multi-Horizon Temporal GRU  
**Audit Status:** **PHASE 3 COMPLETE — JUDGE-FACING DASHBOARD OPERATIONAL**  
**Dashboard URL:** `http://127.0.0.1:8000`

---

## 1. Verified System Status

| Component | Status | Verification Detail |
| :--- | :--- | :--- |
| **V73 Causal DQN Policy** | **VERIFIED & OPERATIONAL** | 4-layer MLP ($170 \to 256 \to 128 \to 64 \to 5$), 85,253 parameters. Input shape `(1, 170)`, output shape `(1, 5)`. |
| **Multi-Horizon Temporal GRU** | **VERIFIED & OPERATIONAL** | 2-layer GRU (hidden 64) + MLP head, 44,628 parameters. Input `(1, 10, 10)`, output `(1, 5, 4)` for $H+1, H+3, H+5, H+10$. Exact numerical alignment with demo rollout. |
| **Hero RF Waterfall** | **VERIFIED & OPERATIONAL** | Dual-layer visualization separating Simulation Truth (hidden ground truth) from Causal Receiver Observation with prominent cursor and boundary divider. |
| **Demo Story Panel** | **VERIFIED & OPERATIONAL** | Dynamic 7-step narrative explaining the causal loop at the current timestep in plain language. |
| **DQN Q-Value Bars** | **VERIFIED & OPERATIONAL** | Horizontal bar visualization of actual state-action values with "Relevant causal state signals" explanation. |
| **Simulation Benchmark Tab** | **VERIFIED & OPERATIONAL** | Scientifically honest comparison highlighting V73 (highest reward 164.48), V70 (highest Pd 18.43%, lowest delay 1.76), and Round Robin (highest fairness 100%, lowest Pfa 0.85%). |
| **Temporal GRU Ablation Tab** | **VERIFIED & OPERATIONAL** | Verified checkpoint76 deltas (+13.99 reward, +6.90 hits, +3.60 pp fairness) with scientific explanation of scan allocation efficiency. |
| **Causal Security Firewall** | **VERIFIED & TESTED** | Automated tests prove future ground truth changes have 0.00% effect on scheduler decisions at time $t$. |
| **Automated Test Suite** | **6/6 PASSED** | `python -m pytest tests/test_causal_system.py -v` executes in 3.5s with zero errors. |

---

## 2. Operational Modes

1. **VALIDATED REPLAY:**
   - Pre-loads all 500 validated demonstration steps (`checkpoint77_demo_rollout.npz`) into client memory for instant 0ms scrubbing and 60 FPS playback at 1x, 2x, 5x, and 10x speeds.
   - Deterministically reproduces all 145 Hits, 50 Misses, 304 Clean, and 1 False Alarm.
2. **LIVE NEURAL INFERENCE ("Live Neural Inference / Causal Replay Environment"):**
   - Evaluates active PyTorch forward passes on both GRU and DQN at runtime.
   - Updates the causal 10-step history buffer dynamically and computes live simulated receiver scan feedback.

---

## 3. Disclaimers & Boundaries

- **Prototype Nature:** Simulation-based prototype. Evaluation metrics are derived from 40 controlled synthetic RF environments (240 evaluation episodes) and are not field-validation results.
- **Physical Boundary:** Ground truth RF data exists only inside the simulated receiver front-end to compute detection feedback. The neural scheduler receives strictly past observations.

---

## 4. How to Launch

```powershell
python run_dashboard.py
```
Open `http://127.0.0.1:8000` in any web browser.
