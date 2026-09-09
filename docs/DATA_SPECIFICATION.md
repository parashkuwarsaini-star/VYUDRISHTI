# Data Specification: Demonstration Rollout & Telemetry

**DRDO / SIH Problem Statement 26055**  
**Document:** File Contents, Data Dtypes, Array Shapes, and Semantics for `checkpoint77_demo_rollout.npz`

---

## 1. Overview of Demonstration Rollout

The file `demo/checkpoint77_demo_rollout.npz` (size: 19,762 bytes, SHA256 prefix: `dceef0ec03315fc4`) contains a fully synchronized, 500-timestep execution trace of the V73 Causal DQN operating on an EW receiver environment. It captures the interaction between:
1. Ground truth RF spectrum reality (hidden from the scheduler).
2. Receiver scans, observations, and telemetry (causally observed).
3. Temporal GRU forecasts.
4. Cumulative operational EW performance metrics.

---

## 2. Array Inventory & Exact Shapes

The NPZ archive contains 14 verified keys:

| Key | Shape | Dtype | Range (Min / Max) | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ground_truth` | `(5, 500)` | `float32` | `0.0` to `1.0` | True binary emitter activity across 5 bands over 500 steps |
| `snr_hidden` | `(5, 500)` | `float32` | `-10.81` to `+41.78` dB | True RF signal-to-noise ratio across all 5 bands |
| `actions` | `(500,)` | `int32` | `0` to `4` | Scanned frequency band action selected at each step |
| `observations` | `(500,)` | `int32` | `-1` to `+1` | Receiver detection outcome (+1: HIT, 0: MISS/CLEAN, -1: FA) |
| `measured_snr` | `(500,)` | `float32` | `-12.17` to `+32.55` dB | Observed SNR on the scanned band only |
| `prediction_h1` | `(500, 5)` | `float32` | `0.042` to `0.945` | GRU-predicted activity probabilities at horizon H+1 |
| `rewards` | `(500,)` | `float32` | `-0.24` to `+0.96` | Instantaneous RL step reward received by the scheduler |
| `scan_matrix` | `(5, 500)` | `float32` | `0.0` to `1.0` | One-hot binary indicator matrix of scanned bands per step |
| `observation_matrix` | `(5, 500)` | `float32` | `-1.0` to `+1.0` (with `NaN`) | Sparse observation placed at scanned band (`NaN` for unscanned) |
| `scan_counts` | `(5,)` | `float32` | `[100, 124, 85, 89, 102]` | Total number of scans allocated to each band (Sum = 500) |
| `pd_overall` | `()` (scalar) | `float64` | `15.8817%` | True overall probability of detection across all bands |
| `pd_scanned` | `()` (scalar) | `float64` | `74.3590%` | Detection probability when the scanned band was active |
| `pfa` | `()` (scalar) | `float64` | `0.3279%` | Empirical Probability of False Alarm on inactive scans |
| `fairness` | `()` (scalar) | `float32` | `98.1817%` | Jain's Fairness Index across the 5 band scan distributions |

---

## 3. Detection Semantics & Confusion Matrix

Analysis of the 500 steps reveals the exact physical semantics of `observations`:

| Observation Code | Physical State | Count in Rollout | Ground Truth | Mean In-Band SNR | Mean Reward | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `+1` | **HIT** | 145 | Active (`1`) | $+15.28$ dB | $+0.960$ | Successful intercept of active radar/comm emitter |
| `0` | **MISS** | 50 | Active (`1`) | $+2.08$ dB | $-0.057$ | Emitter active, but missed due to low SNR / channel fade |
| `0` | **CLEAN** | 304 | Inactive (`0`) | $-0.11$ dB | $-0.059$ | Band correctly observed as inactive (noise background) |
| `-1` | **FALSE ALARM** | 1 | Inactive (`0`) | $+0.10$ dB | $-0.240$ | Noise fluctuation falsely triggered detection threshold |
| **Total** | | **500** | | | | |

---

## 4. Derived Metric Formulations & Verification

All scalar metrics stored in `checkpoint77_demo_rollout.npz` strictly match their physical radar/EW definitions:

### 4.1 Overall Probability of Detection ($P_{d, \text{overall}}$)
Ratio of successful detections to total emitter transmission opportunities across all 5 channels over the entire 500-step episode:
$$P_{d, \text{overall}} = \frac{\text{Hits}}{\sum_{b=0}^{4} \sum_{t=0}^{499} \text{ground\_truth}[b, t]} = \frac{145}{913} = 15.88170865\%$$

### 4.2 Scanned Probability of Detection ($P_{d, \text{scanned}}$)
Detection efficiency when the receiver chose to scan an active channel:
$$P_{d, \text{scanned}} = \frac{\text{Hits}}{\text{Scanned True Active}} = \frac{145}{145 + 50} = \frac{145}{195} = 74.35897436\%$$

### 4.3 Probability of False Alarm ($P_{fa}$)
Rate of false detections when the receiver scanned an inactive channel:
$$P_{fa} = \frac{\text{False Alarms}}{\text{Scanned Inactive Opportunities}} = \frac{1}{1 + 304} = \frac{1}{305} = 0.32786885\%$$

### 4.4 Jain's Fairness Index ($J$)
Measures scan balance across the $B=5$ bands for scan counts $\mathbf{s} = [100, 124, 85, 89, 102]$:
$$J = \frac{\left(\sum_{b=0}^{4} s_b\right)^2}{5 \cdot \sum_{b=0}^{4} s_b^2} = \frac{500^2}{5 \cdot (100^2 + 124^2 + 85^2 + 89^2 + 102^2)} = \frac{250,000}{5 \cdot 50,926} = \frac{250,000}{254,630} = 98.181679\%$$

---

## 5. Band Allocation Distribution

The scan distribution demonstrates intelligent, balanced cognitive scanning:
- Band 0: 100 scans (20.0%)
- Band 1: 124 scans (24.8%)
- Band 2: 85 scans (17.0%)
- Band 3: 89 scans (17.8%)
- Band 4: 102 scans (20.4%)
- **Total:** 500 scans (100.0%)
- **Min Scan Share:** 17.0% (Band 2)
- **Max Scan Share:** 24.8% (Band 1)
- **Max-to-Min Scan Ratio:** $124 / 85 = 1.4588$\n