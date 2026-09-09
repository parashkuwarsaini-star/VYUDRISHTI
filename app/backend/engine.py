import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from app.backend.models import ScalableDQN, MultiHorizonActivityGRU, load_dqn_model, load_gru_model

def to_clean_json(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [to_clean_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: to_clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_clean_json(x) for x in obj]
    return obj


class ValidatedReplayEngine:
    def __init__(self, demo_path: str, gru_model: MultiHorizonActivityGRU, dqn_model: ScalableDQN):
        self.demo_path = demo_path
        self.gru = gru_model
        self.dqn = dqn_model
        self.data = np.load(demo_path, allow_pickle=True)
        self.n_steps = len(self.data['actions']) # 500
        self.n_bands = 5

        self.gt = self.data['ground_truth']               # (5, 500)
        self.snr_hidden = self.data['snr_hidden']         # (5, 500)
        self.actions = self.data['actions']               # (500,)
        self.observations = self.data['observations']     # (500,)
        self.measured_snr = self.data['measured_snr']     # (500,)
        self.prediction_h1 = self.data['prediction_h1']   # (500, 5)
        self.rewards = self.data['rewards']               # (500,)
        self.scan_matrix = self.data['scan_matrix']       # (5, 500)
        self.obs_matrix = np.nan_to_num(self.data['observation_matrix'], nan=0.0) # (5, 500)

        self._precompute_timeline()

    def _precompute_timeline(self):
        self.gru_predictions = np.zeros((self.n_steps, 5, 4), dtype=np.float32)
        self.dqn_q_values = np.zeros((self.n_steps, 5), dtype=np.float32)
        self.running_metrics = []
        self.causal_signals = []
        self.cached_steps = []

        hist_buf = np.zeros((10, 10), dtype=np.float32)
        cum_hits = 0
        cum_misses = 0
        cum_fa = 0
        cum_clean = 0
        cum_scans = np.zeros(5, dtype=int)
        cum_hits_per_band = np.zeros(5, dtype=int)
        last_seen_step = np.full(5, -1, dtype=int)
        cum_reward = 0.0

        for t in range(self.n_steps):
            # 1. Run GRU on strictly causal history buffer
            with torch.no_grad():
                probs, _ = self.gru(torch.tensor(hist_buf, dtype=torch.float32).unsqueeze(0))
                p_matrix = probs.squeeze(0).numpy()
            self.gru_predictions[t] = p_matrix

            # 2. Construct causal DQN state
            s = np.zeros(170, dtype=np.float32)
            s[:100] = hist_buf.flatten()
            total_s = max(1, cum_scans.sum())
            for b in range(5):
                base = 100 + b * 14
                s[base + 8 : base + 12] = p_matrix[b]
                s[base + 0] = cum_scans[b] / total_s
            
            # 3. Run DQN
            with torch.no_grad():
                q_vals = self.dqn(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            self.dqn_q_values[t] = q_vals

            # Process step t outcome
            a_t = int(self.actions[t])
            cum_scans[a_t] += 1
            o_t = int(self.observations[t])
            r_t = float(self.rewards[t])
            cum_reward += r_t

            gt_active = (self.gt[a_t, t] > 0.5)
            if o_t == 1:
                cum_hits += 1
                cum_hits_per_band[a_t] += 1
                result_type = 'HIT'
                last_seen_step[a_t] = t
            elif o_t == -1:
                cum_fa += 1
                result_type = 'FALSE ALARM'
            elif gt_active:
                cum_misses += 1
                result_type = 'MISS'
            else:
                cum_clean += 1
                result_type = 'CLEAN'

            # Metrics
            total_active_so_far = float(np.sum(self.gt[:, :t+1]))
            scanned_active_so_far = float(cum_hits + cum_misses)
            scanned_inactive_so_far = float(cum_fa + cum_clean)

            pd_overall = float((cum_hits / total_active_so_far * 100.0) if total_active_so_far > 0 else 0.0)
            pd_scanned = float((cum_hits / scanned_active_so_far * 100.0) if scanned_active_so_far > 0 else 0.0)
            pfa = float((cum_fa / scanned_inactive_so_far * 100.0) if scanned_inactive_so_far > 0 else 0.0)

            sum_s = float(cum_scans.sum())
            sum_sq = float((cum_scans**2).sum())
            fairness = float(((sum_s**2) / (5.0 * sum_sq) * 100.0) if sum_sq > 0 else 100.0)

            metric_entry = {
                'step': int(t),
                'cum_hits': int(cum_hits),
                'cum_misses': int(cum_misses),
                'cum_fa': int(cum_fa),
                'cum_clean': int(cum_clean),
                'cum_reward': float(round(cum_reward, 2)),
                'scan_counts': [int(c) for c in cum_scans],
                'pd_overall': float(round(pd_overall, 2)),
                'pd_scanned': float(round(pd_scanned, 2)),
                'pfa': float(round(pfa, 3)),
                'fairness': float(round(fairness, 2)),
                'result_type': result_type
            }
            self.running_metrics.append(metric_entry)

            # Causal state signals
            recency = (t - last_seen_step[a_t]) if last_seen_step[a_t] >= 0 else t
            band_hit_rate = (cum_hits_per_band[a_t] / cum_scans[a_t]) if cum_scans[a_t] > 0 else 0.0
            signals = {
                'selected_band': a_t,
                'max_q': float(round(float(q_vals[a_t]), 2)),
                'h1_prob': float(round(float(p_matrix[a_t, 0]) * 100, 1)),
                'h3_prob': float(round(float(p_matrix[a_t, 1]) * 100, 1)),
                'h5_prob': float(round(float(p_matrix[a_t, 2]) * 100, 1)),
                'h10_prob': float(round(float(p_matrix[a_t, 3]) * 100, 1)),
                'scan_share': float(round((cum_scans[a_t] / total_s) * 100, 1)),
                'band_hit_rate': float(round(band_hit_rate * 100, 1)),
                'recency': int(recency)
            }
            self.causal_signals.append(signals)

            # Update causal history buffer
            step_vec = np.zeros(10, dtype=np.float32)
            for b in range(5):
                step_vec[2*b] = float(self.obs_matrix[b, t])
                step_vec[2*b+1] = float(self.scan_matrix[b, t])
            hist_buf = np.roll(hist_buf, -1, axis=0)
            hist_buf[-1] = step_vec

        # Prebuild clean JSON steps
        for t in range(self.n_steps):
            self.cached_steps.append(self._build_step_dict(t))

    def _build_step_dict(self, t: int) -> Dict[str, Any]:
        a_t = int(self.actions[t])
        o_t = int(self.observations[t])
        snr_meas = float(self.measured_snr[t])
        r_t = float(self.rewards[t])
        metrics = self.running_metrics[t]
        sig = self.causal_signals[t]

        # Story narration steps
        next_band = int(np.argmax(self.dqn_q_values[t]))
        story = [
            f"1. Limited receiver tunes to Band {a_t} at time t={t}.",
            f"2. Simulated receiver scan evaluates channel -> Reports {metrics['result_type']} (SNR: {snr_meas:+.1f} dB).",
            f"3. Observation appended to 10-step causal history buffer H_t.",
            f"4. Multi-Horizon GRU evaluates H_t -> Forecasts Band {a_t} activity: H+1: {sig['h1_prob']:.1f}%, H+3: {sig['h3_prob']:.1f}%.",
            f"5. V73 DQN policy evaluates 170-D causal scheduler state across all 5 bands.",
            f"6. Band {next_band} produces highest Q-value ({self.dqn_q_values[t, next_band]:.2f}).",
            f"7. Scheduler commands receiver to tune to Band {next_band} next."
        ]

        res = {
            'step': int(t),
            'mode': 'VALIDATED_REPLAY',
            'action': int(a_t),
            'next_action': next_band,
            'observation': int(o_t),
            'result_type': metrics['result_type'],
            'measured_snr': float(round(snr_meas, 2)),
            'reward': float(round(r_t, 3)),
            'signals': sig,
            'story': story,
            'q_values': [float(round(float(q), 3)) for q in self.dqn_q_values[t]],
            'gru_predictions': [[float(round(float(p), 4)) for p in row] for row in self.gru_predictions[t]],
            'metrics': metrics,
            'gt_active_now': [bool(self.gt[b, t] > 0.5) for b in range(5)],
            'scan_matrix_step': [int(self.scan_matrix[b, t]) for b in range(5)],
            'total_steps': int(self.n_steps)
        }
        return to_clean_json(res)

    def get_step(self, t: int) -> Dict[str, Any]:
        t = max(0, min(t, self.n_steps - 1))
        return self.cached_steps[t]


class LiveInferenceEngine:
    def __init__(self, demo_path: str, gru_model: MultiHorizonActivityGRU, dqn_model: ScalableDQN):
        self.demo_path = demo_path
        self.gru = gru_model
        self.dqn = dqn_model
        self.data = np.load(demo_path, allow_pickle=True)
        self.n_steps = len(self.data['actions'])
        self.gt = self.data['ground_truth']
        self.snr_hidden = self.data['snr_hidden']
        self.reset()

    def reset(self):
        self.current_step = 0
        self.hist_buf = np.zeros((10, 10), dtype=np.float32)
        self.cum_hits = 0
        self.cum_misses = 0
        self.cum_fa = 0
        self.cum_clean = 0
        self.cum_scans = np.zeros(5, dtype=int)
        self.cum_hits_per_band = np.zeros(5, dtype=int)
        self.last_seen_step = np.full(5, -1, dtype=int)
        self.cum_reward = 0.0
        self.history_records = []

    def step(self) -> Dict[str, Any]:
        if self.current_step >= self.n_steps:
            return self.history_records[-1] if self.history_records else {}

        t = self.current_step

        # 1. LIVE GRU INFERENCE
        with torch.no_grad():
            probs, _ = self.gru(torch.tensor(self.hist_buf, dtype=torch.float32).unsqueeze(0))
            p_matrix = probs.squeeze(0).numpy()

        # 2. CONSTRUCT CAUSAL 170-D STATE
        s = np.zeros(170, dtype=np.float32)
        s[:100] = self.hist_buf.flatten()
        total_s = max(1, self.cum_scans.sum())
        for b in range(5):
            base = 100 + b * 14
            s[base + 8 : base + 12] = p_matrix[b]
            s[base + 0] = self.cum_scans[b] / total_s

        # 3. LIVE DQN INFERENCE
        with torch.no_grad():
            q_vals = self.dqn(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

        # 4. SELECT ACTION
        a_t = int(np.argmax(q_vals))
        self.cum_scans[a_t] += 1

        # 5. SIMULATED RECEIVER SCAN PHYSICAL INTERACTION
        gt_active = bool(self.gt[a_t, t] > 0.5)
        hidden_snr_val = float(self.snr_hidden[a_t, t])
        
        meas_snr = hidden_snr_val + float(np.random.normal(0, 0.5)) if gt_active else float(np.random.normal(-0.5, 0.5))
        
        if gt_active and hidden_snr_val >= 3.0:
            o_t = 1
            result_type = 'HIT'
            r_t = 0.96
            self.cum_hits += 1
            self.cum_hits_per_band[a_t] += 1
            self.last_seen_step[a_t] = t
        elif gt_active:
            o_t = 0
            result_type = 'MISS'
            r_t = -0.06
            self.cum_misses += 1
        elif np.random.rand() < 0.003:
            o_t = -1
            result_type = 'FALSE ALARM'
            r_t = -0.24
            self.cum_fa += 1
        else:
            o_t = 0
            result_type = 'CLEAN'
            r_t = -0.06
            self.cum_clean += 1

        self.cum_reward += r_t

        # 6. RUNNING METRICS
        total_active_so_far = float(np.sum(self.gt[:, :t+1]))
        scanned_active_so_far = float(self.cum_hits + self.cum_misses)
        scanned_inactive_so_far = float(self.cum_fa + self.cum_clean)

        pd_overall = float((self.cum_hits / total_active_so_far * 100.0) if total_active_so_far > 0 else 0.0)
        pd_scanned = float((self.cum_hits / scanned_active_so_far * 100.0) if scanned_active_so_far > 0 else 0.0)
        pfa = float((self.cum_fa / scanned_inactive_so_far * 100.0) if scanned_inactive_so_far > 0 else 0.0)
        sum_s = float(self.cum_scans.sum())
        sum_sq = float((self.cum_scans**2).sum())
        fairness = float(((sum_s**2) / (5.0 * sum_sq) * 100.0) if sum_sq > 0 else 100.0)

        metrics = {
            'step': int(t),
            'cum_hits': int(self.cum_hits),
            'cum_misses': int(self.cum_misses),
            'cum_fa': int(self.cum_fa),
            'cum_clean': int(self.cum_clean),
            'cum_reward': float(round(self.cum_reward, 2)),
            'scan_counts': [int(c) for c in self.cum_scans],
            'pd_overall': float(round(pd_overall, 2)),
            'pd_scanned': float(round(pd_scanned, 2)),
            'pfa': float(round(pfa, 3)),
            'fairness': float(round(fairness, 2)),
            'result_type': result_type
        }

        # 7. CAUSAL BUFFER UPDATE
        step_vec = np.zeros(10, dtype=np.float32)
        step_vec[2 * a_t] = float(o_t)
        step_vec[2 * a_t + 1] = 1.0
        self.hist_buf = np.roll(self.hist_buf, -1, axis=0)
        self.hist_buf[-1] = step_vec

        recency = (t - self.last_seen_step[a_t]) if self.last_seen_step[a_t] >= 0 else t
        band_hit_rate = (self.cum_hits_per_band[a_t] / self.cum_scans[a_t]) if self.cum_scans[a_t] > 0 else 0.0
        signals = {
            'selected_band': a_t,
            'max_q': float(round(float(q_vals[a_t]), 2)),
            'h1_prob': float(round(float(p_matrix[a_t, 0]) * 100, 1)),
            'h3_prob': float(round(float(p_matrix[a_t, 1]) * 100, 1)),
            'h5_prob': float(round(float(p_matrix[a_t, 2]) * 100, 1)),
            'h10_prob': float(round(float(p_matrix[a_t, 3]) * 100, 1)),
            'scan_share': float(round((self.cum_scans[a_t] / total_s) * 100, 1)),
            'band_hit_rate': float(round(band_hit_rate * 100, 1)),
            'recency': int(recency)
        }

        story = [
            f"1. Limited receiver scans Band {a_t} dynamically at time step t={t}.",
            f"2. Simulated receiver scan evaluates channel -> Reports {result_type} (SNR: {meas_snr:+.1f} dB).",
            f"3. Observation added to rolling 10-step history buffer H_t.",
            f"4. Live PyTorch GRU executes forward pass -> Forecasts future activity (H+1: {signals['h1_prob']:.1f}%).",
            f"5. Live PyTorch V73 DQN executes forward pass on 170-D causal scheduler state.",
            f"6. Band {a_t} evaluated with maximum state-action value Q={q_vals[a_t]:.2f}.",
            f"7. Next scan command queued for adaptive execution."
        ]

        record = {
            'step': int(t),
            'mode': 'LIVE_MODEL_INFERENCE',
            'action': int(a_t),
            'next_action': a_t,
            'observation': int(o_t),
            'result_type': result_type,
            'measured_snr': float(round(meas_snr, 2)),
            'reward': float(round(r_t, 3)),
            'signals': signals,
            'story': story,
            'q_values': [float(round(float(q), 3)) for q in q_vals],
            'gru_predictions': [[float(round(float(p), 4)) for p in row] for row in p_matrix],
            'metrics': metrics,
            'gt_active_now': [bool(self.gt[b, t] > 0.5) for b in range(5)],
            'scan_matrix_step': [1 if b == a_t else 0 for b in range(5)],
            'total_steps': int(self.n_steps)
        }

        clean_record = to_clean_json(record)
        self.history_records.append(clean_record)
        self.current_step += 1
        return clean_record


def load_results_data(results_dir: str) -> Dict[str, Any]:
    metrics_path = os.path.join(results_dir, 'checkpoint78_final_metrics.csv')
    comp_path = os.path.join(results_dir, 'checkpoint78_v73_comparisons.csv')
    ablation_delta_path = os.path.join(results_dir, 'checkpoint76_prediction_ablation_delta.csv')
    ablation_res_path = os.path.join(results_dir, 'checkpoint76_prediction_ablation_results.csv')

    df_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    df_comp = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()
    df_ab_delta = pd.read_csv(ablation_delta_path) if os.path.exists(ablation_delta_path) else pd.DataFrame()
    df_ab_res = pd.read_csv(ablation_res_path) if os.path.exists(ablation_res_path) else pd.DataFrame()

    res = {
        'final_metrics': df_metrics.to_dict(orient='records'),
        'v73_comparisons': df_comp.to_dict(orient='records'),
        'ablation_delta': df_ab_delta.to_dict(orient='records'),
        'ablation_summary': {
            'methods': ['V73_With_GRU', 'V73_No_GRU'],
            'count': len(df_ab_res)
        }
    }
    return to_clean_json(res)
