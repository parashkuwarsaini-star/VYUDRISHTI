import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
import torch
import os
import sys

from app.backend.models import load_dqn_model, load_gru_model
from app.backend.engine import ValidatedReplayEngine, LiveInferenceEngine, load_results_data

class TestCausalSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = 'models'
        cls.demo_path = 'demo/checkpoint77_demo_rollout.npz'
        cls.results_dir = 'results'

        cls.dqn = load_dqn_model(os.path.join(cls.models_dir, 'best_dqn_scalable_v73.pt'))
        cls.gru, cls.gru_meta = load_gru_model(os.path.join(cls.models_dir, 'multi_horizon_activity_gru.pt'))
        cls.replay = ValidatedReplayEngine(cls.demo_path, cls.gru, cls.dqn)
        cls.live = LiveInferenceEngine(cls.demo_path, cls.gru, cls.dqn)

    def test_01_dqn_loaded(self):
        self.assertIsNotNone(self.dqn)
        dummy_in = torch.randn(1, 170)
        out = self.dqn(dummy_in)
        self.assertEqual(out.shape, (1, 5))
        total_params = sum(p.numel() for p in self.dqn.parameters())
        self.assertEqual(total_params, 85253)

    def test_02_gru_loaded(self):
        self.assertIsNotNone(self.gru)
        dummy_in = torch.randn(1, 10, 10)
        probs, logits = self.gru(dummy_in)
        self.assertEqual(probs.shape, (1, 5, 4))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_03_replay_steps(self):
        step0 = self.replay.get_step(0)
        self.assertEqual(step0['step'], 0)
        self.assertEqual(step0['action'], int(self.replay.actions[0]))
        self.assertIn(step0['result_type'], ['HIT', 'MISS', 'CLEAN', 'FALSE ALARM'])
        self.assertEqual(len(step0['q_values']), 5)
        self.assertEqual(len(step0['gru_predictions']), 5)
        self.assertEqual(len(step0['gru_predictions'][0]), 4)

        # Check end of rollout
        step499 = self.replay.get_step(499)
        self.assertEqual(step499['step'], 499)
        self.assertAlmostEqual(step499['metrics']['pd_overall'], 15.88, places=1)
        self.assertAlmostEqual(step499['metrics']['fairness'], 98.18, places=1)

    def test_04_live_inference_stepping(self):
        self.live.reset()
        self.assertEqual(self.live.current_step, 0)
        
        step0 = self.live.step()
        self.assertEqual(step0['step'], 0)
        self.assertEqual(step0['mode'], 'LIVE_MODEL_INFERENCE')
        self.assertIn(step0['action'], [0, 1, 2, 3, 4])
        self.assertEqual(len(step0['q_values']), 5)
        self.assertEqual(len(step0['gru_predictions']), 5)

        step1 = self.live.step()
        self.assertEqual(step1['step'], 1)
        self.assertEqual(self.live.current_step, 2)

    def test_05_benchmarks_loading(self):
        bdata = load_results_data(self.results_dir)
        self.assertIn('final_metrics', bdata)
        self.assertIn('v73_comparisons', bdata)
        self.assertIn('ablation_delta', bdata)
        self.assertEqual(len(bdata['final_metrics']), 6) # 6 methods
        self.assertEqual(len(bdata['v73_comparisons']), 4) # 4 pairwise deltas
        self.assertEqual(len(bdata['ablation_delta']), 8) # 8 ablation metrics

    def test_06_causal_firewall_verification(self):
        # Test that altering future ground truth has ZERO effect on scheduler decision at time t
        self.live.reset()
        # Take 10 steps under original environment
        for _ in range(10):
            rec = self.live.step()
        
        orig_q_step10 = list(rec['q_values'])
        orig_act_step10 = rec['action']

        # Now reset and scramble all future ground truth from step 11 to 500
        self.live.reset()
        fake_gt = self.live.gt.copy()
        fake_gt[:, 11:] = 1.0 - fake_gt[:, 11:]
        self.live.gt = fake_gt

        # Run up to step 10 again
        for _ in range(10):
            scrambled_rec = self.live.step()
        
        scrambled_q_step10 = list(scrambled_rec['q_values'])
        scrambled_act_step10 = scrambled_rec['action']

        # Future changes MUST have ZERO impact on past decisions
        self.assertEqual(orig_act_step10, scrambled_act_step10)
        for q1, q2 in zip(orig_q_step10, scrambled_q_step10):
            self.assertAlmostEqual(q1, q2, places=3)
        print('\nCausal Firewall Test: PASSED (Future ground truth changes have exactly ZERO impact on past/present decisions).')

if __name__ == '__main__':
    unittest.main()
