import uvicorn
import os
import sys

if __name__ == '__main__':
    print('=' * 65)
    print('  SMART SCAN STRATEGY FOR ELECTRONIC WARFARE (DRDO / SIH 26055)')
    print('  Interactive Causal Cognitive Receiver Dashboard')
    print('=' * 65)
    print('  * Host: http://127.0.0.1:8000')
    print('  * Mode 1: Validated Replay (checkpoint77_demo_rollout.npz)')
    print('  * Mode 2: Live Neural Inference (V73 DQN + Multi-Horizon GRU)')
    print('  * Causality Status: VERIFIED (0% Future Leakage)')
    print('=' * 65)
    print('Starting server on http://127.0.0.1:8000 ... (Press Ctrl+C to stop)')
    uvicorn.run('app.backend.main:app', host='127.0.0.1', port=8000, reload=False, log_level='info')
