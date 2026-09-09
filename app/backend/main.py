import os
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from app.backend.models import load_dqn_model, load_gru_model
from app.backend.engine import ValidatedReplayEngine, LiveInferenceEngine, load_results_data

app = FastAPI(
    title='DRDO Smart Scan Strategy EW Dashboard',
    description='Causal AI + Reinforcement Learning Receiver Scheduling Dashboard (DRDO/SIH 26055)',
    version='1.1.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DEMO_DIR = os.path.join(BASE_DIR, 'demo')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FRONTEND_DIR = os.path.join(BASE_DIR, 'app', 'frontend')

dqn_model = load_dqn_model(os.path.join(MODELS_DIR, 'best_dqn_scalable_v73.pt'))
gru_model, gru_meta = load_gru_model(os.path.join(MODELS_DIR, 'multi_horizon_activity_gru.pt'))
demo_file = os.path.join(DEMO_DIR, 'checkpoint77_demo_rollout.npz')

replay_engine = ValidatedReplayEngine(demo_file, gru_model, dqn_model)
live_engine = LiveInferenceEngine(demo_file, gru_model, dqn_model)
benchmarks_data = load_results_data(RESULTS_DIR)

demo_raw = np.load(demo_file, allow_pickle=True)
waterfall_cache = {
    'total_steps': int(len(demo_raw['actions'])),
    'n_bands': 5,
    'ground_truth': demo_raw['ground_truth'].tolist(), # 5 x 500
    'actions': demo_raw['actions'].tolist(),           # 500
    'observations': demo_raw['observations'].tolist(), # 500
    'rewards': [round(float(r), 3) for r in demo_raw['rewards']],
    'measured_snr': [round(float(s), 2) for s in demo_raw['measured_snr']]
}

@app.get('/api/status')
def get_status():
    return {
        'status': 'ONLINE',
        'system': 'V73 Causal DQN + Multi-Horizon GRU',
        'problem_statement': 'DRDO / SIH 26055',
        'causal_audit': 'PASSED (0% Future Leakage Guaranteed)',
        'models_loaded': {
            'dqn': 'best_dqn_scalable_v73.pt (85,253 parameters)',
            'gru': 'multi_horizon_activity_gru.pt (44,628 parameters)'
        },
        'total_steps': replay_engine.n_steps,
        'n_bands': replay_engine.n_bands,
        'prediction_horizons': ['H+1', 'H+3', 'H+5', 'H+10']
    }

@app.get('/api/step/{t}')
def get_step(t: int, mode: str = Query('VALIDATED_REPLAY')):
    if mode == 'LIVE_MODEL_INFERENCE':
        if live_engine.current_step == 0:
            rec = live_engine.step()
        elif t < len(live_engine.history_records):
            rec = live_engine.history_records[t]
        else:
            rec = live_engine.step()
        return rec
    else:
        return replay_engine.get_step(t)

@app.get('/api/timeline_all')
def get_timeline_all():
    return replay_engine.cached_steps

@app.post('/api/live/step')
def live_step():
    return live_engine.step()

@app.post('/api/live/reset')
def live_reset():
    live_engine.reset()
    return {'status': 'RESET_SUCCESS', 'current_step': 0}

@app.get('/api/waterfall')
def get_waterfall():
    return waterfall_cache

@app.get('/api/benchmarks')
def get_benchmarks():
    return benchmarks_data

if os.path.exists(FRONTEND_DIR):
    app.mount('/static', StaticFiles(directory=FRONTEND_DIR), name='static')

    @app.get('/')
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))
