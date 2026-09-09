// Electronic Warfare Tactical HUD Dashboard Controller - Phase 3 Polished
let currentStep = 0;
const totalSteps = 500;
let isPlaying = false;
let playbackInterval = null;
let playbackSpeed = 5; // steps per second
let demoMode = 'VALIDATED_REPLAY';
let waterfallData = null;
let timelineAllCache = null;

// DOM Elements
const valTimeStep = document.getElementById('valTimeStep');
const valSelectedBand = document.getElementById('valSelectedBand');
const valResultBadge = document.getElementById('valResultBadge');
const valMeasuredSNR = document.getElementById('valMeasuredSNR');
const valReward = document.getElementById('valReward');
const valPdOverall = document.getElementById('valPdOverall');
const valPdScanned = document.getElementById('valPdScanned');
const valPfa = document.getElementById('valPfa');
const valFairness = document.getElementById('valFairness');

const bigResultBadge = document.getElementById('bigResultBadge');
const subResultText = document.getElementById('subResultText');
const meterSnrVal = document.getElementById('meterSnrVal');
const snrBarFill = document.getElementById('snrBarFill');

const timelineSlider = document.getElementById('timelineSlider');
const sliderStepVal = document.getElementById('sliderStepVal');
const btnPlayPause = document.getElementById('btnPlayPause');
const modeNotice = document.getElementById('modeNotice');

const gruTableBody = document.getElementById('gruTableBody');
const bannerSelectedBand = document.getElementById('bannerSelectedBand');
const bannerMaxQ = document.getElementById('bannerMaxQ');
const qBarsListContainer = document.getElementById('qBarsListContainer');
const whySignalsContent = document.getElementById('whySignalsContent');
const storyPanelContent = document.getElementById('storyPanelContent');

const canvas = document.getElementById('waterfallCanvas');
const ctx = canvas.getContext('2d');

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
  await Promise.all([
    fetchWaterfallData(),
    fetchTimelineAll(),
    fetchBenchmarks()
  ]);
  renderWaterfall();
  loadStep(0);
});

// Switch Demo Mode
function setDemoMode(mode) {
  demoMode = mode;
  document.getElementById('btnModeReplay').classList.toggle('active', mode === 'VALIDATED_REPLAY');
  document.getElementById('btnModeLive').classList.toggle('active', mode === 'LIVE_MODEL_INFERENCE');
  
  if (mode === 'LIVE_MODEL_INFERENCE') {
    modeNotice.textContent = 'Mode: Live Neural Inference';
    modeNotice.style.color = '#10b981';
    pausePlayback();
    fetch('/api/live/reset', { method: 'POST' });
  } else {
    modeNotice.textContent = 'Mode: Demo Mode';
    modeNotice.style.color = '#94a3b8';
  }
  loadStep(currentStep);
}

// Switch Navigation Tabs
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
  
  const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
  if (activeBtn) activeBtn.classList.add('active');
  
  const content = document.getElementById(tabId);
  if (content) content.classList.add('active');
  
  if (tabId === 'tabLiveOps') {
    renderWaterfall();
  }
}

// Fetch Precomputed Waterfall Arrays
async function fetchWaterfallData() {
  try {
    const res = await fetch('/api/waterfall');
    waterfallData = await res.json();
  } catch (err) {
    console.error('Failed to load waterfall cache:', err);
  }
}

// Fetch Full Replay Timeline for Instant 0ms Scrubbing & 60 FPS Playback
async function fetchTimelineAll() {
  try {
    const res = await fetch('/api/timeline_all');
    timelineAllCache = await res.json();
  } catch (err) {
    console.error('Failed to pre-cache timeline:', err);
  }
}

// Load Timestep Data
async function loadStep(t) {
  currentStep = Math.max(0, Math.min(t, totalSteps - 1));
  timelineSlider.value = currentStep;
  sliderStepVal.textContent = `t=${currentStep}`;

  if (demoMode === 'VALIDATED_REPLAY' && timelineAllCache && timelineAllCache[currentStep]) {
    updateUI(timelineAllCache[currentStep]);
    renderWaterfall();
  } else {
    try {
      const res = await fetch(`/api/step/${currentStep}?mode=${demoMode}`);
      const data = await res.json();
      updateUI(data);
      renderWaterfall();
    } catch (err) {
      console.error('Failed to fetch step data:', err);
    }
  }
}

// Update UI Telemetry, Panels, and Story
function updateUI(data) {
  if (!data) return;

  valTimeStep.textContent = `t = ${data.step} / ${totalSteps}`;
  valSelectedBand.textContent = `Band ${data.action}`;
  valReward.textContent = (data.reward > 0 ? '+' : '') + Number(data.reward).toFixed(2);
  
  // Detection Badge
  const rType = data.result_type;
  if (rType === 'HIT') {
    valResultBadge.innerHTML = '<span class="badge-hit">HIT (+1)</span>';
    bigResultBadge.className = 'res-badge-large badge-hit';
    bigResultBadge.textContent = '● HIT (+1)';
    subResultText.textContent = 'Active Emitter Intercepted';
  } else if (rType === 'MISS') {
    valResultBadge.innerHTML = '<span class="badge-miss">MISS (0)</span>';
    bigResultBadge.className = 'res-badge-large badge-miss';
    bigResultBadge.textContent = '● MISS (0)';
    subResultText = 'Active Emitter Present, Low SNR';
  } else if (rType === 'FALSE ALARM') {
    valResultBadge.innerHTML = '<span class="badge-fa">FALSE ALARM (-1)</span>';
    bigResultBadge.className = 'res-badge-large badge-fa';
    bigResultBadge.textContent = '● FALSE ALARM (-1)';
    subResultText = 'Noise Threshold Exceeded';
  } else {
    valResultBadge.innerHTML = '<span class="badge-clean">CLEAN (0)</span>';
    bigResultBadge.className = 'res-badge-large badge-clean';
    bigResultBadge.textContent = '● CLEAN (0)';
    subResultText = 'Channel Quiet (Noise Floor)';
  }

  // Measured SNR Gauge
  valMeasuredSNR.textContent = `${Number(data.measured_snr).toFixed(1)} dB`;
  meterSnrVal.textContent = `${Number(data.measured_snr).toFixed(1)} dB`;
  const clampedSnr = Math.max(-15, Math.min(35, data.measured_snr));
  const snrPct = ((clampedSnr + 15) / 50) * 100;
  snrBarFill.style.width = `${snrPct}%`;

  // Explicit Percentage Formatting for Metrics
  if (data.metrics) {
    valPdOverall.textContent = `${Number(data.metrics.pd_overall).toFixed(2)} %`;
    valPdScanned.textContent = `${Number(data.metrics.pd_scanned).toFixed(2)} %`;
    valPfa.textContent = `${Number(data.metrics.pfa).toFixed(3)} %`;
    valFairness.textContent = `${Number(data.metrics.fairness).toFixed(2)} %`;
  }

  // Multi-Horizon GRU Table (5x4)
  if (data.gru_predictions) {
    let gruHtml = '';
    for (let b = 0; b < 5; b++) {
      gruHtml += `<tr><td><strong>Band ${b}</strong></td>`;
      for (let h = 0; h < 4; h++) {
        const prob = data.gru_predictions[b][h];
        const pct = Math.round(prob * 100);
        const alpha = Math.min(1.0, Math.max(0.08, prob));
        const bg = `rgba(0, 240, 255, ${alpha * 0.45})`;
        gruHtml += `<td class="gru-cell" style="background: ${bg}; color: ${prob > 0.35 ? '#00f0ff' : '#cbd5e1'};">${pct}%</td>`;
      }
      gruHtml += `</tr>`;
    }
    gruTableBody.innerHTML = gruHtml;
  }

  // DQN Q-Value Horizontal Bars
  if (data.q_values) {
    const nextBand = data.next_action !== undefined ? data.next_action : data.action;
    bannerSelectedBand.textContent = `SCAN BAND ${nextBand}`;
    bannerMaxQ.textContent = `Q = ${Number(data.q_values[nextBand]).toFixed(2)}`;

    // Normalize Q-values for bar width
    const minQ = Math.min(...data.q_values);
    const maxQ = Math.max(...data.q_values);
    const rangeQ = Math.max(0.1, maxQ - minQ);

    let barsHtml = '';
    for (let b = 0; b < 5; b++) {
      const isSel = (b === nextBand);
      const q = Number(data.q_values[b]);
      const widthPct = Math.max(10, Math.min(100, ((q - minQ) / rangeQ) * 100));
      barsHtml += `
        <div class="q-bar-row ${isSel ? 'selected' : ''}">
          <span>B${b}</span>
          <div class="q-bar-track">
            <div class="q-bar-fill" style="width: ${widthPct}%;"></div>
          </div>
          <span style="text-align: right;">${q.toFixed(2)}${isSel ? ' ◄' : ''}</span>
        </div>
      `;
    }
    qBarsListContainer.innerHTML = barsHtml;

    // Why Signals
    if (data.signals) {
      const s = data.signals;
      whySignalsContent.innerHTML = `
        • <strong>Band ${s.selected_band}</strong> evaluated with highest state-action value (Q=${s.max_q.toFixed(2)}).<br>
        • GRU activity forecast: <strong>H+1: ${s.h1_prob}%</strong> | <strong>H+3: ${s.h3_prob}%</strong>.<br>
        • Historical scan share allocated: <strong>${s.scan_share}%</strong> | Band hit rate: <strong>${s.band_hit_rate}%</strong>.<br>
        • Recency (steps since last observation): <strong>${s.recency} timesteps</strong>.
      `;
    }
  }

  // Demo Story Narrative
  if (data.story && Array.isArray(data.story)) {
    let storyHtml = '';
    data.story.forEach(line => {
      const parts = line.split('. ');
      const stepTag = parts[0];
      const desc = parts.slice(1).join('. ');
      storyHtml += `
        <div class="story-line">
          <span class="story-num">${stepTag}.</span>
          <span>${desc}</span>
        </div>
      `;
    });
    storyPanelContent.innerHTML = storyHtml;
  }
}

// Hero Canvas Waterfall Renderer
function renderWaterfall() {
  if (!waterfallData) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const nBands = 5;
  const totalLayers = 2; // Ground Truth top, Receiver Observation bottom
  const layerHeight = (h - 26) / totalLayers;
  const rowHeight = layerHeight / nBands;
  const colWidth = (w - 45) / totalSteps;
  const xOffset = 40; // Left margin for Band 0..4 text labels

  // 1. Draw Simulation Truth Layer (Top)
  for (let b = 0; b < nBands; b++) {
    const y = b * rowHeight + 4;
    
    // Draw row background and label
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`B${b}`, xOffset - 6, y + rowHeight - 3);

    for (let t = 0; t < totalSteps; t++) {
      if (waterfallData.ground_truth[b][t] > 0.5) {
        ctx.fillStyle = '#dc2626'; // Deep Red ground truth emitter burst
        ctx.fillRect(xOffset + t * colWidth, y, Math.max(1, colWidth), rowHeight - 1);
      }
    }
  }

  // 2. Middle Boundary Line
  const boundaryY = layerHeight + 8;
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(xOffset, boundaryY);
  ctx.lineTo(w, boundaryY);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Draw Causal Receiver Observation Layer (Bottom)
  const obsStartY = boundaryY + 8;
  for (let b = 0; b < nBands; b++) {
    const y = obsStartY + b * rowHeight;
    
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`B${b}`, xOffset - 6, y + rowHeight - 3);

    // Draw unscanned dark base
    ctx.fillStyle = '#060a15';
    ctx.fillRect(xOffset, y, (w - xOffset), rowHeight - 1);
  }

  for (let t = 0; t <= currentStep; t++) {
    const act = waterfallData.actions[t];
    const obs = waterfallData.observations[t];
    const y = obsStartY + act * rowHeight;

    if (obs === 1) {
      ctx.fillStyle = '#10b981'; // Green HIT
    } else if (obs === -1) {
      ctx.fillStyle = '#ef4444'; // Red FA
    } else if (waterfallData.ground_truth[act][t] > 0.5) {
      ctx.fillStyle = '#f59e0b'; // Amber MISS
    } else {
      ctx.fillStyle = '#334155'; // Slate CLEAN
    }
    ctx.fillRect(xOffset + t * colWidth, y, Math.max(1, colWidth), rowHeight - 1);
  }

  // 4. Draw Current Time Cursor Line
  const cursorX = xOffset + currentStep * colWidth;
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();

  // Highlight Current Scanned Band in Observation Layer
  if (currentStep < totalSteps) {
    const currentAction = waterfallData.actions[currentStep];
    const highlightY = obsStartY + currentAction * rowHeight;
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(cursorX - 2, highlightY - 1, colWidth + 4, rowHeight);
  }
}

// Transport Controls
function togglePlayPause() {
  if (isPlaying) {
    pausePlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  isPlaying = true;
  btnPlayPause.textContent = '⏸ PAUSE';
  btnPlayPause.classList.remove('primary');
  btnPlayPause.style.background = '#f59e0b';

  playbackInterval = setInterval(() => {
    if (currentStep >= totalSteps - 1) {
      pausePlayback();
      return;
    }
    loadStep(currentStep + 1);
  }, 1000 / playbackSpeed);
}

function pausePlayback() {
  isPlaying = false;
  btnPlayPause.textContent = '▶ START';
  btnPlayPause.classList.add('primary');
  btnPlayPause.style.background = '#2563eb';
  if (playbackInterval) clearInterval(playbackInterval);
}

function resetPlayback() {
  pausePlayback();
  if (demoMode === 'LIVE_MODEL_INFERENCE') {
    fetch('/api/live/reset', { method: 'POST' });
  }
  loadStep(0);
}

function stepForward() {
  pausePlayback();
  if (currentStep < totalSteps - 1) {
    loadStep(currentStep + 1);
  }
}

function stepBackward() {
  pausePlayback();
  if (currentStep > 0) {
    loadStep(currentStep - 1);
  }
}

function onSliderInput(val) {
  pausePlayback();
  loadStep(parseInt(val, 10));
}

function setPlaybackSpeed(speed) {
  playbackSpeed = parseFloat(speed);
  if (isPlaying) {
    pausePlayback();
    startPlayback();
  }
}

// Fetch Benchmark & Ablation Data
async function fetchBenchmarks() {
  try {
    const res = await fetch('/api/benchmarks');
    const data = await res.json();
    populateBenchmarkTable(data.final_metrics);
    populateComparisonsTable(data.v73_comparisons);
    populateAblationGrid(data.ablation_delta);
  } catch (err) {
    console.error('Failed to load benchmarks:', err);
  }
}

function populateBenchmarkTable(metrics) {
  const tbody = document.getElementById('benchmarkTableBody');
  if (!tbody || !metrics) return;

  let html = '';
  metrics.forEach(row => {
    const isV73 = row.method.includes('V73');
    html += `
      <tr class="${isV73 ? 'highlight-v73' : ''}">
        <td><strong>${row.method}</strong></td>
        <td>${row.reward_mean.toFixed(2)} &plusmn; ${row.reward_std.toFixed(1)}</td>
        <td>${row.hits_mean.toFixed(1)}</td>
        <td>${(row.Pd_mean * 100).toFixed(2)} %</td>
        <td>${(row.scanned_Pd_mean * 100).toFixed(2)} %</td>
        <td>${(row.Pfa_mean * 100).toFixed(2)} %</td>
        <td>${(row.interception_mean * 100).toFixed(2)} %</td>
        <td>${row.delay_mean.toFixed(2)}</td>
        <td>${(row.fairness_mean * 100).toFixed(2)} %</td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

function populateComparisonsTable(comps) {
  const tbody = document.getElementById('comparisonsTableBody');
  if (!tbody || !comps) return;

  let html = '';
  comps.forEach(row => {
    html += `
      <tr>
        <td><strong>${row['Comparison']}</strong></td>
        <td style="color: ${row['Reward Δ'] > 0 ? '#10b981' : '#f87171'}; font-weight: bold;">
          ${row['Reward Δ'] > 0 ? '+' : ''}${row['Reward Δ'].toFixed(2)}
        </td>
        <td>${row['Pd Δ (pp)'] > 0 ? '+' : ''}${row['Pd Δ (pp)'].toFixed(2)} pp</td>
        <td>${row['Pfa Δ (pp)'].toFixed(4)} pp</td>
        <td>${row['Intercept Δ (pp)'] > 0 ? '+' : ''}${row['Intercept Δ (pp)'].toFixed(2)} pp</td>
        <td>${row['Delay Δ'].toFixed(3)}</td>
        <td style="color: ${row['Fairness Δ (pp)'] > 0 ? '#10b981' : '#f87171'};">
          ${row['Fairness Δ (pp)'] > 0 ? '+' : ''}${row['Fairness Δ (pp)'].toFixed(2)} pp
        </td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

function populateAblationGrid(ablationDeltas) {
  const container = document.getElementById('ablationGridContainer');
  if (!container || !ablationDeltas) return;

  let html = '';
  ablationDeltas.forEach(item => {
    const isGain = item.delta_with_minus_without > 0;
    const deltaStr = (isGain ? '+' : '') + Number(item.delta_with_minus_without).toFixed(2);
    html += `
      <div class="ablation-card">
        <div class="m-title">${item.metric.replace(/_/g, ' ')}</div>
        <div class="d-val" style="color: ${isGain ? '#10b981' : (item.metric === 'pfa' ? '#10b981' : '#f59e0b')};">${deltaStr}</div>
        <div class="c-vals">With GRU: ${Number(item.with_gru).toFixed(2)} | W/O: ${Number(item.without_gru).toFixed(2)}</div>
      </div>
    `;
  });
  container.innerHTML = html;
}
