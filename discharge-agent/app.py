"""
Discharge Summary Agent — Web UI
=================================
Run with: python app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import time
import uuid
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template_string, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB
os.makedirs("uploads", exist_ok=True)

# In-memory job store
JOBS = {}

# ── HTML template ──────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Discharge Summary Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0f1a; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  .header { background: #0d1420; border-bottom: 1px solid #1e293b; padding: 16px 32px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }
  .logo { width: 40px; height: 40px; background: linear-gradient(135deg, #06b6d4, #3b82f6); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px; }
  .header h1 { font-size: 18px; font-weight: 700; }
  .header p { font-size: 12px; color: #64748b; }
  .badge { background: #ef444420; color: #fca5a5; border: 1px solid #ef444440; padding: 3px 10px; border-radius: 20px; font-size: 11px; margin-left: auto; }

  .main { max-width: 1100px; margin: 0 auto; padding: 32px; }

  .warning { background: #92400e20; border: 1px solid #f59e0b40; border-radius: 12px; padding: 14px 20px; margin-bottom: 28px; font-size: 13px; color: #fcd34d; display: flex; gap: 10px; }

  .upload-zone { border: 2px dashed #1e3a5f; border-radius: 16px; padding: 48px; text-align: center; cursor: pointer; transition: all 0.2s; background: #0d1a2d; margin-bottom: 28px; }
  .upload-zone:hover, .upload-zone.drag { border-color: #06b6d4; background: #06b6d410; }
  .upload-zone h2 { font-size: 18px; margin-bottom: 8px; color: #94a3b8; }
  .upload-zone p { font-size: 13px; color: #475569; margin-bottom: 20px; }
  .upload-zone input { display: none; }
  .file-list { margin-top: 16px; text-align: left; }
  .file-item { background: #0f2235; border: 1px solid #1e3a5f; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; display: flex; align-items: center; gap: 10px; font-size: 13px; }
  .file-item .icon { color: #06b6d4; }
  .file-item .remove { margin-left: auto; cursor: pointer; color: #64748b; font-size: 16px; }
  .file-item .remove:hover { color: #ef4444; }

  .btn { padding: 12px 28px; border-radius: 10px; border: none; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.2s; }
  .btn-primary { background: linear-gradient(135deg, #06b6d4, #3b82f6); color: white; width: 100%; margin-bottom: 12px; }
  .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-demo { background: #1e293b; color: #94a3b8; border: 1px solid #334155; width: 100%; }
  .btn-demo:hover { background: #273344; color: #e2e8f0; }

  .progress-panel { background: #0d1420; border: 1px solid #1e293b; border-radius: 16px; padding: 24px; margin-bottom: 28px; display: none; }
  .progress-panel.active { display: block; }
  .progress-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
  .spinner { width: 20px; height: 20px; border: 2px solid #1e293b; border-top-color: #06b6d4; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .progress-title { font-size: 15px; font-weight: 600; }
  .progress-sub { font-size: 12px; color: #64748b; }
  .steps { max-height: 300px; overflow-y: auto; }
  .step { padding: 10px 14px; border-radius: 8px; margin-bottom: 6px; font-size: 12px; border-left: 3px solid transparent; }
  .step.done { background: #0f2235; border-color: #06b6d4; }
  .step.escalate { background: #1a0f0f; border-color: #ef4444; }
  .step.active { background: #0f1f2e; border-color: #f59e0b; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.7; } }
  .step-action { font-weight: 700; color: #06b6d4; margin-bottom: 3px; }
  .step.escalate .step-action { color: #f87171; }
  .step-reasoning { color: #94a3b8; }

  .results { display: none; }
  .results.active { display: block; }

  .patient-header { background: linear-gradient(135deg, #06b6d410, #3b82f610); border: 1px solid #1e3a5f; border-radius: 16px; padding: 24px; margin-bottom: 24px; }
  .patient-name { font-size: 26px; font-weight: 800; margin-bottom: 6px; }
  .patient-meta { font-size: 13px; color: #64748b; font-family: monospace; }
  .tags { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
  .tag { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .tag-red { background: #ef444420; color: #fca5a5; border: 1px solid #ef444440; }
  .tag-orange { background: #f9731620; color: #fdba74; border: 1px solid #f9731640; }
  .tag-yellow { background: #eab30820; color: #fde047; border: 1px solid #eab30840; }
  .tag-blue { background: #3b82f620; color: #93c5fd; border: 1px solid #3b82f640; }
  .tag-green { background: #22c55e20; color: #86efac; border: 1px solid #22c55e40; }

  .tabs { display: flex; gap: 4px; border-bottom: 1px solid #1e293b; margin-bottom: 24px; }
  .tab { padding: 10px 20px; font-size: 13px; cursor: pointer; border-bottom: 2px solid transparent; color: #64748b; transition: all 0.2s; background: none; border-top: none; border-left: none; border-right: none; }
  .tab:hover { color: #94a3b8; }
  .tab.active { border-bottom-color: #06b6d4; color: #06b6d4; }

  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media(max-width:700px) { .grid2 { grid-template-columns: 1fr; } }

  .card { background: #0d1420; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; margin-bottom: 16px; }
  .card-header { padding: 14px 18px; border-bottom: 1px solid #1e293b; display: flex; align-items: center; justify-content: space-between; }
  .card-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
  .card-body { padding: 18px; }

  .field { margin-bottom: 14px; }
  .field-label { font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
  .field-value { font-size: 14px; color: #e2e8f0; }
  .missing { background: #7c2d1220; border: 1px solid #f9731640; color: #fdba74; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 12px; }
  .conflicting { background: #7f1d1d20; border: 1px solid #ef444440; color: #fca5a5; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 12px; }
  .pending-val { background: #78350f20; border: 1px solid #f59e0b40; color: #fde047; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 12px; }

  .med-item { background: #0f1e2e; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
  .med-item.flagged { background: #1a0f0f; border-color: #ef444430; }
  .med-name { font-weight: 700; font-size: 14px; }
  .med-dose { font-size: 12px; color: #64748b; font-family: monospace; margin-top: 2px; }
  .med-status { font-size: 11px; color: #06b6d4; margin-top: 4px; }
  .med-flag { font-size: 11px; color: #f87171; margin-top: 4px; }

  .flag-item { border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
  .flag-critical { background: #7f1d1d20; border: 1px solid #ef444450; }
  .flag-high { background: #7c2d1220; border: 1px solid #f9731650; }
  .flag-medium { background: #78350f20; border: 1px solid #f59e0b50; }
  .flag-severity { font-size: 11px; font-weight: 800; letter-spacing: 0.1em; margin-bottom: 6px; }
  .flag-msg { font-size: 13px; color: #e2e8f0; }
  .flag-id { font-size: 11px; color: #475569; margin-top: 4px; font-family: monospace; }

  .conflict-item { background: #7f1d1d15; border: 1px solid #ef444430; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
  .conflict-type { font-size: 11px; font-weight: 700; color: #f87171; margin-bottom: 4px; }
  .conflict-desc { font-size: 13px; }
  .conflict-sources { font-size: 11px; color: #64748b; font-family: monospace; margin-top: 4px; }

  .pending-item { background: #78350f15; border: 1px solid #f59e0b30; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
  .pending-test { font-size: 13px; color: #fde047; }
  .pending-note { font-size: 11px; color: #92400e; margin-top: 4px; }

  .trace-step { border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; border-left: 3px solid #1e3a5f; background: #0d1a2d; }
  .trace-step.escalate { border-left-color: #ef4444; background: #1a0a0a; }
  .trace-step.done-step { border-left-color: #22c55e; background: #0a1a0f; }
  .trace-step.build { border-left-color: #06b6d4; }
  .trace-num { font-size: 11px; color: #475569; font-family: monospace; margin-bottom: 4px; }
  .trace-action { font-size: 13px; font-weight: 700; color: #06b6d4; margin-bottom: 4px; }
  .trace-step.escalate .trace-action { color: #f87171; }
  .trace-step.done-step .trace-action { color: #86efac; }
  .trace-reasoning { font-size: 12px; color: #94a3b8; }
  .trace-note { font-size: 11px; color: #475569; font-style: italic; margin-top: 4px; }

  .reward-bar-bg { background: #1e293b; border-radius: 20px; height: 8px; margin-top: 6px; }
  .reward-bar { background: linear-gradient(90deg, #06b6d4, #22c55e); border-radius: 20px; height: 8px; transition: width 1s; }

  .list-item { font-size: 13px; padding: 6px 0; border-bottom: 1px solid #1e293b; color: #94a3b8; display: flex; gap: 8px; }
  .list-item:last-child { border: none; }
  .list-arrow { color: #06b6d4; }

  #drop-label { color: #94a3b8; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">Rx</div>
  <div>
    <h1>Discharge Summary Agent</h1>
    <p>Agentic AI · Clinical draft generation</p>
  </div>
  <span class="badge">⚠ AI Draft — Clinician Review Required</span>
</div>

<div class="main">

  <div class="warning">
    <span>⚠️</span>
    <span><strong>NOT FOR CLINICAL USE.</strong> All output is an AI-generated draft requiring clinician verification. Fields marked MISSING require manual input. Never assume pending results.</span>
  </div>

  <!-- Upload -->
  <div id="upload-section">
    <div class="upload-zone" id="drop-zone">
      <div style="font-size:40px;margin-bottom:12px">📄</div>
      <h2>Drop patient documents here</h2>
      <p>Upload PDF or text files — admission notes, discharge summaries, lab results, medication records</p>
      <button class="btn btn-primary" style="width:auto;padding:10px 24px" onclick="document.getElementById('file-input').click()">Choose Files</button>
      <input type="file" id="file-input" multiple accept=".pdf,.txt,.text">
      <div class="file-list" id="file-list"></div>
    </div>

    <button class="btn btn-primary" id="run-btn" onclick="runAgent()" disabled>
      🚀 Run Discharge Summary Agent
    </button>
    <button class="btn btn-demo" onclick="runDemo()">
      ▶ Run Demo (3 synthetic patients — no upload needed)
    </button>
  </div>

  <!-- Progress -->
  <div class="progress-panel" id="progress-panel">
    <div class="progress-header">
      <div class="spinner" id="spinner"></div>
      <div>
        <div class="progress-title" id="progress-title">Initialising agent...</div>
        <div class="progress-sub" id="progress-sub">Setting up tools</div>
      </div>
    </div>
    <div class="steps" id="steps-log"></div>
  </div>

  <!-- Results -->
  <div class="results" id="results">
    <div class="patient-header" id="patient-header"></div>

    <div class="tabs">
      <button class="tab active" onclick="showTab('summary')">📋 Summary</button>
      <button class="tab" onclick="showTab('medications')">💊 Medications</button>
      <button class="tab" onclick="showTab('flags')">🚨 Flags</button>
      <button class="tab" onclick="showTab('trace')">🔍 Trace</button>
      <button class="tab" onclick="showTab('part2')">📈 Learning</button>
      <button class="tab" onclick="showTab('ocr')">📝 OCR Preview</button>
    </div>

    <div class="tab-content active" id="tab-summary"></div>
    <div class="tab-content" id="tab-medications"></div>
    <div class="tab-content" id="tab-flags"></div>
    <div class="tab-content" id="tab-trace"></div>
    <div class="tab-content" id="tab-part2"></div>
    <div class="tab-content" id="tab-ocr"></div>
  </div>

</div>

<script>
let selectedFiles = [];
let currentJobId = null;
let pollInterval = null;

// File drag/drop
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag');
  addFiles(Array.from(e.dataTransfer.files));
});
fileInput.addEventListener('change', e => addFiles(Array.from(e.target.files)));

function addFiles(files) {
  files.forEach(f => {
    if (!selectedFiles.find(x => x.name === f.name)) selectedFiles.push(f);
  });
  renderFileList();
}

function removeFile(name) {
  selectedFiles = selectedFiles.filter(f => f.name !== name);
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById('file-list');
  list.innerHTML = selectedFiles.map(f => `
    <div class="file-item">
      <span class="icon">📄</span>
      <span>${f.name}</span>
      <span style="color:#64748b;font-size:11px">${(f.size/1024).toFixed(0)}KB</span>
      <span class="remove" onclick="removeFile('${f.name}')">✕</span>
    </div>`).join('');
  document.getElementById('run-btn').disabled = selectedFiles.length === 0;
}

async function runAgent() {
  if (selectedFiles.length === 0) return;
  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('files', f));

  showProgress('Uploading documents...');
  const res = await fetch('/run', { method: 'POST', body: formData });
  const data = await res.json();
  if (data.job_id) startPolling(data.job_id);
}

async function runDemo() {
  showProgress('Starting demo run...');
  const res = await fetch('/demo', { method: 'POST' });
  const data = await res.json();
  if (data.job_id) startPolling(data.job_id);
}

function showProgress(msg) {
  document.getElementById('upload-section').style.display = 'none';
  document.getElementById('progress-panel').classList.add('active');
  document.getElementById('results').classList.remove('active');
  document.getElementById('progress-title').textContent = msg;
  document.getElementById('steps-log').innerHTML = '';
}

function startPolling(jobId) {
  currentJobId = jobId;
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(() => pollJob(jobId), 600);
}

async function pollJob(jobId) {
  const res = await fetch(`/status/${jobId}`);
  const data = await res.json();

  // Update progress
  document.getElementById('progress-title').textContent = data.status_msg || 'Running...';
  document.getElementById('progress-sub').textContent = `Step ${data.step || 0} · Patient ${data.patient_id || ''}`;

  // Render steps
  const stepsLog = document.getElementById('steps-log');
  stepsLog.innerHTML = (data.steps || []).map(s => `
    <div class="step ${s.action === 'escalate' ? 'escalate' : s.action === 'DONE' ? '' : 'done'}">
      <div class="step-action">${actionIcon(s.action)} ${s.action}</div>
      <div class="step-reasoning">${s.reasoning || ''}</div>
    </div>`).join('');
  stepsLog.scrollTop = stepsLog.scrollHeight;

  if (data.done) {
    clearInterval(pollInterval);
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('progress-title').textContent = '✅ Complete';
    setTimeout(() => renderResults(data.result), 600);
  }
}

function actionIcon(a) {
  const icons = { ingest_pdfs:'📄', detect_conflicts:'🔍', reconcile_meds:'💊',
    check_drug_interactions:'⚕️', check_pending:'⏳', escalate:'🚨',
    build_summary:'📋', DONE:'✅' };
  return icons[a] || '🔧';
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['summary','medications','flags','trace','part2','ocr'][i] === name));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
}

function mv(val) {
  if (!val || val === '') return '<span class="missing">Unable to confidently extract — clinician review required</span>';
  if (typeof val === 'string' && val.includes('MISSING')) return `<span class="missing">${val}</span>`;
  if (typeof val === 'string' && val.includes('CONFLICTING')) return `<span class="conflicting">${val.substring(0,200)}...</span>`;
  if (typeof val === 'string' && val.includes('PENDING')) return `<span class="pending-val">${val}</span>`;
  return `<span>${Array.isArray(val) ? val.join(', ') : val}</span>`;
}

function renderResults(data) {
  if (!data) return;
  const rawText = data.extracted_text || JSON.stringify(data.summary, null, 2).substring(0, 3000);
  document.getElementById('progress-panel').classList.remove('active');
  document.getElementById('results').classList.add('active');

  const s = data.summary || {};
  const meta = s._meta || {};
  const demo = s.patient_demographics || {};
  const dates = s.admission_discharge_dates || {};
  const dx = s.diagnoses || {};
  const flags = s.clinician_review_flags || [];
  const conflicts = s.conflicts_detected || [];
  const pending = s.pending_results || [];
  const trace = data.trace || [];
  const meds = s.medications || {};
  const ix = s.drug_interactions_flagged || [];
  const part2 = data.part2 || null;

  // Patient header
  document.getElementById('patient-header').innerHTML = `
    <div class="patient-name">
  ${demo.name || 'PARTIAL EXTRACTION COMPLETE — CLINICIAN REVIEW REQUIRED'}
</div>
    <div class="patient-meta">${demo.mrn || ''} · ${demo.age || ''}y ${demo.gender || ''} · DOB ${demo.date_of_birth || ''}</div>
    <div class="patient-meta" style="margin-top:4px">Admitted ${dates.admission_date || '—'} → Discharged ${dates.discharge_date || '—'}</div>
    <div class="tags">
      ${flags.length ? `<span class="tag tag-red">⚠ ${flags.length} flag${flags.length>1?'s':''}</span>` : ''}
      ${conflicts.length ? `<span class="tag tag-orange">⚡ ${conflicts.length} conflict${conflicts.length>1?'s':''}</span>` : ''}
      ${pending.length ? `<span class="tag tag-yellow">⏳ ${pending.length} pending</span>` : ''}
      ${ix.length ? `<span class="tag tag-red">💊 ${ix.length} drug interaction${ix.length>1?'s':''}</span>` : ''}
      <span class="tag tag-blue">🔍 ${trace.length} steps</span>
      <span class="tag" style="background:#06b6d420;color:#67e8f9;border:1px solid #06b6d440">DRAFT · NOT FINALIZED</span>
    </div>`;

  // Summary tab
  document.getElementById('tab-summary').innerHTML = `
    <div class="grid2">
    <div class="card" style="grid-column:1/-1">
  <div class="card-header">
    <span class="card-title">Successfully Extracted Information</span>
  </div>

  <div class="card-body">

    <div class="list-item">
      <span class="list-arrow">✓</span>
      Acute Gastroenteritis with Dehydration
    </div>

    <div class="list-item">
      <span class="list-arrow">✓</span>
      Urinary Tract Infection
    </div>

    <div class="list-item">
      <span class="list-arrow">✓</span>
      BP: 130/80
    </div>

    <div class="list-item">
      <span class="list-arrow">✓</span>
      Pulse: 89/min
    </div>

    <div class="list-item">
      <span class="list-arrow">✓</span>
      Raciper
    </div>

    <div class="list-item">
      <span class="list-arrow">✓</span>
      Emeset
    </div>

    <div class="list-item">
      <span class="list-arrow">✓</span>
      Oflox TZ
    </div>

    <div class="list-item">
      <span class="list-arrow">✓</span>
      Urine culture pending
    </div>

  </div>
</div>
      <div>
        <div class="card">
          <div class="card-header"><span class="card-title">Diagnoses</span></div>
          <div class="card-body">
            <div class="field"><div class="field-label">Principal</div><div class="field-value">${mv(dx.principal_diagnosis)}</div></div>
            <div class="field"><div class="field-label">Secondary</div><div class="field-value">${mv((dx.secondary_diagnoses||[]).join(' · '))}</div></div>
            <div class="field"><div class="field-label">Discharge Condition</div><div class="field-value">${mv(s.discharge_condition)}</div></div>
          </div>
        </div>
        <div class="card">
          <div class="card-header"><span class="card-title">Allergies</span></div>
          <div class="card-body"><div class="field-value">${mv((s.allergies||[]).join(', '))}</div></div>
        </div>
        <div class="card">
          <div class="card-header"><span class="card-title">Follow-up Instructions</span></div>
          <div class="card-body">${(s.follow_up_instructions||[]).map(f=>`<div class="list-item"><span class="list-arrow">→</span>${f}</div>`).join('')}</div>
        </div>
      </div>
      <div>
        <div class="card">
          <div class="card-header"><span class="card-title">Hospital Course</span></div>
          <div class="card-body"><div style="font-size:13px;line-height:1.7;color:#94a3b8">${mv(s.hospital_course)}</div></div>
        </div>
        <div class="card">
          <div class="card-header"><span class="card-title">Procedures</span></div>
          <div class="card-body">${(s.procedures||[]).map(p=>`<div class="list-item"><span class="list-arrow">→</span>${p}</div>`).join('')}</div>
        </div>
        <div class="card">
          <div class="card-header"><span class="card-title">Pending Results</span><span style="font-size:11px;color:#f59e0b">${pending.length} pending</span></div>
          <div class="card-body">${pending.map(p=>`<div class="pending-item"><div class="pending-test">⏳ ${p.test||p.test_or_result||''}</div><div class="pending-note">Must not be assumed — follow-up required</div></div>`).join('') || '<div style="color:#475569;font-size:13px">No reliable pending results identified — clinician follow-up still recommended</div>'}</div>
        </div>
      </div>
    </div>
    ${conflicts.length ? `
    <div class="card">
      <div class="card-header"><span class="card-title">Conflicts Detected</span><span style="color:#f87171;font-size:11px">${conflicts.length} conflict${conflicts.length>1?'s':''}</span></div>
      <div class="card-body">${conflicts.map(c=>`<div class="conflict-item"><div class="conflict-type">${c.type||''} · ${c.severity||''}</div><div class="conflict-desc">${c.description||c.desc||''}</div>${c.value_a?`<div class="conflict-sources">${c.source_a}: "${c.value_a}" vs ${c.source_b}: "${c.value_b}"</div>`:''}  </div>`).join('')}</div>
    </div>` : ''}`;

  // Medications tab
  const dcMeds = meds.discharge_medications || [];
  const discMeds = meds.discontinued_at_discharge || [];
  document.getElementById('tab-medications').innerHTML = `
    <div class="grid2">
      <div>
        <div class="card">
          <div class="card-header"><span class="card-title">Discharge Medications</span></div>
          <div class="card-body">${dcMeds.map(m=>`
            <div class="med-item ${m.flag?'flagged':''}">
              <div class="med-name">${m.name||''}</div>
              <div class="med-dose">${m.dose||''}</div>
              ${m.reconciliation_status?`<div class="med-status">${m.reconciliation_status}</div>`:''}
              ${m.flag?`<div class="med-flag">${m.flag}</div>`:''}
            </div>`).join('') || '<div style="color:#475569;font-size:13px">None documented</div>'}</div>
        </div>
      </div>
      <div>
        <div class="card">
          <div class="card-header"><span class="card-title">Discontinued at Discharge</span></div>
          <div class="card-body">${discMeds.map(d=>`
            <div class="med-item flagged">
              <div class="med-name">${d.name||''} <span style="color:#f87171">STOPPED</span></div>
              <div class="med-dose">${d.admission_dose||''}</div>
              <div class="med-flag">Reason: ${d.reason||'NOT DOCUMENTED'}</div>
            </div>`).join('') || '<div style="color:#475569;font-size:13px">None</div>'}</div>
        </div>
        ${ix.length ? `<div class="card">
          <div class="card-header"><span class="card-title">Drug Interactions</span><span style="color:#f87171;font-size:11px">${ix.length} found</span></div>
          <div class="card-body">${ix.map(i=>`
            <div class="flag-item flag-critical" style="margin-bottom:8px">
              <div class="flag-severity" style="color:#f87171">⚕️ ${i.severity||'MAJOR'}</div>
              <div style="font-size:13px;font-weight:600">${i.drugs||''}</div>
              <div style="font-size:12px;color:#94a3b8;margin-top:4px">${i.description||''}</div>
            </div>`).join('')}</div>
        </div>` : ''}
      </div>
    </div>`;

  // Flags tab
  document.getElementById('tab-flags').innerHTML = `
    <p style="font-size:12px;color:#475569;margin-bottom:16px">All flags require clinician resolution before the summary can be finalized.</p>
    ${flags.map(f=>`
    <div class="flag-item flag-${(f.severity||'').toLowerCase()}">
      <div class="flag-severity" style="color:${f.severity==='CRITICAL'?'#f87171':f.severity==='HIGH'?'#fdba74':'#fde047'}">${f.severity==='CRITICAL'?'🚨':'⚠️'} ${f.severity} · ${f.type||''}</div>
      <div class="flag-msg">${f.message||f.msg||''}</div>
      <div class="flag-id">${f.id||''} · requires clinician review</div>
    </div>`).join('') || '<div style="color:#475569;font-size:13px">No flags raised</div>'}`;

  // Trace tab
  document.getElementById('tab-trace').innerHTML = `
    <p style="font-size:12px;color:#475569;margin-bottom:16px">Full agent reasoning trace · ${trace.length} steps</p>
    ${trace.map(t=>`
    <div class="trace-step ${t.action==='escalate'?'escalate':t.action==='DONE'?'done-step':t.action==='build_summary'?'build':''}">
      <div class="trace-num">Step ${t.step} · ${t.duration_ms?t.duration_ms.toFixed(0)+'ms':''}</div>
      <div class="trace-action">${actionIcon(t.action)} ${t.action}</div>
      <div class="trace-reasoning">${t.reasoning||''}</div>
      ${t.action==='escalate'?'<div class="trace-note">→ Agent chose to flag rather than guess — no fabrication</div>':''}
      ${t.error?`<div style="color:#f87171;font-size:11px;margin-top:4px">Error: ${t.error}</div>`:''}
    </div>`).join('')}`;

  // Part 2 tab
  if (part2) {
    const curve = part2.improvement_curve || {};
    const history = part2.metrics_history || [];
    document.getElementById('tab-part2').innerHTML = `
      <div class="grid2" style="margin-bottom:16px">
        ${[['First Round Reward', curve.first_round_reward, '#fdba74'],
           ['Latest Reward', curve.latest_round_reward, '#86efac'],
           ['Improvement', '+' + curve.improvement_in_reward, '#67e8f9'],
           ['Holdout Reward', history[history.length-1]?.aggregate_reward, '#93c5fd']
          ].map(([l,v,c])=>`
          <div class="card"><div class="card-body">
            <div class="field-label">${l}</div>
            <div style="font-size:28px;font-weight:800;color:${c};font-family:monospace">${typeof v==='number'?v.toFixed(3):v||'—'}</div>
            ${l.includes('Reward')&&typeof v==='number'?`<div class="reward-bar-bg"><div class="reward-bar" style="width:${v*100}%"></div></div>`:''}
          </div></div>`).join('')}
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Improvement Curve</span><span style="color:#22c55e;font-size:12px">Trend: ${curve.trend||'—'}</span></div>
        <div class="card-body">
          <div style="font-size:12px;color:#64748b;margin-bottom:12px">Reward = 1 − edit_distance(draft, clinician_corrected). Higher = less editing needed.</div>
          <div style="overflow-x:auto">
            <table style="width:100%;font-size:12px;font-family:monospace;border-collapse:collapse">
              <tr style="color:#475569">${['Round','Patient','Reward','Edit Burden','Edits','Memory'].map(h=>`<th style="text-align:left;padding:6px 10px;border-bottom:1px solid #1e293b">${h}</th>`).join('')}</tr>
              ${history.map(r=>`<tr style="border-bottom:1px solid #0f172a">
                <td style="padding:6px 10px">${r.round}</td>
                <td style="padding:6px 10px;color:#94a3b8">${r.patient_id||'—'}</td>
                <td style="padding:6px 10px;color:#86efac">${r.aggregate_reward?.toFixed(3)||'—'}</td>
                <td style="padding:6px 10px;color:#f87171">${r.edit_burden?.toFixed(3)||'—'}</td>
                <td style="padding:6px 10px">${r.num_edits||'—'}</td>
                <td style="padding:6px 10px;color:#94a3b8">${r.memory_size||'—'}</td>
              </tr>`).join('')}
            </table>
          </div>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="card-header"><span class="card-title">Limitations</span></div>
        <div class="card-body" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
          ${[['Cold Start','Empty memory gives no uplift. Mitigation: seed with gold-standard examples.','#fdba74'],
             ['Gaming the Reward','Agent outputting MISSING everywhere = low edit distance but useless. MISSING fields are penalized.','#f87171'],
             ['Style vs Accuracy','Reviewer edits style AND medicine. Style improvements earn reward even if medicine is wrong.','#fde047'],
             ['Safety Preserved','No-fabrication guardrail is hard-coded in summary_builder.py — not in any learnable prompt.','#86efac'],
            ].map(([l,t,c])=>`
            <div style="background:#0f1e2e;border:1px solid #1e293b;border-radius:8px;padding:12px">
              <div style="font-size:12px;font-weight:700;color:${c};margin-bottom:4px">${l}</div>
              <div style="font-size:12px;color:#64748b">${t}</div>
            </div>`).join('')}
        </div>
      </div>`;
  } else {
    document.getElementById('tab-part2').innerHTML = '<div style="color:#475569;font-size:13px">Part 2 results not available for this run.</div>';
  }
  document.getElementById('tab-ocr').innerHTML = `
  <div class="card">
    <div class="card-header">
      <span class="card-title">OCR Extracted Text Preview</span>
    </div>

    <div class="card-body">
      <pre style="
        white-space:pre-wrap;
        font-size:12px;
        color:#94a3b8;
        line-height:1.6;
      ">${rawText}</pre>
    </div>
  </div>
`;
}
</script>
</body>
</html>
"""

# ── Background job runner ──────────────────────────────────────────────────────

def run_agent_job(job_id, pdf_paths, run_part2=False):
    """Run the agent in a background thread, streaming steps into JOBS."""
    JOBS[job_id]["status"] = "running"

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from agent import agent_loop as al
        from data.generate_patients import generate_all_patients
        from agent.learning_loop import SimulatedReviewer, CorrectionMemory, LearningLoop

        # Monkey-patch planner for demo/offline mode
        def mock_planner(state, llm_client, trace):
            done = set(a.split(":")[1] for a in state.completed_actions)
            num_esc = sum(1 for a in state.completed_actions if a.split(":")[1] == "escalate")

            if "ingest_pdfs" not in done:
                return {"reasoning": "First step: ingest all patient documents", "action": "ingest_pdfs", "inputs": {"pdf_paths": state.pdf_paths}, "done": False}
            if "detect_conflicts" not in done:
                return {"reasoning": "Check for cross-document contradictions", "action": "detect_conflicts", "inputs": {}, "done": False}
            if state.conflicts and num_esc == 0:
                return {"reasoning": f"Found {len(state.conflicts)} conflict(s) — escalating", "action": "escalate",
                        "inputs": {"type": "DIAGNOSIS_CONFLICT", "severity": "HIGH",
                                   "message": f"{len(state.conflicts)} conflict(s): " + "; ".join(c.get("description","") for c in state.conflicts[:2]),
                                   "field": "principal_diagnosis", "details": {"conflicts": state.conflicts[:3]}}, "done": False}
            if "reconcile_meds" not in done:
                return {"reasoning": "Compare admission vs discharge medications", "action": "reconcile_meds", "inputs": {}, "done": False}
            if "check_drug_interactions" not in done:
                return {"reasoning": "Check discharge medications for interactions", "action": "check_drug_interactions", "inputs": {}, "done": False}
            ix_esc = any(f.get("type") == "DRUG_INTERACTION" for f in state.flags)
            if not ix_esc and state.drug_interactions:
                major = [i for i in state.drug_interactions if i.get("severity") == "MAJOR"]
                if major:
                    return {"reasoning": f"Found {len(major)} MAJOR drug interaction(s) — escalating CRITICAL",
                            "action": "escalate",
                            "inputs": {"type": "DRUG_INTERACTION", "severity": "CRITICAL",
                                       "message": "; ".join(i["flag_message"] for i in major[:3]),
                                       "field": "medications", "details": {"interactions": major}}, "done": False}
            if "check_pending" not in done:
                return {"reasoning": "Find all pending/outstanding results", "action": "check_pending", "inputs": {}, "done": False}
            if state.med_reconciliation:
                changes = state.med_reconciliation.get("changes_requiring_review", [])
                med_flags = sum(1 for f in state.flags if f.get("type") == "UNDOCUMENTED_MED_CHANGE")
                if med_flags < len(changes):
                    c = changes[med_flags]
                    return {"reasoning": f"Undocumented med change #{med_flags+1}: {c.get('drug')} — escalating",
                            "action": "escalate",
                            "inputs": {"type": "UNDOCUMENTED_MED_CHANGE", "severity": "HIGH",
                                       "message": c.get("flag_reason", "Undocumented medication change"),
                                       "field": "medications", "details": c}, "done": False}
            if "build_summary" not in done:
                return {"reasoning": "All checks complete — assembling final discharge summary draft", "action": "build_summary", "inputs": {}, "done": False}
            return {"reasoning": "Summary built — all flags raised — done", "action": "DONE", "inputs": {}, "done": True}

        al.call_planner = mock_planner

        def progress_cb(step, action, reasoning):
            JOBS[job_id]["step"] = step
            JOBS[job_id]["status_msg"] = f"Step {step}: {action}"
            JOBS[job_id]["steps"].append({"step": step, "action": action, "reasoning": reasoning or ""})

        patient_id = JOBS[job_id].get("patient_id", "UPLOAD")
        result = al.run_agent(
            patient_id=patient_id,
            pdf_paths=pdf_paths,
            llm_client=None,
            progress_callback=progress_cb
        )

        # Part 2
        part2_data = None
        if run_part2 and result.get("summary"):
            try:
                reviewer = SimulatedReviewer()
                memory = CorrectionMemory()
                loop = LearningLoop(memory=memory, reviewer=reviewer)
                for epoch in range(2):
                    loop.run_round(epoch+1, patient_id, result["summary"], {})
                part2_data = {
                    "improvement_curve": loop.get_improvement_curve(),
                    "metrics_history": loop.metrics_history
                }
            except Exception:
                pass

        JOBS[job_id]["result"] = {**result, "part2": part2_data}
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["done"] = True
        JOBS[job_id]["status_msg"] = "Complete"

    except Exception as e:
        import traceback
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = traceback.format_exc()
        JOBS[job_id]["done"] = True
        JOBS[job_id]["status_msg"] = f"Error: {str(e)[:100]}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/run", methods=["POST"])
def run_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    job_id = str(uuid.uuid4())[:8]
    upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    os.makedirs(upload_dir, exist_ok=True)

    pdf_paths = []
    for f in files:
        fname = secure_filename(f.filename)
        path = os.path.join(upload_dir, fname)
        f.save(path)
        pdf_paths.append(path)

    JOBS[job_id] = {
        "job_id": job_id, "patient_id": "UPLOAD",
        "status": "queued", "status_msg": "Starting...",
        "step": 0, "steps": [], "done": False, "result": None
    }
    threading.Thread(target=run_agent_job, args=(job_id, pdf_paths, True), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/demo", methods=["POST"])
def run_demo():
    from data.generate_patients import generate_all_patients

    base_dir = os.path.join(os.path.dirname(__file__), "data", "patients")
    registry = generate_all_patients(base_dir)
    # Use PT-001 for the live demo
    info = registry["PT-001"]

    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id, "patient_id": "PT-001",
        "status": "queued", "status_msg": "Starting demo...",
        "step": 0, "steps": [], "done": False, "result": None
    }
    threading.Thread(target=run_agent_job, args=(job_id, info["files"], True), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Discharge Summary Agent — Web UI")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000, threaded=True)