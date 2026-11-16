"""
dashboard_mp16.py — FITUR TAMBAHAN MP16: dashboard monitoring sistem KWS final (Streamlit).

Jalankan dari folder "Mini Project 16":
    streamlit run dashboard_mp16.py

Isi: status acceptance, metrik ringkas MP13–MP16, event log JSONL dari demo stream kontinu,
grafik, dan klasifikasi WAV yang di-upload memakai pipeline beku (front-end MP9 + model INT8 MP15).

Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)
"""
from pathlib import Path
import io
import json
import sys

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'features'))


def load_json(rel):
    p = ROOT/rel
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


st.set_page_config(page_title='KWS Embedded - MP16 Dashboard', layout='wide')
st.title('Keyword Spotting Embedded — Dashboard Proyek Final (MP16)')
st.caption('nyala / mati / naik / turun · model beku MP15 · tanpa post-test tuning')

acc = load_json('results/metrics/mp16_acceptance_metrics.json')
mp13 = load_json('results/metrics/mp13_final_test_metrics.json')
mp15 = load_json('results/metrics/mp15_deployment_metrics.json')
ext = load_json('results/metrics/mp16_extension_system_metrics.json')
status = load_json('project_status.json')

c1, c2, c3, c4 = st.columns(4)
c1.metric('Status proyek', status.get('status', '-'))
c2.metric('Model deployment', acc.get('selected_model', '-'))
c3.metric('Replay accuracy (test)', f"{acc.get('replay_accuracy', float('nan')):.2f}")
c4.metric('p95 pipeline (host)', f"{acc.get('pipeline_latency_ms', {}).get('p95', float('nan')):.2f} ms")

st.subheader('Acceptance checklist')
checks = acc.get('acceptance_checks', {})
st.table([{'check': k, 'hasil': 'PASS' if v else 'FAIL'} for k, v in checks.items()])

st.subheader('Ringkasan evaluasi')
col_a, col_b = st.columns(2)
with col_a:
    st.write('**MP13 frozen test (macro-F1)**')
    st.json({'classical': mp13.get('classical', {}).get('macro_f1'), 'tiny_cnn': mp13.get('tiny_cnn', {}).get('macro_f1')})
with col_b:
    st.write('**MP15 kuantisasi**')
    ci = mp15.get('classical_int8', {})
    st.json({'model': ci.get('name'), 'agreement_all': ci.get('agreement_all_vs_float', ci.get('agreement_all_24_vs_float')), 'payload_bytes': ci.get('payload_bytes_estimate')})

if ext:
    st.subheader('Demo stream kontinu + state machine (FITUR TAMBAHAN)')
    st.json(ext.get('stream_scores', {}))
    log = ROOT/'results/logs/mp16_event_log.jsonl'
    if log.exists():
        events = [json.loads(l) for l in log.read_text(encoding='utf-8').splitlines() if l.strip()]
        st.dataframe(events, width='stretch')

fig = ROOT/'results/figures/mp16_extension_system_analysis.png'
if fig.exists():
    st.image(str(fig), caption='Analisis sistem MP16')

st.subheader('Coba klasifikasi WAV (16 kHz mono)')
up = st.file_uploader('Upload WAV', type=['wav'])
if up is not None:
    from scipy.io import wavfile
    from kws_frontend import KWSFrontend
    fe = KWSFrontend(ROOT)
    fs, y = wavfile.read(io.BytesIO(up.getvalue()))
    if fs != fe.fs:
        st.error(f'Sample rate {fs} Hz; dibutuhkan {fe.fs} Hz')
    else:
        y = y.mean(axis=1) if y.ndim > 1 else y
        y = y.astype(np.float64)/(np.iinfo(y.dtype).max if np.issubdtype(y.dtype, np.integer) else 1.0)
        v, _ = fe.features(y)
        pkg = np.load(ROOT/'models/mp15_classical_int8.npz', allow_pickle=True)
        z = (v[pkg['selected_indices']]-pkg['selected_mean'])/pkg['selected_scale']
        zq = np.clip(np.round(z/pkg['input_quant_scale'][0]), -127, 127)
        if str(pkg['kind']) == 'tree':
            n = 0
            while pkg['tree_feature'][n] >= 0:
                n = pkg['tree_left'][n] if zq[pkg['tree_feature'][n]] <= pkg['tree_threshold_q'][n] else pkg['tree_right'][n]
            k = int(pkg['tree_leaf_class'][n])
        else:
            s = (pkg['weight_q'].astype(int)@zq)*pkg['weight_scales']*pkg['input_quant_scale'][0]+pkg['bias']
            k = int(np.argmax(s))
        st.success(f"Prediksi: **{pkg['classes'][k]}**")
