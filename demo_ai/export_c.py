"""Kuantisasi INT8 + ekspor header C + verifikasi gcc (C == numpy).

Langkah:
 1. kalibrasi skala aktivasi dari data TRAIN (persentil 99,99)
 2. bangun model INT8 (int8_ref.Int8KWS), ukur agreement INT8 vs float di VALIDATION
 3. hitung ulang prototipe penutur dengan model INT8 (data train)
 4. tulis firmware_src/kws_model_int8.h + kws_tables.h
 5. compile test_engine.c (gcc) lalu bandingkan dengan numpy:
      (a) jaringan INT8 pada input int8 identik -> peta fitur terakhir harus SAMA PERSIS
      (b) front-end C vs Python pada waveform val+test -> selisih log-Mel & agreement input int8
      (c) prediksi akhir end-to-end (gelombang -> kelas) C vs numpy
"""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import torch

import common as C
from int8_ref import Int8KWS, calibrate, load_float
from train import load, predict

FW = C.ROOT/'firmware_src'
BUILD = C.OUT/'build'


def carr(name, arr, ctype, per_line=24, fmt=None):
    a = np.asarray(arr).ravel()
    if fmt is None:
        fmt = (lambda v: f'{float(v):.9e}f') if 'float' in ctype else (lambda v: str(int(v)))
    body = ',\n  '.join(', '.join(fmt(v) for v in a[i:i+per_line]) for i in range(0, len(a), per_line))
    return f'static const {ctype} {name}[{len(a)}] = {{\n  {body}\n}};\n'


def write_model_header(q: Int8KWS, protos, spk_thr, tau, path, macs=0):
    L = q.layers; c1 = L[0]
    H1 = (C.N_FRAMES+2*c1['pad'][0]-c1['k'][0])//c1['stride'][0]+1
    W1 = (C.N_MELS+2*c1['pad'][1]-c1['k'][1])//c1['stride'][1]+1
    H, W = H1, W1
    for s in q.strides:
        H, W = (H-1)//s+1, (W-1)//s+1
    o = ['/* AUTO-GENERATED oleh demo_ai/export_c.py — DS-CNN INT8 multi-tugas (kata + penutur). Jangan diedit manual. */',
         '#ifndef KWS_MODEL_INT8_H', '#define KWS_MODEL_INT8_H', '#include <stdint.h>',
         f'#define KWS_WIDTH {q.width}', f'#define KWS_EMB {q.e_w.shape[0]}', f'#define KWS_N_BLOCKS {len(q.strides)}',
         f'#define KWS_C1_H {H1}', f'#define KWS_C1_W {W1}', f'#define KWS_C1_KH {c1["k"][0]}', f'#define KWS_C1_KW {c1["k"][1]}',
         f'#define KWS_C1_SH {c1["stride"][0]}', f'#define KWS_C1_SW {c1["stride"][1]}', f'#define KWS_C1_PH {c1["pad"][0]}',
         f'#define KWS_C1_PW {c1["pad"][1]}', f'#define KWS_LAST_H {H}', f'#define KWS_LAST_W {W}',
         f'#define KWS_S_IN {np.float32(q.s_in):.9e}f', f'#define KWS_S_LAST {np.float32(q.s_last):.9e}f',
         f'#define KWS_GAP_SCALE {np.float32(q.s_last/(H*W)):.9e}f', f'#define KWS_PROTO_TAU {tau:.6e}f',
         f'#define KWS_SPK_THRESHOLD {spk_thr:.6e}f', f'#define KWS_MODEL_BYTES {q.sizes()["total_bytes"]}', f'#define KWS_MACS {int(macs)}', '',
         'typedef struct { int stride; const int8_t *dw_w; const int32_t *dw_b, *dw_m0, *dw_rs;',
         '                 const int8_t *pw_w; const int32_t *pw_b, *pw_m0, *pw_rs; } kws_block_t;', '']
    o.append(carr('KWS_C1_WT', c1['w'], 'int8_t')); o.append(carr('KWS_C1_B', c1['b'], 'int32_t'))
    o.append(carr('KWS_C1_M0', c1['m0'], 'int32_t')); o.append(carr('KWS_C1_RS', c1['rs'], 'int32_t'))
    blocks = []
    for i in range(len(q.strides)):
        dw, pw = L[1+2*i], L[2+2*i]
        for tag, Lx in (('DW', dw), ('PW', pw)):
            o.append(carr(f'KWS_B{i}_{tag}_W', Lx['w'], 'int8_t')); o.append(carr(f'KWS_B{i}_{tag}_B', Lx['b'], 'int32_t'))
            o.append(carr(f'KWS_B{i}_{tag}_M0', Lx['m0'], 'int32_t')); o.append(carr(f'KWS_B{i}_{tag}_RS', Lx['rs'], 'int32_t'))
        blocks.append(f'  {{{q.strides[i]}, KWS_B{i}_DW_W, KWS_B{i}_DW_B, KWS_B{i}_DW_M0, KWS_B{i}_DW_RS, '
                      f'KWS_B{i}_PW_W, KWS_B{i}_PW_B, KWS_B{i}_PW_M0, KWS_B{i}_PW_RS}}')
    o.append('static const kws_block_t KWS_BLOCKS[KWS_N_BLOCKS] = {\n' + ',\n'.join(blocks) + '\n};\n')
    o.append(carr('KWS_KW_W', q.kw_w, 'float', 8)); o.append(carr('KWS_KW_B', q.kw_b, 'float', 8))
    o.append(carr('KWS_E_W', q.e_w, 'float', 8)); o.append(carr('KWS_E_B', q.e_b, 'float', 8))
    o.append('/* prototipe penutur 4 anggota (urutan: Ezra, Abdillah, Ivang, Jahzeel), embedding INT8-model ternormalisasi */')
    o.append(carr('KWS_SPK_PROTO', protos, 'float', 8))
    o.append('#endif')
    path.write_text('\n'.join(o), encoding='utf-8')


def write_tables(path):
    n = C.NFFT
    bitrev = [int(f'{i:09b}'[::-1], 2) for i in range(n)]
    k = np.arange(n//2)
    o = ['/* AUTO-GENERATED oleh demo_ai/export_c.py — tabel front-end log-Mel (FFT 512, Hamming 400, 40 mel). */',
         '#ifndef KWS_TABLES_H', '#define KWS_TABLES_H', '#include <stdint.h>']
    o.append(carr('KWS_BITREV', bitrev, 'uint16_t'))
    o.append(carr('KWS_TW_COS', np.cos(2*np.pi*k/n).astype(np.float32), 'float', 8))
    o.append(carr('KWS_TW_SIN', (-np.sin(2*np.pi*k/n)).astype(np.float32), 'float', 8))
    o.append(carr('KWS_HAMMING', C.WINDOW, 'float', 8))
    starts, lens, ofs, ws = [], [], [], []
    for m in range(C.N_MELS):
        nz = np.nonzero(C.MEL_FB[m])[0]
        starts.append(int(nz[0])); lens.append(int(nz[-1]-nz[0]+1)); ofs.append(len(ws))
        ws += C.MEL_FB[m, nz[0]:nz[-1]+1].tolist()
    o.append(carr('KWS_MEL_START', starts, 'uint16_t')); o.append(carr('KWS_MEL_LEN', lens, 'uint16_t'))
    o.append(carr('KWS_MEL_OFS', ofs, 'uint16_t')); o.append(carr('KWS_MEL_W', np.float32(ws), 'float', 8))
    o.append('#endif')
    path.write_text('\n'.join(o), encoding='utf-8')


def write_web_page():
    import gzip
    html = (C.ROOT.parent/'demo_web'/'index.html').read_bytes()
    gz = gzip.compress(html, 9, mtime=0)
    o = ['/* AUTO-GENERATED oleh demo_ai/export_c.py dari demo_web/index.html (gzip). */', '#pragma once', '#include <stdint.h>',
         f'static const unsigned int WEB_PAGE_GZ_LEN = {len(gz)};', carr('WEB_PAGE_GZ', np.frombuffer(gz, np.uint8), 'uint8_t', 32)]
    (C.ROOT/'esp32_kws_ai'/'web_page.h').write_text('\n'.join(o), encoding='utf-8')


def run_c(mode, data, n_w, n_e, last_size):
    BUILD.mkdir(parents=True, exist_ok=True)
    fin, fout = BUILD/f'in_{mode}.bin', BUILD/f'out_{mode}.bin'
    with open(fin, 'wb') as f:
        f.write(np.int32(len(data)).tobytes()); f.write(np.ascontiguousarray(data).tobytes())
    subprocess.run([str(BUILD/'test_engine.exe'), mode, str(fin), str(fout)], check=True)
    rec = np.dtype([('xq', np.int8, C.N_FRAMES*C.N_MELS), ('logits', '<f4', 6), ('f', '<f4', n_w), ('e', '<f4', n_e),
                    ('last', np.int8, last_size)])
    return np.fromfile(fout, dtype=rec)


def main():
    model = load_float()
    tr, va = load('train'), load('val')
    rs = np.random.default_rng(C.SEED)
    cal_idx = rs.choice(len(tr['kw']), 3000, replace=False)
    scales = calibrate(model, tr['X'][cal_idx].astype(np.float32))
    q = Int8KWS(model, scales)

    # agreement INT8 vs float (validation)
    Xv = va['X'].astype(np.float32)
    pf, *_ = predict(model, torch.tensor(Xv).unsqueeze(1))
    outs = [q.forward(x) for x in Xv]
    pq = np.stack([o['kw'] for o in outs])
    agree_val = float(np.mean(pq.argmax(1) == pf.argmax(1)))
    acc_f = float(np.mean(pf.argmax(1) == va['kw'])); acc_q = float(np.mean(pq.argmax(1) == va['kw']))

    # prototipe penutur dari model INT8 (data TRAIN, contoh keyword per penutur, maks 600 per penutur)
    protos = []
    for s in range(4):
        idx = np.where((tr['spk'] == s) & (tr['kw'] < 4))[0][:600]
        E = np.stack([q.forward(tr['X'][i].astype(np.float32))['e'] for i in idx])
        E /= np.linalg.norm(E, axis=1, keepdims=True); p = E.mean(0); protos.append(p/np.linalg.norm(p))
    protos = np.stack(protos).astype(np.float32)
    fs = json.loads((C.OUT/'fewshot_report.json').read_text(encoding='utf-8')) if (C.OUT/'fewshot_report.json').exists() else {}
    spk_thr = float(fs.get('speaker_threshold', 0.6)); tau = float(fs.get('proto_tau', 10.0))

    FW.mkdir(parents=True, exist_ok=True)
    write_model_header(q, protos, spk_thr, tau, FW/'kws_model_int8.h', macs=model.macs())
    write_web_page()
    write_tables(FW/'kws_tables.h')                                     # tulis dulu, baru salin ke sketch
    for fn in ('kws_engine.c', 'kws_engine.h', 'kws_model_int8.h', 'kws_tables.h'):   # sketch Arduino butuh salinan lokal
        shutil.copy(FW/fn, C.ROOT/'esp32_kws_ai'/fn)
    np.savez(C.OUT/'int8_params.npz', protos=protos, **{f'scale_{k}': v for k, v in scales.items()})
    C.save_json(C.OUT/'act_scales.json', scales)

    gcc = shutil.which('gcc'); assert gcc, 'gcc tidak ditemukan'
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run([gcc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror', '-I', str(FW), str(FW/'kws_engine.c'),
                    str(FW/'test_engine.c'), '-o', str(BUILD/'test_engine.exe'), '-lm'], check=True)
    H, W = q.forward(Xv[0])['A'].shape[:2]; last_size = H*W*q.width

    # (a) jaringan INT8 saja: input int8 identik
    XQ = np.stack([q.quantize_input(x) for x in Xv])
    rc = run_c('xq', XQ.reshape(len(XQ), -1), q.width, q.e_w.shape[0], last_size)
    last_np = np.stack([o['A'].ravel() for o in outs])
    net_exact = bool(np.array_equal(rc['last'], last_np))
    net_logit_err = float(np.max(np.abs(rc['logits']-pq)))
    net_agree = float(np.mean(rc['logits'].argmax(1) == pq.argmax(1)))

    # (b)+(c) end-to-end dari waveform bersih val+test
    wv = np.concatenate([np.load(C.OUT/'waves_val.npz')['W'], np.load(C.OUT/'waves_test.npz')['W']])
    pcm = np.clip(np.round(wv*32768.0), -32768, 32767).astype('<i2')
    re = run_c('wave', pcm, q.width, q.e_w.shape[0], last_size)
    lm_py = np.stack([C.logmel(p.astype(np.float32)/32768.0) for p in pcm])
    xq_py = np.stack([q.quantize_input(x).ravel() for x in lm_py])
    xq_agree = float(np.mean(re['xq'] == xq_py)); xq_maxdiff = int(np.max(np.abs(re['xq'].astype(int)-xq_py.astype(int))))
    py_pred = np.array([q.forward(x)['kw'].argmax() for x in lm_py])
    e2e_agree = float(np.mean(re['logits'].argmax(1) == py_pred))

    sz = q.sizes()
    rep = {'calibration': 'train subset 3000 contoh, persentil 99,99', 'act_scales': scales,
           'val_float_accuracy': acc_f, 'val_int8_accuracy': acc_q, 'val_int8_vs_float_agreement': agree_val,
           'c_network_exact_last_featuremap': net_exact, 'c_network_max_logit_err': net_logit_err, 'c_network_argmax_agreement': net_agree,
           'c_frontend_int8_input_agreement': xq_agree, 'c_frontend_int8_input_max_lsb_diff': xq_maxdiff,
           'c_end_to_end_argmax_agreement': e2e_agree, 'n_waves': int(len(pcm)), 'model_bytes': sz,
           'macs': int(load_float().macs()), 'speaker_threshold': spk_thr, 'proto_tau': tau,
           'header': 'firmware_src/kws_model_int8.h', 'gcc': subprocess.run([gcc, '--version'], capture_output=True, text=True).stdout.splitlines()[0]}
    C.save_json(C.OUT/'export_report.json', rep)
    for k, v in rep.items():
        if k not in ('act_scales',):
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
