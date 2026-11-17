"""Board tiruan (mock) untuk halaman demo `index.html`.

Menjalankan PIPELINE ASLI proyek di PC (front-end beku MP9 `kws_frontend.py` + model INT8 MP15 +
gate OOD MP16 + command state machine MP16) dan berbicara dengan protokol WebSocket yang sama
dengan firmware board nanti (lihat PROTOKOL.md). Server ini juga bisa dipakai apa adanya di
Raspberry Pi.

Pemakaian (dari folder proyek):
    python demo_web/mock_board.py                # tanpa mic: hanya kontrol manual + suara dari web
    python demo_web/mock_board.py --mic          # mic PC/Pi -> deteksi kontinu (endpoint trigger)
    python demo_web/mock_board.py --replay       # putar WAV test (block 6) bergiliran, untuk uji tampilan
Lalu buka http://localhost:8080 (atau http://<ip-pc>:8080 dari HP di WiFi yang sama).

PENTING: angka latency yang dikirim adalah latency PC/Pi tempat skrip ini berjalan, dan `hello.device`
menyebutkannya dengan jujur. Ini BUKAN angka ESP32.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

HERE = Path(__file__).resolve().parent
FS, CLIP_N, BLOCK = 16000, 19200, 1600          # 1,2 s klip; blok mic 100 ms
LVL_MIN, LVL_MAX, LVL_ON = 0, 10, 5
COOLDOWN_S, WATCHDOG_S = 0.8, 0.5                 # sama dengan FSM MP16 (8 tick / 5 tick × 100 ms)


# ----------------------------------------------------------------------------- pipeline beku
class Pipeline:
    def __init__(self, root: Path):
        sys.path.insert(0, str(root/'features'))
        from kws_frontend import KWSFrontend
        self.fe = KWSFrontend(root)
        self.pkg = np.load(root/'models/mp15_classical_int8.npz', allow_pickle=True)
        self.classes = self.pkg['classes'].astype(str).tolist()
        d = np.load(root/'features/dataset/mp09_features_39d.npz', allow_pickle=True)
        X = d['X'].astype(np.float64); trm = d['train_mask'].astype(bool); vam = d['val_mask'].astype(bool)
        self.mu, self.sd = X[trm].mean(0), X[trm].std(0)+1e-9
        self.lw = LedoitWolf().fit((X[trm]-self.mu)/self.sd)
        self.ood_thr = 1.25*max(self.ood(x) for x in X[trm | vam])       # sama dengan MP16 (train+val saja)
        h = root/'firmware/mp15_classical_model_int8.h'
        self.sha = hashlib.sha256(h.read_bytes()).hexdigest() if h.exists() else ''
        self.payload = int(self.pkg['weight_q'].size + 4*(self.pkg['weight_scales'].size + self.pkg['bias'].size)
                           + 4*3*self.pkg['selected_indices'].size + 4) if str(self.pkg['kind']) == 'linear' else None
        self.root = root

    def ood(self, v39):
        return float(np.sqrt(self.lw.mahalanobis(((np.asarray(v39, float)-self.mu)/self.sd)[None])[0]))

    def predict_int8(self, v39):
        p = self.pkg
        z = (v39[p['selected_indices']]-p['selected_mean'])/p['selected_scale']
        zq = np.clip(np.round(z/p['input_quant_scale'][0]), -127, 127).astype(np.int32)
        s = (p['weight_q'].astype(np.int32)@zq)*p['weight_scales']*p['input_quant_scale'][0]+p['bias']
        o = np.sort(s)
        return self.classes[int(np.argmax(s))], float(o[-1]-o[-2])

    def run(self, y):
        t0 = time.perf_counter()
        v39, _ = self.fe.features(np.asarray(y, float))
        dist = self.ood(v39)
        label, margin = self.predict_int8(v39)
        return label, margin, dist, (time.perf_counter()-t0)*1000


# ----------------------------------------------------------------------------- pipeline deep learning (siklus 2)
class DLPipeline:
    """Meniru firmware esp32_kws_ai (jaringan bit-exact via referensi numpy demo_ai/int8_ref.py):
    jendela 1,0 s -> log-Mel 98x40 -> DS-CNN INT8 -> kata + embedding penutur -> prototipe (kws_decide)."""
    KW = ['nyala', 'mati', 'naik', 'turun', 'diam', 'lain']

    def __init__(self, ai_root: Path, classical):
        sys.path.insert(0, str(ai_root))
        import common as AC
        from int8_ref import Int8KWS, load_float
        self.C = AC
        model = load_float(AC.OUT/'kwsnet.pt')
        p = dict(np.load(AC.OUT/'int8_params.npz'))
        self.q = Int8KWS(model, {k[6:]: float(v) for k, v in p.items() if k.startswith('scale_')})
        rep = json.loads((AC.OUT/'export_report.json').read_text(encoding='utf-8'))
        self.spk_thr, self.tau = float(rep['speaker_threshold']), float(rep['proto_tau'])
        self.payload, self.macs, self.width = rep['model_bytes']['total_bytes'], rep['macs'], model.width
        self.users = [{'name': n, 'spk': p['protos'][i], 'kw': None} for i, n in enumerate(['Ezra', 'Abdillah', 'Ivang', 'Jahzeel'])]
        self.user_file = AC.OUT/'mock_users.npz'
        if self.user_file.exists():
            d = np.load(self.user_file, allow_pickle=True)
            for n, s, k in zip(d['names'], d['spk'], d['kw']):
                self.users.append({'name': str(n), 'spk': s, 'kw': k})
        self.classical = classical

    def save_users(self):
        extra = self.users[4:]
        np.savez(self.user_file, names=np.array([u['name'] for u in extra]),
                 spk=np.array([u['spk'] for u in extra]).reshape(-1, 32),
                 kw=np.array([u['kw'] for u in extra]).reshape(-1, 4, self.width))

    def users_json(self):
        return [{'name': u['name'], 'enrolled': i >= 4} for i, u in enumerate(self.users)]

    def run(self, y):
        t0 = time.perf_counter()
        w, _ = self.C.center_window(np.asarray(y, np.float32))
        lm = self.C.logmel(w); t1 = time.perf_counter()
        xq = self.q.quantize_input(lm); o = self.q.forward_q(xq)
        z = o['kw'].astype(np.float64); pr = np.exp(z-z.max()); pr /= pr.sum()
        e = o['e']/(np.linalg.norm(o['e'])+1e-9)
        cos = [float(e@u['spk']) for u in self.users]; bs = int(np.argmax(cos))
        spk = bs if cos[bs] >= self.spk_thr else -1
        kw, kp = int(np.argmax(pr)), float(pr.max())
        if spk >= 0 and self.users[spk]['kw'] is not None and kw < 4:           # fusi prototipe (sama dengan kws_decide)
            fn = o['f']/(np.linalg.norm(o['f'])+1e-9); zz = self.tau*(self.users[spk]['kw']@fn)
            pp = np.exp(zz-zz.max()); pp /= pp.sum(); v = 0.5*pr[:4]+0.5*pr[:4].sum()*pp
            kw, kp = int(np.argmax(v)), float(v.max())
        camk = kw if kw < 4 else int(np.argmax(pr[:4]))
        cam = np.round(255*self.q.cam(o['A'], camk)).astype(np.uint8)
        t2 = time.perf_counter()
        out = {'kw': kw, 'kw_prob': kp, 'probs': pr, 'speaker': spk, 'cos': cos, 'f': o['f'], 'e': e,
               'spec': (xq.astype(np.int16)+128).astype(np.uint8), 'cam': cam, 'lat_fe': (t1-t0)*1000, 'lat_nn': (t2-t1)*1000}
        if self.classical is not None:
            lab, _, dist, _ = self.classical.run(np.asarray(y, float))
            out['classical'] = {'label': lab, 'reject': bool(dist > self.classical.ood_thr), 'ood': round(dist, 2)}
        return out


# ----------------------------------------------------------------------------- state machine
class FSM:
    def __init__(self):
        self.power, self.level, self.state = False, LVL_ON, 'IDLE'
        self.cool_until, self.last_audio = 0.0, time.monotonic()

    def tick(self):
        now = time.monotonic()
        if self.state == 'COOLDOWN' and now >= self.cool_until:
            self.state = 'IDLE'; return True
        return False

    def apply(self, label, bypass_cooldown=False):
        if self.state == 'FAULT':
            return 'NONE'
        if not bypass_cooldown and time.monotonic() < self.cool_until:
            return 'NONE'
        a = 'NONE'
        if label == 'nyala':
            self.power, a = True, 'ON'
        elif label == 'mati':
            self.power, a = False, 'OFF'
        elif label in ('naik', 'turun') and self.power:
            nx = self.level + (1 if label == 'naik' else -1)
            if LVL_MIN <= nx <= LVL_MAX:
                self.level, a = nx, ('LEVEL_UP' if label == 'naik' else 'LEVEL_DOWN')
            else:
                a = 'IGNORED_BOUND'
        if a not in ('NONE', 'IGNORED_BOUND'):
            self.cool_until, self.state = time.monotonic()+COOLDOWN_S, 'COOLDOWN'
        return a

    def snap(self):
        return {'power': self.power, 'level': self.level, 'state': self.state}


# ----------------------------------------------------------------------------- server
class Board:
    def __init__(self, pipe, args, dl=None):
        self.pipe, self.args, self.fsm, self.dl = pipe, args, FSM(), dl
        self.lock, self.enroll = False, None
        self.clients: set = set()
        self.t0 = time.monotonic()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.page = (HERE/'index.html').read_bytes()

    async def broadcast(self, msg):
        data = json.dumps(msg)
        for ws in list(self.clients):
            try:
                await ws.send(data)
            except Exception:
                self.clients.discard(ws)

    def emit(self, msg):                                # aman dipanggil dari thread mic
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)

    def classify(self, y, source):
        if self.dl is not None:
            return self.classify_dl(y, source)
        label, margin, dist, lat = self.pipe.run(y)
        t_ms = int((time.monotonic()-self.t0)*1000)
        if dist > self.pipe.ood_thr:
            ev = {'type': 'event', 'kind': 'reject', 'label': None, 'pred_without_gate': label, 'action': 'REJECT'}
        else:
            ev = {'type': 'event', 'kind': 'keyword', 'label': label, 'action': self.fsm.apply(label)}
        ev.update(self.fsm.snap(), t_ms=t_ms, score=round(margin, 3), ood=round(dist, 2), latency_ms=round(lat, 2), source=source)
        print(f"[{t_ms/1000:8.2f}s] {source:9s} {ev['kind']:7s} {str(ev['label']):6s} -> {ev['action']:13s} "
              f"power={ev['power']} level={ev['level']} ood={dist:.1f} lat={lat:.1f} ms", flush=True)
        return ev

    def classify_dl(self, y, source):
        r = self.dl.run(y); t_ms = int((time.monotonic()-self.t0)*1000); KW = DLPipeline.KW
        if self.enroll is not None:
            return self.enroll_consume(r)
        is_kw = r['kw'] < 4 and r['kw_prob'] >= self.args.kw_min_prob
        spk_name = self.dl.users[r['speaker']]['name'] if r['speaker'] >= 0 else None
        if not is_kw:
            ev = {'kind': 'reject', 'label': None, 'action': 'REJECT'}
        elif self.lock and r['speaker'] < 0:
            ev = {'kind': 'reject', 'label': KW[r['kw']], 'action': 'REJECT_SPEAKER'}
        else:
            ev = {'kind': 'keyword', 'label': KW[r['kw']], 'action': self.fsm.apply(KW[r['kw']])}
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
        ev.update(type='event', t_ms=t_ms, source=source, score=round(r['kw_prob'], 3),
                  latency_ms=round(r['lat_fe']+r['lat_nn'], 2), lat_fe_ms=round(r['lat_fe'], 2), lat_nn_ms=round(r['lat_nn'], 2),
                  probs=[round(float(v), 3) for v in r['probs']], speaker=spk_name, speaker_cos=round(max(r['cos']), 3),
                  speaker_scores={u['name']: round(c, 3) for u, c in zip(self.dl.users, r['cos'])},
                  cam_h=int(r['cam'].shape[0]), cam_w=int(r['cam'].shape[1]), cam_b64=b64(r['cam']),
                  spec_t=98, spec_f=40, spec_b64=b64(r['spec']), **self.fsm.snap())
        if 'classical' in r:
            ev['classical'] = r['classical']
        cl = r.get('classical', {})
        print(f"[{t_ms/1000:8.2f}s] {source:9s} {ev['kind']:7s} {KW[r['kw']]:6s} p={r['kw_prob']:.2f} "
              f"spk={spk_name or '?'}({max(r['cos']):.2f}) -> {ev['action']:14s} power={ev['power']} level={ev['level']} "
              f"| klasik={cl.get('label')}{'(tolak)' if cl.get('reject') else ''}", flush=True)
        return ev

    ENROLL_WORDS, ENROLL_TOTAL = ['nyala', 'mati', 'naik', 'turun'], 12

    def enroll_prompt(self):
        e = self.enroll
        return {'type': 'enroll', 'state': 'prompt', 'name': e['name'], 'index': e['i'], 'total': self.ENROLL_TOTAL,
                'word': self.ENROLL_WORDS[e['i'] % 4]}

    def enroll_consume(self, r):
        e = self.enroll
        if r['kw'] == 4 and r['probs'][4] > 0.8:
            self.emit({'type': 'enroll', 'state': 'retry', 'reason': 'suara terlalu pelan / tidak terdengar'})
            return self.enroll_prompt()
        k = e['i'] % 4; e['f'][k] += r['f']/(np.linalg.norm(r['f'])+1e-9); e['e'] += r['e']; e['i'] += 1
        print(f"[enrol] {e['name']} {e['i']}/{self.ENROLL_TOTAL} ({self.ENROLL_WORDS[k]}) model mengira: {DLPipeline.KW[r['kw']]}", flush=True)
        if e['i'] < self.ENROLL_TOTAL:
            return self.enroll_prompt()
        u = {'name': e['name'], 'spk': (e['e']/np.linalg.norm(e['e'])).astype(np.float32),
             'kw': (e['f']/np.linalg.norm(e['f'], axis=1, keepdims=True)).astype(np.float32)}
        idx = next((i for i in range(4, len(self.dl.users)) if self.dl.users[i]['name'] == u['name']), None)
        if idx is None:
            if len(self.dl.users) >= 8:
                self.dl.users.pop(4)
            self.dl.users.append(u)
        else:
            self.dl.users[idx] = u
        self.dl.save_users(); self.enroll = None
        self.emit({'type': 'users', 'list': self.dl.users_json()})
        return {'type': 'enroll', 'state': 'done', 'name': u['name']}

    async def handle_text_dl(self, m):
        t = m.get('type')
        if t == 'set' and 'lock' in m:
            self.lock = bool(m['lock']); await self.broadcast({'type': 'settings', 'lock': self.lock})
        elif t == 'enroll_start' and str(m.get('name', '')).strip():
            self.enroll = {'name': str(m['name']).strip()[:20], 'i': 0, 'f': np.zeros((4, self.dl.width)), 'e': np.zeros(32)}
            await self.broadcast(self.enroll_prompt())
        elif t == 'enroll_cancel':
            self.enroll = None; await self.broadcast({'type': 'enroll', 'state': 'cancelled'})
        elif t == 'user_delete':
            self.dl.users = self.dl.users[:4]+[u for u in self.dl.users[4:] if u['name'] != m.get('name')]
            self.dl.save_users(); await self.broadcast({'type': 'users', 'list': self.dl.users_json()})

    def hello(self):
        if self.dl is None:
            return {'type': 'hello', 'device': self.args.device_name, 'model': 'classical_logreg_int8',
                    'model_sha': self.pipe.sha, 'payload_bytes': self.pipe.payload, 'fs': FS,
                    'ood_threshold': round(self.pipe.ood_thr, 2), **self.fsm.snap()}
        return {'type': 'hello', 'device': self.args.device_name, 'model': f'ds_cnn_int8_w{self.dl.width}',
                'payload_bytes': self.dl.payload, 'macs': self.dl.macs, 'fs': FS, 'classes': DLPipeline.KW,
                'features': ['probs', 'speaker', 'cam', 'spec', 'enroll', 'lock', 'classical'], 'lock': self.lock,
                'users': self.dl.users_json(), 'spk_threshold': round(self.dl.spk_thr, 3), 'kw_min_prob': self.args.kw_min_prob,
                **self.fsm.snap()}

    # HTTP: "/" -> halaman demo, "/ws" -> WebSocket
    def process_request(self, connection, request):
        if request.path == '/ws':
            return None
        if request.path in ('/', '/index.html'):
            return Response(HTTPStatus.OK, 'OK', Headers([('Content-Type', 'text/html; charset=utf-8'),
                                                          ('Content-Length', str(len(self.page))), ('Cache-Control', 'no-store')]), self.page)
        return connection.respond(HTTPStatus.NOT_FOUND, 'Not found\n')

    async def handler(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps(self.hello()))
        try:
            async for msg in ws:
                if isinstance(msg, bytes):                                   # suara dari mic HP: PCM16 mono 16 kHz
                    y = np.frombuffer(msg, dtype='<i2').astype(np.float64)/32767.0
                    if not 0.25*FS <= len(y) <= 3*FS:
                        await ws.send(json.dumps({'type': 'error', 'message': f'panjang audio {len(y)} sampel tidak valid'}))
                        continue
                    ev = await asyncio.to_thread(self.classify, y, 'web_mic')
                    await self.broadcast(ev)
                    continue
                try:
                    m = json.loads(msg)
                except ValueError:
                    continue
                if self.dl is not None and m.get('type') != 'cmd':
                    await self.handle_text_dl(m)
                elif m.get('type') == 'cmd':
                    lab = {'ON': 'nyala', 'OFF': 'mati', 'LEVEL_UP': 'naik', 'LEVEL_DOWN': 'turun'}.get(m.get('action'))
                    if lab:
                        a = self.fsm.apply(lab, bypass_cooldown=True)
                        await self.broadcast({'type': 'event', 'kind': 'manual', 'label': None, 'action': a, 'source': 'web',
                                              't_ms': int((time.monotonic()-self.t0)*1000), **self.fsm.snap()})
        finally:
            self.clients.discard(ws)

    async def housekeeping(self):
        while True:                                     # cooldown -> IDLE, watchdog mic
            await asyncio.sleep(0.1)
            if self.fsm.tick():
                await self.broadcast({'type': 'state', **self.fsm.snap()})
            if self.args.mic:
                gap = time.monotonic()-self.fsm.last_audio
                if gap > WATCHDOG_S and self.fsm.state != 'FAULT':
                    self.fsm.power, self.fsm.state = False, 'FAULT'
                    await self.broadcast({'type': 'event', 'kind': 'failsafe', 'action': 'FAILSAFE_OFF', 'source': 'board_mic',
                                          't_ms': int((time.monotonic()-self.t0)*1000), **self.fsm.snap()})
                elif gap <= WATCHDOG_S and self.fsm.state == 'FAULT':
                    self.fsm.state = 'IDLE'
                    await self.broadcast({'type': 'event', 'kind': 'recover', 'action': 'RECOVER', 'source': 'board_mic',
                                          't_ms': int((time.monotonic()-self.t0)*1000), **self.fsm.snap()})

    async def replay(self):
        split = json.loads((self.pipe.root/'splits/mp09_split.json').read_text(encoding='utf-8'))
        ids = split['test_ids']; i = 0
        while True:
            await asyncio.sleep(3.0)
            if not self.clients:
                continue
            y = self.pipe.fe.read_wav(self.pipe.root/'data/raw/mp09_dataset'/f'{ids[i % len(ids)]}.wav')
            await self.broadcast(await asyncio.to_thread(self.classify, y, 'replay'))
            i += 1

    # ------------------------------------------------------------------------- mic kontinu
    def mic_thread(self):
        """Endpoint trigger seperti desain final MP16 v2: buka saat energi > lantai derau + gate, tutup saat
        energi turun mendekati lantai derau ATAU turun >= 20 dB dari puncak; lalu klasifikasi jendela 1,2 s."""
        import sounddevice as sd
        ring = deque(maxlen=int(2.5*FS/BLOCK))
        floor_db, active, peak_db, quiet, since_open = -70.0, False, -120.0, 0, 0
        warm: list = []
        GATE_DB, END_DB, DROP_DB = self.args.gate_db, 6.0, 20.0
        with sd.InputStream(samplerate=FS, channels=1, dtype='float32', blocksize=BLOCK, device=self.args.device) as st:
            print(f'[mic] aktif: {sd.query_devices(self.args.device, "input")["name"]}', flush=True)
            while True:
                x, _ = st.read(BLOCK); x = x[:, 0].astype(np.float64)
                self.fsm.last_audio = time.monotonic()
                ring.append(x)
                db = 20*np.log10(np.sqrt(np.mean(x**2))+1e-9)
                warm.append(db)
                if len(warm) <= 10:                      # 1 s pertama: kalibrasi lantai derau (median)
                    floor_db = float(np.median(warm))
                    self.emit({'type': 'level', 'rms_db': round(float(db), 1), 'vad': False})
                    continue
                if not active:                           # lantai derau adaptif: turun cepat, naik pelan (0,5 dB/s)
                    floor_db = 0.8*floor_db+0.2*db if db < floor_db else min(floor_db+0.05, db)
                self.emit({'type': 'level', 'rms_db': round(float(db), 1), 'vad': bool(active)})
                if not active and db > floor_db+GATE_DB:
                    active, peak_db, quiet, since_open = True, db, 0, 0
                elif active:
                    since_open += 1; peak_db = max(peak_db, db)
                    quiet = quiet+1 if (db < floor_db+END_DB or db < peak_db-DROP_DB) else 0
                    if quiet >= 2 or since_open >= 15:                                       # 200 ms hening / maks 1,5 s
                        active = False
                        y = np.concatenate(ring)
                        e = np.convolve(y**2, np.ones(CLIP_N), 'valid')[::160] if len(y) > CLIP_N else None
                        s0 = int(np.argmax(e))*160 if e is not None else 0
                        clip = np.pad(y, (0, max(0, CLIP_N-len(y))))[s0:s0+CLIP_N]
                        self.emit(self.classify(clip, 'board_mic'))
                        ring.clear()

    async def main(self):
        self.loop = asyncio.get_running_loop()
        tasks = [asyncio.create_task(self.housekeeping())]
        if self.args.replay:
            tasks.append(asyncio.create_task(self.replay()))
        if self.args.mic:
            threading.Thread(target=self.mic_thread, daemon=True).start()
        async with serve(self.handler, self.args.host, self.args.port, process_request=self.process_request, max_size=2**20):
            info = f'DS-CNN INT8 w{self.dl.width}, ambang penutur {self.dl.spk_thr:.2f}' if self.dl else f'ambang OOD {self.pipe.ood_thr:.2f}'
            print(f'Board tiruan siap: http://localhost:{self.args.port}  (WebSocket /ws) | {info} | Ctrl+C untuk berhenti', flush=True)
            await asyncio.Future()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(HERE.parent/'Mini Project 16'), help='folder MP16 (artefak beku)')
    ap.add_argument('--host', default='0.0.0.0'); ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--mic', action='store_true', help='pakai mikrofon PC/Pi (deteksi kontinu)')
    ap.add_argument('--device', default=None, help='nama/indeks device mic (sounddevice)')
    ap.add_argument('--gate-db', type=float, default=12.0, help='ambang buka endpoint di atas lantai derau')
    ap.add_argument('--replay', action='store_true', help='putar WAV test bergiliran tiap 3 s')
    ap.add_argument('--device-name', default='PC (board tiruan, BUKAN ESP32)')
    ap.add_argument('--model', choices=['dl', 'classical'], default='dl', help='dl = DS-CNN INT8 siklus 2 (default), classical = MP16')
    ap.add_argument('--kw-min-prob', type=float, default=0.60)
    args = ap.parse_args()
    pipe = Pipeline(Path(args.root))
    dl = DLPipeline(HERE.parent/'demo_ai', pipe) if args.model == 'dl' else None
    board = Board(pipe, args, dl)
    try:
        asyncio.run(board.main())
    except KeyboardInterrupt:
        print('\nberhenti.')


if __name__ == '__main__':
    main()
