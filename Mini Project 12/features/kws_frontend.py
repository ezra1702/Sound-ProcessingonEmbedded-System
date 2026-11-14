"""
kws_frontend.py — Front-end KWS beku (MP6 filter + MP7 preprocessing + MP8 feature schema)
==========================================================================================
FITUR TAMBAHAN Mini Project 9. Modul ini memadatkan fungsi `preprocess/extract` MP9 menjadi
satu kelas yang dapat di-import ulang oleh MP13 (robustness test), MP14 (streaming) dan MP16
(integrasi + dashboard), sehingga training dan inference memakai kode front-end yang IDENTIK
(prinsip materi: "preprocessing adalah bagian dari model").

`selftest()` membuktikan modul ini mereproduksi `features/dataset/mp09_features_39d.npz`
dan log-Mel MP9 secara bit-exact (selisih 0).

Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import lfilter


class KWSFrontend:
    TARGET_FRAMES = 96

    def __init__(self, root: str | Path):
        self.root = Path(root)
        r = self.root
        proj = json.loads((r/'config/project_config.json').read_text(encoding='utf-8'))
        self.pre = json.loads((r/'config/preprocessing_config.json').read_text(encoding='utf-8'))
        self.feat = json.loads((r/'config/feature_config.json').read_text(encoding='utf-8'))
        filt = np.load(r/'features/intermediate/mp06_filter_final.npz', allow_pickle=True)
        self.fs = int(proj.get('sampling_rate_hz', proj.get('sample_rate_hz', 16000)))
        self.labels = list(proj.get('labels', ['nyala', 'mati', 'naik', 'turun']))
        self.b = np.asarray(filt['b'], float)
        self.a = np.asarray(filt['a'], float)
        self.N = int(self.pre['frame_length_samples'])
        self.H = int(self.pre['hop_length_samples'])
        self.peak = float(self.pre['normalization']['target_peak'])
        self.alpha = float(self.pre['pre_emphasis_alpha'])
        self.vad_factor = float(self.pre['vad']['relative_factor'])
        self.margin = int(self.pre['vad'].get('margin_frames', 2))
        m = self.feat['mel']
        self.n_mels, self.fmin, self.fmax = int(m['n_mels']), float(m['fmin_hz']), float(m['fmax_hz'])
        self.n_mfcc = int(self.feat['mfcc']['n_mfcc'])
        p = self.feat['pitch']
        self.pmin, self.pmax, self.pconf = float(p['min_hz']), float(p['max_hz']), float(p['confidence_threshold'])
        self.fb = self._melbank(self.N)
        self.freq = np.fft.rfftfreq(self.N, 1/self.fs)
        self.window = np.hamming(self.N)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _hz2mel(f):
        return 2595*np.log10(1+np.asarray(f)/700.0)

    @staticmethod
    def _mel2hz(m):
        return 700*(10**(np.asarray(m)/2595.0)-1)

    def _melbank(self, n_fft):
        m = np.linspace(self._hz2mel(self.fmin), self._hz2mel(self.fmax), self.n_mels+2)
        hz = self._mel2hz(m)
        bins = np.floor((n_fft+1)*hz/self.fs).astype(int)
        fb = np.zeros((self.n_mels, n_fft//2+1))
        for i in range(1, self.n_mels+1):
            l, c, r = bins[i-1], bins[i], bins[i+1]
            c = max(c, l+1); r = max(r, c+1)
            for k in range(l, min(c, fb.shape[1])):
                fb[i-1, k] = (k-l)/(c-l)
            for k in range(c, min(r, fb.shape[1])):
                fb[i-1, k] = (r-k)/(r-c)
        return fb

    def read_wav(self, path):
        fs, y = wavfile.read(path)
        if fs != self.fs:
            raise RuntimeError(f'{path}: fs {fs} != {self.fs}')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if np.issubdtype(y.dtype, np.integer):
            return y.astype(np.float64)/np.iinfo(y.dtype).max
        return y.astype(np.float64)

    # ------------------------------------------------------------------ pipeline
    def preprocess(self, y, return_bounds=False):
        N, H = self.N, self.H
        y = lfilter(self.b, self.a, np.asarray(y, float))
        y = y - np.mean(y)
        y = y*(self.peak/(np.max(np.abs(y))+1e-12))
        if len(y) < N:
            y = np.pad(y, (0, N-len(y)))
        starts = np.arange(0, len(y)-N+1, H)
        raw = np.array([y[s:s+N] for s in starts])
        rms = np.sqrt(np.mean(raw**2, axis=1)+1e-12)
        active = np.where(rms >= self.vad_factor*np.max(rms))[0]
        s0, s1 = 0, len(y)
        if len(active):
            i0 = max(0, int(active[0])-self.margin)
            i1 = min(len(raw)-1, int(active[-1])+self.margin)
            s0, s1 = int(starts[i0]), int(min(len(y), starts[i1]+N))
            y = y[s0:s1]
        if len(y) < N:
            y = np.pad(y, (0, N-len(y)))
        starts = np.arange(0, len(y)-N+1, H)
        fr = np.array([y[s:s+N] for s in starts])
        fr = np.concatenate([fr[:, :1], fr[:, 1:]-self.alpha*fr[:, :-1]], axis=1)
        fr = fr*self.window[None, :]
        return (fr, (s0, s1)) if return_bounds else fr

    def _pitch(self, frame):
        x = frame-np.mean(frame)
        ac = np.correlate(x, x, mode='full')[len(x)-1:]
        if ac[0] <= 1e-12:
            return 0.0
        ac = ac/(ac[0]+1e-12)
        lo = max(1, int(self.fs/self.pmax)); hi = min(len(ac)-1, int(self.fs/self.pmin))
        if hi <= lo:
            return 0.0
        lag = int(np.argmax(ac[lo:hi+1])+lo)
        return 0.0 if float(ac[lag]) < self.pconf else float(self.fs/lag)

    def features(self, y):
        """Return (vec39 float32, logmel T×26 float32)."""
        frames = self.preprocess(y)
        rms = np.sqrt(np.mean(frames**2, axis=1)+1e-12)
        zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
        power = (np.abs(np.fft.rfft(frames, axis=1))+1e-12)**2
        den = np.sum(power, axis=1)+1e-12
        F = self.freq
        centroid = np.sum(power*F[None, :], axis=1)/den
        bw = np.sqrt(np.sum(power*(F[None, :]-centroid[:, None])**2, axis=1)/den)
        cum = np.cumsum(power, axis=1); total = cum[:, -1]
        roll = np.array([F[min(np.searchsorted(cum[i], .85*total[i]), len(F)-1)] for i in range(len(frames))])
        pitch = np.array([self._pitch(f) for f in frames]); voiced = pitch > 0
        logmel = np.log(np.maximum(power@self.fb.T, 1e-12))
        mfcc = dct(logmel, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]
        vals = []
        for j in range(self.n_mfcc):
            vals += [float(np.mean(mfcc[:, j])), float(np.std(mfcc[:, j]))]
        for a in (rms, zcr, centroid, bw, roll):
            vals += [float(np.mean(a)), float(np.std(a))]
        pv = pitch[voiced]
        vals += [float(np.mean(pv)) if len(pv) else 0.0, float(np.std(pv)) if len(pv) else 0.0, float(np.mean(voiced))]
        v = np.asarray(vals, np.float32)
        if v.shape != (39,) or not np.isfinite(v).all():
            raise RuntimeError('Feature vector tidak valid')
        return v, logmel.astype(np.float32)

    def fixed_logmel(self, logmel):
        """Crop/pad ke 96 frame dengan nilai minimum (identik MP12/MP13)."""
        x = np.asarray(logmel, np.float32)
        T = self.TARGET_FRAMES
        if x.shape[0] >= T:
            return x[:T]
        fill = float(np.min(x)) if x.size else 0.0
        return np.pad(x, ((0, T-x.shape[0]), (0, 0)), mode='constant', constant_values=fill)

    # ------------------------------------------------------------------ verification
    def selftest(self):
        D = np.load(self.root/'features/dataset/mp09_features_39d.npz', allow_pickle=True)
        max_vec, max_lm = 0.0, 0.0
        for sid, path, x_ref in zip(D['sample_ids'], D['paths'], D['X']):
            v, lm = self.features(self.read_wav(self.root/str(path)))
            lm_ref = np.load(self.root/'features/logmel/mp09'/f'{sid}.npz')['log_mel']
            max_vec = max(max_vec, float(np.max(np.abs(v-x_ref))))
            max_lm = max(max_lm, float(np.max(np.abs(lm-lm_ref))) if lm.shape == lm_ref.shape else float('inf'))
        return {'n_samples': int(len(D['X'])), 'max_abs_diff_vec39': max_vec, 'max_abs_diff_logmel': max_lm,
                'bit_exact': bool(max_vec == 0.0 and max_lm == 0.0)}


if __name__ == '__main__':
    import sys
    fe = KWSFrontend(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
    print(fe.selftest())
