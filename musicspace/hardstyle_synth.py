#!/usr/bin/env python3
"""
ハードスタイル・インストゥルメント・ジェネレータ（二階微分方程式ベース）

共通の原理:  x'' + 2*zeta*w(t)*x' + w(t)^2 * (x - u(t)) = 0

  Kick : u(t)=0, 初期変位 x(0)=1 の自由減衰振動。
         w(t) をピッチ下降させて「ドン」を作る。
  Lead : u(t)=ユニゾン・ノコギリ波(スーパーソウ) の強制振動。
         これは共振ローパスフィルタそのもの。w(t) がカットオフ、zeta がレゾナンス。

必要: numpy, matplotlib (tkinter は標準)
任意: sounddevice (再生ボタン用)
    pip install numpy matplotlib sounddevice

新しい楽器の追加方法: INSTRUMENTS に {"params": [...], "synth": 関数} を足すだけ。
"""
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import wave

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import sounddevice as sd
except Exception:
    sd = None

SR = 44100
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------- Synths
def synth_kick(p, sr=SR):
    n = int(p["dur"] * sr)
    dt = 1.0 / sr
    t = np.arange(n) * dt
    w = 2 * np.pi * (p["f_end"] + (p["f_start"] - p["f_end"])
                     * np.exp(-t / (p["tau"] / 1000.0)))
    zeta = p["zeta"]

    out = np.empty(n)
    x, v = 1.0, 0.0
    for i, wi in enumerate(w.tolist()):
        a = -2.0 * zeta * wi * v - wi * wi * x
        v += a * dt          # 半陰的オイラー法
        x += v * dt
        out[i] = x

    out /= np.max(np.abs(out)) + 1e-12
    if p["drive"] > 0:
        g = 1.0 + p["drive"]
        out = np.tanh(g * out) / np.tanh(g)
    if p["click"] > 0:
        rng = np.random.default_rng(0)
        out += p["click"] * rng.uniform(-1, 1, n) * np.exp(-t / 0.002)

    fade = int(0.01 * sr)
    out[-fade:] *= np.linspace(1, 0, fade)
    out /= np.max(np.abs(out)) + 1e-12
    return (out * 0.95).astype(np.float32)


def synth_lead(p, sr=SR):
    os_ = 2                      # 2倍オーバーサンプリング（数値安定性とエイリアス対策）
    sro = sr * os_
    n = int(p["dur"] * sr) * os_
    dt = 1.0 / sro
    t = np.arange(n) * dt
    rng = np.random.default_rng(1)

    # --- 駆動力 u(t): ユニゾン・スーパーソウ ---
    f0 = 440.0 * 2 ** ((p["note"] - 69) / 12.0)
    voices = int(round(p["unison"]))
    if voices == 1:
        cents = np.array([0.0])
    else:
        cents = np.linspace(-1, 1, voices) * p["detune"]
    u = np.zeros(n)
    for c in cents:
        f = f0 * 2 ** (c / 1200.0)
        phase = rng.uniform(0, 1) + np.cumsum(np.full(n, f * dt))
        u += 2.0 * (phase % 1.0) - 1.0
    u /= voices

    # --- 共振器（二階ODE）。カットオフを時間変化させる ---
    w = 2 * np.pi * (p["f_end"] + (p["f_start"] - p["f_end"])
                     * np.exp(-t / (p["tau"] / 1000.0)))
    zeta = p["zeta"]
    y = np.empty(n)
    x, v = 0.0, 0.0
    for i, (wi, ui) in enumerate(zip(w.tolist(), u.tolist())):
        a = wi * wi * (ui - x) - 2.0 * zeta * wi * v
        v += a * dt
        x += v * dt
        y[i] = x

    out = y.reshape(-1, os_).mean(axis=1)       # デシメーション
    out -= out.mean()
    m = len(out)

    # --- 音量エンベロープ ---
    env = np.ones(m)
    a_n = max(1, int(p["attack"] / 1000.0 * sr))
    r_n = max(1, int(p["release"] / 1000.0 * sr))
    env[:a_n] = np.linspace(0, 1, a_n)
    r_n = min(r_n, m)
    env[-r_n:] *= np.linspace(1, 0, r_n)

    out /= np.max(np.abs(out)) + 1e-12
    if p["drive"] > 0:
        g = 1.0 + p["drive"]
        out = np.tanh(g * out) / np.tanh(g)
    out *= env
    out /= np.max(np.abs(out)) + 1e-12
    return (out * 0.9).astype(np.float32)


# ------------------------------------------------------- Parameter table
def fmt_note(v):
    n = int(round(v))
    hz = 440.0 * 2 ** ((n - 69) / 12.0)
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1} ({hz:.1f}Hz)"


# (ラベル, キー, 最小, 最大, 初期値, 刻み, 表示フォーマッタ(任意))
INSTRUMENTS = {
    "Kick": {
        "synth": synth_kick,
        "params": [
            ("固有振動数 f_end (Hz)", "f_end", 30, 120, 50, 0.5, None),
            ("初期周波数 f_start (Hz)", "f_start", 100, 1000, 350, 1, None),
            ("ピッチ下降 τ (ms)", "tau", 2, 150, 25, 0.5, None),
            ("減衰比 ζ", "zeta", 0.001, 0.2, 0.012, 0.001, None),
            ("長さ (秒)", "dur", 0.2, 1.5, 0.6, 0.01, None),
            ("ディストーション", "drive", 0, 30, 8, 0.1, None),
            ("クリック量", "click", 0, 1, 0.15, 0.01, None),
        ],
    },
    "Lead": {
        "synth": synth_lead,
        "params": [
            ("ノート (MIDI)", "note", 36, 96, 57, 1, fmt_note),
            ("ユニゾン数", "unison", 1, 9, 7, 1, None),
            ("デチューン (cent)", "detune", 0, 60, 25, 0.5, None),
            ("カットオフ開始 (Hz)", "f_start", 200, 8000, 6000, 10, None),
            ("固有振動数/カットオフ終了 (Hz)", "f_end", 100, 6000, 1200, 10, None),
            ("カットオフ下降 τ (ms)", "tau", 5, 1000, 150, 1, None),
            ("減衰比 ζ (小=レゾナンス強)", "zeta", 0.02, 1.0, 0.15, 0.01, None),
            ("ディストーション", "drive", 0, 30, 6, 0.1, None),
            ("アタック (ms)", "attack", 1, 300, 5, 1, None),
            ("リリース (ms)", "release", 5, 800, 120, 1, None),
            ("長さ (秒)", "dur", 0.2, 3.0, 1.0, 0.01, None),
        ],
    },
}


# -------------------------------------------------------------------- IO
def save_wav(path, data, sr=SR):
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


# ------------------------------------------------------------------- GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hardstyle Instrument Generator (Damped Oscillator)")
        self.geometry("920x780")
        self.data = None
        self._job = None
        self.panels = {}   # instrument名 -> (frame, {key: (var, step)})
        self.current = None

        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Instrument:").pack(side=tk.LEFT)
        self.inst_var = tk.StringVar(value="Kick")
        cb = ttk.Combobox(top, textvariable=self.inst_var, state="readonly",
                          values=list(INSTRUMENTS.keys()), width=12)
        cb.pack(side=tk.LEFT, padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.select(self.inst_var.get()))
        ttk.Button(top, text="▶ 再生", command=self.play).pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text="WAV保存", command=self.save).pack(side=tk.LEFT)

        self.panel_host = ttk.Frame(self, padding=8)
        self.panel_host.pack(side=tk.TOP, fill=tk.X)

        self.fig = Figure(figsize=(8, 4), tight_layout=True)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.select("Kick")

    # --- パネル構築（楽器ごとに1回だけ作り、値を保持する） ---
    def build_panel(self, name):
        frame = ttk.Frame(self.panel_host)
        vars_ = {}
        for r, (label, key, lo, hi, init, step, fmt) in enumerate(INSTRUMENTS[name]["params"]):
            var = tk.DoubleVar(value=init)
            vars_[key] = (var, step)
            ttk.Label(frame, text=label, width=30).grid(row=r, column=0, sticky="w")
            ttk.Scale(frame, from_=lo, to=hi, variable=var, length=420,
                      command=lambda _=None: self.schedule()).grid(row=r, column=1, padx=6)
            val = ttk.Label(frame, width=16)
            val.grid(row=r, column=2, sticky="w")

            def upd(*_, v=var, l=val, s=step, f=fmt):
                q = self.quantize(v.get(), s)
                l.config(text=f(q) if f else f"{q:g}")
            var.trace_add("write", upd)
            upd()
        self.panels[name] = (frame, vars_)

    @staticmethod
    def quantize(v, step):
        return round(v / step) * step

    def select(self, name):
        if self.current:
            self.panels[self.current][0].pack_forget()
        if name not in self.panels:
            self.build_panel(name)
        self.panels[name][0].pack(fill=tk.X)
        self.current = name
        self.inst_var.set(name)
        self.render()

    def params(self):
        vars_ = self.panels[self.current][1]
        return {k: self.quantize(v.get(), s) for k, (v, s) in vars_.items()}

    def schedule(self):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(150, self.render)

    def render(self):
        self._job = None
        self.data = INSTRUMENTS[self.current]["synth"](self.params())
        t = np.arange(len(self.data)) / SR
        self.ax1.clear()
        self.ax1.plot(t, self.data, lw=0.5)
        self.ax1.set_title(f"{self.current} - Waveform")
        self.ax2.clear()
        self.ax2.specgram(self.data, NFFT=1024, Fs=SR, noverlap=768, cmap="magma")
        self.ax2.set_ylim(0, 1500 if self.current == "Kick" else 8000)
        self.ax2.set_title("Spectrogram")
        self.canvas.draw_idle()

    def play(self):
        if sd is None:
            messagebox.showinfo("再生不可", "pip install sounddevice を実行してください。")
            return
        sd.stop()
        sd.play(self.data, SR)

    def save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", initialfile=f"{self.current.lower()}.wav",
            filetypes=[("WAV", "*.wav")])
        if path:
            save_wav(path, self.data)


if __name__ == "__main__":
    App().mainloop()
