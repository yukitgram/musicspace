#!/usr/bin/env python3
"""
ハードスタイル・キックジェネレータ（減衰振動方程式ベース）

  x'' + 2*zeta*w(t)*x' + w(t)^2 * x = 0

  w(t) = 2*pi*( f_end + (f_start - f_end) * exp(-t / tau_pitch) )
  - f_end    : 固有振動数 (Hz)  … キックの最終的な低音の音程
  - f_start  : 初期周波数 (Hz)  … ピッチ下降の開始点
  - tau      : ピッチ下降の時定数 (ms)
  - zeta     : 減衰比 (小さいほど長く鳴る)

必要: numpy, matplotlib (tkinter は標準)
任意: sounddevice (インストールすると再生ボタンが使える)
    pip install numpy matplotlib sounddevice
"""
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


def synth_kick(f_start, f_end, tau_ms, zeta, dur, drive, click, sr=SR):
    """減衰振動を数値積分してキックを生成する。"""
    n = int(dur * sr)
    dt = 1.0 / sr
    t = np.arange(n) * dt
    # 時間変化する固有角周波数
    w = 2 * np.pi * (f_end + (f_start - f_end) * np.exp(-t / (tau_ms / 1000.0)))

    out = np.empty(n)
    x, v = 1.0, 0.0  # 初期変位1, 初速0
    for i in range(n):
        a = -2.0 * zeta * w[i] * v - (w[i] ** 2) * x
        v += a * dt          # 半陰的オイラー法
        x += v * dt
        out[i] = x

    # 正規化 → ディストーション
    out /= np.max(np.abs(out)) + 1e-12
    if drive > 0:
        g = 1.0 + drive
        out = np.tanh(g * out) / np.tanh(g)

    # アタックのクリック（短いノイズバースト）
    if click > 0:
        rng = np.random.default_rng(0)
        env = np.exp(-t / 0.002)
        out += click * rng.uniform(-1, 1, n) * env

    # 末尾フェードアウト（プチノイズ防止）
    fade = int(0.01 * sr)
    out[-fade:] *= np.linspace(1, 0, fade)

    out /= np.max(np.abs(out)) + 1e-12
    return (out * 0.95).astype(np.float32)


def save_wav(path, data, sr=SR):
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


class App(tk.Tk):
    # (ラベル, 変数名, 最小, 最大, 初期値, 刻み)
    PARAMS = [
        ("固有振動数 f_end (Hz)", "f_end", 30, 120, 50, 0.5),
        ("初期周波数 f_start (Hz)", "f_start", 100, 1000, 350, 1),
        ("ピッチ下降 τ (ms)", "tau", 2, 150, 25, 0.5),
        ("減衰比 ζ", "zeta", 0.001, 0.2, 0.012, 0.001),
        ("長さ (秒)", "dur", 0.2, 1.5, 0.6, 0.01),
        ("ディストーション", "drive", 0, 30, 8, 0.1),
        ("クリック量", "click", 0, 1, 0.15, 0.01),
    ]

    def __init__(self):
        super().__init__()
        self.title("Hardstyle Kick Generator (Damped Oscillator)")
        self.geometry("900x700")
        self.vars = {}
        self.data = None
        self._job = None

        panel = ttk.Frame(self, padding=8)
        panel.pack(side=tk.TOP, fill=tk.X)
        for r, (label, key, lo, hi, init, step) in enumerate(self.PARAMS):
            var = tk.DoubleVar(value=init)
            self.vars[key] = var
            ttk.Label(panel, text=label, width=24).grid(row=r, column=0, sticky="w")
            ttk.Scale(panel, from_=lo, to=hi, variable=var, length=480,
                      command=lambda _=None: self.schedule()
                      ).grid(row=r, column=1, padx=6)
            ttk.Label(panel, width=8, textvariable=tk.StringVar()).grid(row=r, column=2)
            val = ttk.Label(panel, width=8)
            val.grid(row=r, column=2)
            var.trace_add("write", lambda *_, v=var, l=val: l.config(text=f"{v.get():.3f}"))
            val.config(text=f"{init:.3f}")

        btns = ttk.Frame(self, padding=8)
        btns.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btns, text="▶ 再生", command=self.play).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="WAV保存", command=self.save).pack(side=tk.LEFT, padx=4)

        self.fig = Figure(figsize=(8, 4), tight_layout=True)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.render()

    def schedule(self):
        """スライダー操作中の連続再計算を間引く。"""
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(150, self.render)

    def render(self):
        p = {k: v.get() for k, v in self.vars.items()}
        self.data = synth_kick(p["f_start"], p["f_end"], p["tau"], p["zeta"],
                               p["dur"], p["drive"], p["click"])
        t = np.arange(len(self.data)) / SR
        self.ax1.clear()
        self.ax1.plot(t, self.data, lw=0.6)
        self.ax1.set_title("Waveform")
        self.ax2.clear()
        self.ax2.specgram(self.data, NFFT=1024, Fs=SR, noverlap=768, cmap="magma")
        self.ax2.set_ylim(0, 1500)
        self.ax2.set_title("Spectrogram")
        self.canvas.draw_idle()

    def play(self):
        if sd is None:
            messagebox.showinfo("再生不可", "pip install sounddevice を実行してください。")
            return
        sd.stop()
        sd.play(self.data, SR)

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            filetypes=[("WAV", "*.wav")])
        if path:
            save_wav(path, self.data)


if __name__ == "__main__":
    App().mainloop()
