#!/usr/bin/env python3
"""
ハードスタイル・シンセ（二階微分方程式ベース）  Kick / Rave Lead + コード演奏

共通の原理:  x'' + 2*zeta*w(t)*x' + w(t)^2 * (x - u(t)) = 0

  Kick : u(t)=0, 初期変位 x(0)=1 の自由減衰振動。w(t) をピッチ下降させる。
  Lead : u(t)=コード全音 x ユニゾン・ノコギリ波 の強制振動 (= 共振ローパス)。
         w(t)=カットオフ, zeta=レゾナンス。ピッチスクープ(Hoover風の
         ダイブ)とサブレイヤーでレイヴ・スタブにする。

必要: numpy, matplotlib (tkinter は標準)
任意: sounddevice (再生に必要)   pip install numpy matplotlib sounddevice
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
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ------------------------------------------------------------ Music data
CHORD_TYPES = {
    "Major": (0, 4, 7), "Minor": (0, 3, 7), "Sus2": (0, 2, 7), "Sus4": (0, 5, 7),
    "Power (5th)": (0, 7, 12), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "Maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "7": (0, 4, 7, 10),
    "m7b5": (0, 3, 6, 10), "add9": (0, 4, 7, 14), "m9": (0, 3, 7, 10, 14),
    "Octave": (0, 12),
}
REV_TYPES = {v: k for k, v in CHORD_TYPES.items()}
CHORD_SUFFIX = {"Major": "", "Minor": "m", "dim": "dim", "aug": "aug",
                "Maj7": "maj7", "m7": "m7", "7": "7", "m7b5": "m7b5"}
MINORISH = {"Minor", "dim", "m7", "m7b5", "m9"}

SCALES = {
    "Natural Minor": [0, 2, 3, 5, 7, 8, 10],
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
}
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]

# コード進行 (スケール度数, 0始まり)
PROGRESSIONS = {
    "i-VI-III-VII": [0, 5, 2, 6],
    "i-VII-VI-VII": [0, 6, 5, 6],
    "i-iv-VII-III": [0, 3, 6, 2],
    "i-VI-iv-v": [0, 5, 3, 4],
    "I-V-vi-IV": [0, 4, 5, 3],
    "vi-IV-I-V": [5, 3, 0, 4],
}
# パターン: (ヒット位置[16分音符単位, 1小節=16], ゲート長[16分単位])
PATTERNS = {
    "Sustain (全音)": ([0], 16),
    "Offbeat 8th": ([2, 6, 10, 14], 2),
    "4つ打ちスタブ": ([0, 4, 8, 12], 3),
    "Rave 3-3-2": ([0, 6, 12], 4),
    "16th スタブ": (list(range(16)), 1),
}


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
    """コード(p['notes'])を共振器(二階ODE)に通すレイヴ・リード。"""
    os_ = 2                          # 2倍オーバーサンプリング
    sro = sr * os_
    n = int(p["dur"] * sr) * os_
    dt = 1.0 / sro
    t = np.arange(n) * dt
    rng = np.random.default_rng(1)

    notes = list(p["notes"])
    voices = int(round(p["unison"]))
    cents = np.array([0.0]) if voices == 1 else np.linspace(-1, 1, voices) * p["detune"]
    # ピッチスクープ: 開始時に scoop 半音ずれて 0 へ収束 (Hoover風ダイブ)
    bend = 2.0 ** (p["scoop"] / 12.0 * np.exp(-t / (p["scoop_tau"] / 1000.0)))

    # --- 駆動力 u(t): 全コード音 x ユニゾン ---
    u = np.zeros(n)
    for note in notes:
        f0 = 440.0 * 2 ** ((note - 69) / 12.0)
        for c in cents:
            f = f0 * 2 ** (c / 1200.0) * bend
            phase = rng.uniform(0, 1) + np.cumsum(f * dt)
            u += 2.0 * (phase % 1.0) - 1.0
    u /= len(notes) * voices
    if p["sub"] > 0:
        f0 = 440.0 * 2 ** ((min(notes) - 12 - 69) / 12.0)
        phase = np.cumsum(f0 * bend * dt)
        u += p["sub"] * (2.0 * (phase % 1.0) - 1.0)

    # --- 共振器 (二階ODE), カットオフ w(t) を時間変化 ---
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

    env = np.ones(m)
    a_n = max(1, int(p["attack"] / 1000.0 * sr))
    r_n = min(m, max(1, int(p["release"] / 1000.0 * sr)))
    env[:a_n] = np.linspace(0, 1, a_n)
    env[-r_n:] *= np.linspace(1, 0, r_n)

    out /= np.max(np.abs(out)) + 1e-12
    if p["drive"] > 0:
        g = 1.0 + p["drive"]
        out = np.tanh(g * out) / np.tanh(g)
    out *= env
    out /= np.max(np.abs(out)) + 1e-12
    return (out * 0.9).astype(np.float32)


def render_progression(p, chords, pattern, bpm, sr=SR):
    """各コードをパターンに従って配置し、ループ可能な1本の波形にする。"""
    hits, gate = pattern
    step = 60.0 / bpm / 4.0                        # 16分音符の長さ(秒)
    tail = p["release"] / 1000.0
    stab_dur = gate * step + tail
    loop_len = int(16 * step * len(chords) * sr)
    buf = np.zeros(loop_len + int(stab_dur * sr) + sr // 10)

    for ci, notes in enumerate(chords):
        q = dict(p, notes=notes, dur=stab_dur)
        stab = synth_lead(q, sr)                   # コードごとに1回だけ生成
        for h in hits:
            s = int((ci * 16 + h) * step * sr)
            seg = stab[:len(buf) - s]
            buf[s:s + len(seg)] += seg

    out = buf[:loop_len].copy()
    wrap = buf[loop_len:]                          # 余韻を先頭へ回し込み(継ぎ目なしループ)
    k = min(len(wrap), loop_len)
    out[:k] += wrap[:k]
    out /= np.max(np.abs(out)) + 1e-12
    return (out * 0.9).astype(np.float32)


# ------------------------------------------------------- Parameter table
# (ラベル, キー, 最小, 最大, 初期値, 刻み)
INSTRUMENTS = {
    "Kick": {
        "synth": synth_kick,
        "params": [
            ("固有振動数 f_end (Hz)", "f_end", 30, 120, 50, 0.5),
            ("初期周波数 f_start (Hz)", "f_start", 100, 1000, 350, 1),
            ("ピッチ下降 τ (ms)", "tau", 2, 150, 25, 0.5),
            ("減衰比 ζ", "zeta", 0.001, 0.2, 0.012, 0.001),
            ("長さ (秒)", "dur", 0.2, 1.5, 0.6, 0.01),
            ("ディストーション", "drive", 0, 30, 8, 0.1),
            ("クリック量", "click", 0, 1, 0.15, 0.01),
        ],
    },
    "Lead": {
        "synth": synth_lead,
        "params": [
            ("ユニゾン数", "unison", 1, 9, 7, 1),
            ("デチューン (cent)", "detune", 0, 60, 28, 0.5),
            ("カットオフ開始 (Hz)", "f_start", 200, 8000, 7000, 10),
            ("固有振動数/カットオフ終了 (Hz)", "f_end", 100, 6000, 1500, 10),
            ("カットオフ下降 τ (ms)", "tau", 5, 1000, 120, 1),
            ("減衰比 ζ (小=レゾナンス強)", "zeta", 0.02, 1.0, 0.12, 0.01),
            ("ピッチスクープ (半音, +で上からダイブ)", "scoop", -12, 12, 4, 0.5),
            ("スクープ時間 (ms)", "scoop_tau", 5, 300, 70, 1),
            ("サブ(1oct下)ミックス", "sub", 0, 1, 0.3, 0.01),
            ("ディストーション", "drive", 0, 30, 7, 0.1),
            ("アタック (ms)", "attack", 1, 300, 2, 1),
            ("リリース (ms)", "release", 5, 800, 100, 1),
            ("長さ (秒, コード単発用)", "dur", 0.1, 3.0, 0.6, 0.01),
        ],
    },
}


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
        self.title("Hardstyle Synth - Kick / Rave Lead (Damped Oscillator)")
        self.geometry("1000x940")
        self.data = None
        self._job = None
        self.panels = {}
        self.current = None

        # --- 上部バー ---
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Instrument:").pack(side=tk.LEFT)
        self.inst_var = tk.StringVar(value="Lead")
        cb = ttk.Combobox(top, textvariable=self.inst_var, state="readonly",
                          values=list(INSTRUMENTS.keys()), width=12)
        cb.pack(side=tk.LEFT, padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.select(self.inst_var.get()))
        ttk.Button(top, text="▶ 再生", command=self.play_current).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="■ 停止", command=self.stop).pack(side=tk.LEFT)
        ttk.Button(top, text="WAV保存", command=self.save).pack(side=tk.LEFT, padx=8)

        self.panel_host = ttk.Frame(self, padding=(8, 0))
        self.panel_host.pack(side=tk.TOP, fill=tk.X)

        self.build_chord_panel()

        self.fig = Figure(figsize=(8, 3.2), tight_layout=True)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.select("Lead")

    # ------------------------------------------------ スライダーパネル
    def build_panel(self, name):
        frame = ttk.Frame(self.panel_host)
        vars_ = {}
        for r, (label, key, lo, hi, init, step) in enumerate(INSTRUMENTS[name]["params"]):
            var = tk.DoubleVar(value=init)
            vars_[key] = (var, step)
            ttk.Label(frame, text=label, width=36).grid(row=r, column=0, sticky="w")
            ttk.Scale(frame, from_=lo, to=hi, variable=var, length=440,
                      command=lambda _=None: self.schedule()).grid(row=r, column=1, padx=6)
            val = ttk.Label(frame, width=10)
            val.grid(row=r, column=2, sticky="w")

            def upd(*_, v=var, l=val, s=step):
                l.config(text=f"{self.quantize(v.get(), s):g}")
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
        if name == "Lead":
            self.chord_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4,
                                  after=self.panel_host)
        else:
            self.chord_frame.pack_forget()
        self.render()

    def params(self):
        vars_ = self.panels[self.current][1]
        p = {k: self.quantize(v.get(), s) for k, (v, s) in vars_.items()}
        if self.current == "Lead":
            p["notes"] = self.manual_chord_notes()
        return p

    # ------------------------------------------------ コード / 演奏パネル
    def build_chord_panel(self):
        f = ttk.LabelFrame(self, text="コード / 演奏", padding=6)
        self.chord_frame = f

        def combo(parent, var, values, width):
            return ttk.Combobox(parent, textvariable=var, values=values,
                                width=width, state="readonly")

        # 行1: キー・スケール
        r1 = ttk.Frame(f)
        r1.pack(fill=tk.X, pady=2)
        self.key_var = tk.StringVar(value="A")
        self.key_oct = tk.StringVar(value="3")
        self.scale_var = tk.StringVar(value="Natural Minor")
        self.seventh_var = tk.BooleanVar(value=False)
        ttk.Label(r1, text="Key").pack(side=tk.LEFT)
        c = combo(r1, self.key_var, NOTE_NAMES, 5)
        c.pack(side=tk.LEFT, padx=3)
        c.bind("<<ComboboxSelected>>", lambda e: self.refresh_diatonic())
        ttk.Label(r1, text="Oct").pack(side=tk.LEFT)
        c = combo(r1, self.key_oct, ["2", "3", "4"], 3)
        c.pack(side=tk.LEFT, padx=3)
        c.bind("<<ComboboxSelected>>", lambda e: self.refresh_diatonic())
        ttk.Label(r1, text="Scale").pack(side=tk.LEFT, padx=(8, 0))
        c = combo(r1, self.scale_var, list(SCALES.keys()), 14)
        c.pack(side=tk.LEFT, padx=3)
        c.bind("<<ComboboxSelected>>", lambda e: self.refresh_diatonic())
        ttk.Checkbutton(r1, text="7th", variable=self.seventh_var,
                        command=self.refresh_diatonic).pack(side=tk.LEFT, padx=8)

        # 行2: ダイアトニックコード鍵盤
        r2 = ttk.Frame(f)
        r2.pack(fill=tk.X, pady=4)
        self.dia_btns = []
        for d in range(7):
            b = ttk.Button(r2, width=9, command=lambda d=d: self.play_degree(d))
            b.pack(side=tk.LEFT, padx=2)
            self.dia_btns.append(b)

        # 行3: 手動コード
        r3 = ttk.Frame(f)
        r3.pack(fill=tk.X, pady=2)
        self.root_var = tk.StringVar(value="A")
        self.oct_var = tk.StringVar(value="3")
        self.type_var = tk.StringVar(value="Minor")
        ttk.Label(r3, text="コード Root").pack(side=tk.LEFT)
        for var, vals, w in ((self.root_var, NOTE_NAMES, 5),):
            c = combo(r3, var, vals, w)
            c.pack(side=tk.LEFT, padx=3)
            c.bind("<<ComboboxSelected>>", lambda e: self.render())
        ttk.Label(r3, text="Oct").pack(side=tk.LEFT)
        c = combo(r3, self.oct_var, ["1", "2", "3", "4", "5", "6"], 3)
        c.pack(side=tk.LEFT, padx=3)
        c.bind("<<ComboboxSelected>>", lambda e: self.render())
        ttk.Label(r3, text="Type").pack(side=tk.LEFT)
        c = combo(r3, self.type_var, list(CHORD_TYPES.keys()), 12)
        c.pack(side=tk.LEFT, padx=3)
        c.bind("<<ComboboxSelected>>", lambda e: self.render())
        ttk.Button(r3, text="▶ コード再生", command=self.play_chord).pack(side=tk.LEFT, padx=8)

        # 行4: コード進行
        r4 = ttk.Frame(f)
        r4.pack(fill=tk.X, pady=4)
        self.prog_var = tk.StringVar(value="i-VI-III-VII")
        self.pat_var = tk.StringVar(value="Rave 3-3-2")
        self.bpm_var = tk.IntVar(value=150)
        self.loop_var = tk.BooleanVar(value=True)
        ttk.Label(r4, text="進行").pack(side=tk.LEFT)
        combo(r4, self.prog_var, list(PROGRESSIONS.keys()), 13).pack(side=tk.LEFT, padx=3)
        ttk.Label(r4, text="パターン").pack(side=tk.LEFT)
        combo(r4, self.pat_var, list(PATTERNS.keys()), 14).pack(side=tk.LEFT, padx=3)
        ttk.Label(r4, text="BPM").pack(side=tk.LEFT)
        ttk.Spinbox(r4, from_=80, to=220, textvariable=self.bpm_var, width=5).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(r4, text="ループ", variable=self.loop_var).pack(side=tk.LEFT, padx=4)
        ttk.Button(r4, text="▶ 進行を再生", command=self.play_progression).pack(side=tk.LEFT, padx=6)

        self.refresh_diatonic()

    # --- 音楽ロジック ---
    def manual_chord_notes(self):
        root = 12 * (int(self.oct_var.get()) + 1) + NOTE_NAMES.index(self.root_var.get())
        return [root + i for i in CHORD_TYPES[self.type_var.get()]]

    def degree_chord(self, d):
        """スケール度数 d のダイアトニックコード -> (root_midi, type名, intervals)"""
        scale = SCALES[self.scale_var.get()]
        ks = (0, 1, 2, 3) if self.seventh_var.get() else (0, 1, 2)
        absn = [scale[(d + 2 * k) % 7] + 12 * ((d + 2 * k) // 7) for k in ks]
        iv = tuple(a - absn[0] for a in absn)
        base = 12 * (int(self.key_oct.get()) + 1) + NOTE_NAMES.index(self.key_var.get())
        root = base + scale[d] - (12 if scale[d] > 6 else 0)   # 音域を揃える
        return root, REV_TYPES.get(iv, "Major"), iv

    def degree_label(self, d):
        root, t, _ = self.degree_chord(d)
        num = ROMAN[d].lower() if t in MINORISH else ROMAN[d]
        acc = "b" if SCALES[self.scale_var.get()][d] < MAJOR_SCALE[d] else ""
        return f"{acc}{num}\n{NOTE_NAMES[root % 12]}{CHORD_SUFFIX.get(t, t)}"

    def refresh_diatonic(self):
        for d, b in enumerate(self.dia_btns):
            b.config(text=self.degree_label(d))

    def play_degree(self, d):
        root, t, _ = self.degree_chord(d)
        self.root_var.set(NOTE_NAMES[root % 12])
        self.oct_var.set(str(root // 12 - 1))
        self.type_var.set(t)
        self.render()
        self.play(self.data)

    def play_chord(self):
        self.flush()
        self.play(self.data)

    def play_progression(self):
        p = self.params()
        chords = []
        for d in PROGRESSIONS[self.prog_var.get()]:
            root, _, iv = self.degree_chord(d)
            chords.append([root + i for i in iv])
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.data = render_progression(p, chords, PATTERNS[self.pat_var.get()],
                                           self.bpm_var.get())
        finally:
            self.config(cursor="")
        self.plot("Progression")
        self.play(self.data, loop=self.loop_var.get())

    # ------------------------------------------------ 描画・再生・保存
    def schedule(self):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(150, self.render)

    def flush(self):
        if self._job or self.data is None:
            self.render()

    def render(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.data = INSTRUMENTS[self.current]["synth"](self.params())
        finally:
            self.config(cursor="")
        self.plot(self.current)

    def plot(self, title):
        t = np.arange(len(self.data)) / SR
        self.ax1.clear()
        self.ax1.plot(t, self.data, lw=0.5)
        self.ax1.set_title(f"{title} - Waveform")
        self.ax2.clear()
        self.ax2.specgram(self.data, NFFT=1024, Fs=SR, noverlap=768, cmap="magma")
        self.ax2.set_ylim(0, 1500 if self.current == "Kick" else 8000)
        self.ax2.set_title("Spectrogram")
        self.canvas.draw_idle()

    def play(self, data, loop=False):
        if sd is None:
            messagebox.showinfo("再生不可", "pip install sounddevice を実行してください。")
            return
        sd.stop()
        sd.play(data, SR, loop=loop)

    def play_current(self):
        self.flush()
        self.play(self.data)

    def stop(self):
        if sd is not None:
            sd.stop()

    def save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", initialfile=f"{self.current.lower()}.wav",
            filetypes=[("WAV", "*.wav")])
        if path:
            save_wav(path, self.data)


if __name__ == "__main__":
    App().mainloop()
