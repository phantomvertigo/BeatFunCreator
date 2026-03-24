import copy
import json
import os
import subprocess
import sys
import tempfile
import time as _time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
import webbrowser

import librosa
import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Polygon as MplPolygon
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv", ".ts", ".m4v"}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

BG_COLOR = "#1e1e2e"
WAVE_COLOR = "#7aa2f7"
BEAT_COLOR = "#f7768e"
BEAT_COLOR_100 = "#f7768e"  # red/pink  — 100 %
BEAT_COLOR_60  = "#ff9e64"  # orange    —  60 %
BEAT_COLOR_20  = "#e0af68"  # yellow    —  20 %
BEAT_INTENSITIES = [0.2, 0.6, 1.0]
BEAT_INTENSITY_COLORS = {1.0: BEAT_COLOR_100, 0.6: BEAT_COLOR_60, 0.2: BEAT_COLOR_20}

def _beat_intensity_color(inten):
    """Return a color for any intensity 0.0–1.0, interpolating between yellow→orange→red."""
    # yellow (#e0af68) at 0, orange (#ff9e64) at 0.5, red (#f7768e) at 1.0
    if inten <= 0.5:
        t = inten / 0.5
        r = int(0xe0 + (0xff - 0xe0) * t)
        g = int(0xaf + (0x9e - 0xaf) * t)
        b = int(0x68 + (0x64 - 0x68) * t)
    else:
        t = (inten - 0.5) / 0.5
        r = int(0xff + (0xf7 - 0xff) * t)
        g = int(0x9e + (0x76 - 0x9e) * t)
        b = int(0x64 + (0x8e - 0x64) * t)
    return f"#{r:02x}{g:02x}{b:02x}"
IN_COLOR = "#9ece6a"
OUT_COLOR = "#f7768e"
GRID_COLOR = "#333346"
TEXT_COLOR = "#a9b1d6"
FUNSCRIPT_COLOR = "#bb9af7"
PLAYHEAD_COLOR = "#2a6ef5"
MARKER_HANDLE_W = 0.008

# Map-pin marker: circle on top, spike pointing down.
# The tip of the spike is at (0, 0) so the marker anchor is the spike tip.
import math as _math
_PIN_VERTS = []
_n = 24
_start_a = -_math.pi / 2 + 0.4  # bottom-right of circle
_end_a = -_math.pi / 2 - 0.4 + 2 * _math.pi  # bottom-left (going CCW through top)
for _i in range(_n + 1):
    _a = _start_a + (_end_a - _start_a) * _i / _n
    _PIN_VERTS.append((_math.cos(_a) * 0.45, _math.sin(_a) * 0.45 + 0.65))
_PIN_VERTS.append((0.0, 0.0))  # spike tip
_PIN_VERTS.append(_PIN_VERTS[0])  # close
_PIN_CODES = [MplPath.MOVETO] + [MplPath.LINETO] * (len(_PIN_VERTS) - 2) + [MplPath.CLOSEPOLY]
PIN_MARKER = MplPath(_PIN_VERTS, _PIN_CODES)
ZONE_COLOR = "#565f89"
TEMPO_GRID_COLOR = "#3d59a1"   # subtle blue for musical beat grid lines
KF_COLOR_ORIGINAL = "#f7768e"     # red   — beat-generated keyframe (unmodified)
KF_COLOR_MODIFIED = "#7aa2f7"     # blue   — beat keyframe that was moved
KF_COLOR_INDEPENDENT = "#9ece6a"  # green  — manually added keyframe (no beat)

MODE_PEAK = "Peak"
MODE_ALTERNATING = "Alternating"
# Alternating sub-modes
ALT_FORCE = "Force direction change"
ALT_KEEP = "Keep direction if short"

APP_VERSION = "1.0.0"
DONATE_URL = "https://ko-fi.com/phantomvertigo"
GITHUB_URL = "https://github.com/phantomvertigo/BeatFunCreator"

# Max points for waveform envelope display — higher = sharper peaks when
# zoomed in but slightly more memory.  20 000 gives ~10 ms resolution for
# a typical 3‑minute track, close to the detection hop (5 ms).
ENVELOPE_MAX_POINTS = 20000

# Default slider values
DEFAULTS = {
    "threshold": 0.7,
    "min_beat_gap": 0.0,
    "speed": 0.5,
    "pos_min": 0.0,
    "pos_max": 1.0,
    "noise_amount": 0.0,
    "bass_only": False,
    "mode": MODE_PEAK,
    "alt_mode": ALT_FORCE,
    "peak_start_at_beat": False,
    "reverse_curve": False,
    "auto_normalize": True,
    "norm_percentile": 100.0,
}

ZONE_MIN_MS = 1000  # minimum zone width in milliseconds

class ToolTip:
    """Simple hover tooltip for a widget."""
    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self._tw = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 2
        self._tw = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self._text, background="#1a1b26",
                         foreground="#a9b1d6", relief="solid", borderwidth=1,
                         font=("Segoe UI", 8), padx=4, pady=2)
        label.pack()

    def _hide(self, event=None):
        if self._tw:
            self._tw.destroy()
            self._tw = None


class CollapsibleFrame(ttk.Frame):
    """A frame with a clickable header that toggles visibility of its content."""

    def __init__(self, parent, text="", collapsed=False, **kwargs):
        super().__init__(parent, **kwargs)
        self._text = text
        self._collapsed = collapsed

        # Header row
        self._header = ttk.Frame(self)
        self._header.pack(fill="x")
        self._toggle_btn = ttk.Label(
            self._header, text=("▶ " if collapsed else "▼ ") + text,
            font=("Segoe UI", 9, "bold"), cursor="hand2")
        self._toggle_btn.pack(side="left", padx=4, pady=2)
        self._toggle_btn.bind("<Button-1>", self._toggle)

        # Content area (a LabelFrame-like inset)
        self.content = ttk.Frame(self)
        if not collapsed:
            self.content.pack(fill="x", padx=2, pady=(0, 2))

    def _toggle(self, event=None):
        self._collapsed = not self._collapsed
        if self._collapsed:
            self.content.pack_forget()
            self._toggle_btn.config(text="▶ " + self._text)
        else:
            self.content.pack(fill="x", padx=2, pady=(0, 2))
            self._toggle_btn.config(text="▼ " + self._text)

    @property
    def collapsed(self):
        return self._collapsed

    @collapsed.setter
    def collapsed(self, value):
        if value != self._collapsed:
            self._toggle()


# Persistent config file — stored next to the script
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bfc_config.json")


def _load_config():
    """Load saved settings (last file path, output dir)."""
    try:
        with open(_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_config(cfg):
    """Persist settings to disk."""
    try:
        with open(_CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def extract_audio_from_video(video_path, progress_cb=None):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    if progress_cb:
        progress_cb("Extracting audio from video…", 5)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "22050", "-ac", "1", tmp.name],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        os.unlink(tmp.name)
        raise RuntimeError("ffmpeg not found. Install ffmpeg to process video files.")
    except subprocess.CalledProcessError as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"ffmpeg error: {e.stderr.decode(errors='replace')}")
    return tmp.name


def load_audio(path, progress_cb=None):
    if progress_cb:
        progress_cb("Loading audio…", 10)
    y, sr = librosa.load(path, sr=22050, mono=True)
    return y, sr


def normalize_audio(y, percentile=100):
    """Compress loud peaks and normalize so the waveform is more even.

    percentile=100 → plain peak normalization (no compression).
    percentile<100 → the lower the value, the more aggressively loud parts
                     are compressed down.  Multiple compression passes are
                     applied so even 50 % produces a very even waveform.
                     No clipping occurs — the result is always peak-normalized.
    """
    peak = np.max(np.abs(y))
    if peak <= 0:
        return y

    if percentile >= 100:
        return (y / peak).astype(np.float32)

    # ── Windowed peak envelope (~10 ms windows) ──
    win = 220  # ~10 ms at 22050 Hz
    n_frames = len(y) // win
    if n_frames < 2:
        return (y / peak).astype(np.float32)

    y_out = y.copy().astype(np.float32)

    # More aggressive compression at lower percentiles:
    # - ratio increases from 4:1 at pct=99 to 20:1 at pct=50
    # - multiple passes (1 at pct=99, up to 4 at pct=50)
    strength = 1.0 - percentile / 100.0  # 0 at 100%, 0.5 at 50%
    ratio = 4.0 + 32.0 * strength
    n_passes = max(1, int(1 + 5 * strength))

    for _ in range(n_passes):
        frames_abs = np.abs(y_out[:n_frames * win]).reshape(n_frames, win)
        peak_env = np.max(frames_abs, axis=1)
        env_max = peak_env.max()
        if env_max <= 0:
            break

        # Only consider frames with actual signal
        active = peak_env[peak_env > env_max * 0.01]
        if len(active) == 0:
            break

        threshold = np.percentile(active, percentile)
        if threshold <= 0:
            break

        # Per-frame gain: compress above threshold
        gain = np.ones(n_frames, dtype=np.float32)
        above = peak_env > threshold
        if np.any(above):
            gain[above] = (threshold * (peak_env[above] / threshold) ** (1.0 / ratio)) / peak_env[above]

        # Smooth gain curve (~50 ms window)
        smooth_win = max(3, int(0.05 * 22050 / win))
        if smooth_win % 2 == 0:
            smooth_win += 1
        kernel = np.ones(smooth_win) / smooth_win
        gain = np.convolve(gain, kernel, mode="same").astype(np.float32)

        # Apply gain per sample
        gain_samples = np.repeat(gain, win)
        y_out[:n_frames * win] *= gain_samples
        if len(y_out) > n_frames * win:
            y_out[n_frames * win:] *= gain[-1]

    # ── Final peak normalization ──
    out_peak = np.max(np.abs(y_out))
    if out_peak > 0:
        y_out /= out_peak

    return y_out


def detect_beats(y, sr, threshold=0.5, in_ms=None, out_ms=None,
                 bass_only=False, min_gap_ms=0, progress_cb=None):
    """Detect amplitude spikes that stand out from their local surroundings.

    Uses the same max-absolute-amplitude envelope as the waveform display
    so detected peaks align exactly with visible spikes.  Each peak is
    then refined to the exact sample with the highest amplitude.

    The *threshold* (0‒1) controls how clearly a spike must stand out
    from its neighbourhood: low = subtle bumps, high = only very
    pronounced spikes.
    """
    if progress_cb:
        progress_cb("Computing amplitude envelope…", 30)

    if in_ms is not None or out_ms is not None:
        start_sample = int((in_ms or 0) / 1000.0 * sr)
        end_sample = int((out_ms or len(y) / sr * 1000) / 1000.0 * sr)
        start_sample = max(0, min(start_sample, len(y)))
        end_sample = max(start_sample, min(end_sample, len(y)))
        y_segment = y[start_sample:end_sample]
        time_offset_ms = (in_ms or 0)
    else:
        y_segment = y
        time_offset_ms = 0
        start_sample = 0

    # Keep the original (unfiltered) segment for refinement — we always
    # want markers to land on peaks visible in the displayed waveform.
    y_original = y_segment

    # Optional low-pass filter: keep only bass frequencies (≤ 200 Hz)
    if bass_only:
        if progress_cb:
            progress_cb("Filtering bass frequencies…", 25)
        from scipy.signal import butter, sosfilt
        nyq = sr / 2.0
        cutoff = min(200, nyq * 0.9)
        sos = butter(4, cutoff / nyq, btype="low", output="sos")
        y_segment = sosfilt(sos, y_segment).astype(y_segment.dtype)

    # Envelope hop — ~5 ms = 110 samples at 22050 Hz
    hop = 110

    # ── Envelope of the ORIGINAL signal ──
    # This is what the user sees.  Every detected beat must sit on a
    # local maximum ("mountain top") of this envelope.
    n_frames = len(y_original) // hop
    if n_frames < 2:
        return []
    orig_trimmed = y_original[:n_frames * hop].reshape(n_frames, hop)
    env_orig = np.max(np.abs(orig_trimmed), axis=1)

    if env_orig.max() <= 0:
        return []

    # ── Envelope used for scoring (filtered when bass_only) ──
    if bass_only:
        n_det = len(y_segment) // hop
        if n_det < 2:
            return []
        det_trimmed = y_segment[:n_det * hop].reshape(n_det, hop)
        env_det = np.max(np.abs(det_trimmed), axis=1)
        # Pad / truncate to match original length
        if len(env_det) < n_frames:
            env_det = np.pad(env_det, (0, n_frames - len(env_det)))
        else:
            env_det = env_det[:n_frames]
    else:
        env_det = env_orig

    if progress_cb:
        progress_cb("Computing local contrast…", 50)

    # Normalised amplitude (0‒1) of the detection envelope
    det_max = env_det.max()
    if det_max <= 0:
        return []
    env_norm = env_det / det_max

    # Local moving average (~300 ms window) as baseline
    win_frames = max(3, int(0.3 * sr / hop))
    if win_frames % 2 == 0:
        win_frames += 1
    kernel = np.ones(win_frames) / win_frames
    local_avg = np.convolve(env_det, kernel, mode="same")

    # Contrast: how much a point stands out from its surroundings
    eps = det_max * 1e-6
    contrast = env_det / (local_avg + eps)

    # Sharpness: how quickly the envelope rises/falls around each point.
    # Computed as average drop to ±2 neighbours, normalised by the peak value.
    sharp = np.zeros_like(env_det, dtype=float)
    for d in range(1, 3):
        left = np.roll(env_det, d)
        right = np.roll(env_det, -d)
        left[:d] = env_det[:d]
        right[-d:] = env_det[-d:]
        sharp += (env_det - left) + (env_det - right)
    sharp /= 4.0  # average over 4 comparisons
    sharp_norm = sharp / (det_max + eps)  # 0‒1

    # Combined score: contrast × (1 + loudness_bonus + sharpness_bonus)
    # Loudness is weighted more heavily than sharpness.
    boosted = contrast * (1.0 + 2.0 * env_norm + 0.5 * np.clip(sharp_norm, 0, 1))

    if progress_cb:
        progress_cb("Finding mountain tops…", 60)

    # ── Step 1: find local maxima in the ORIGINAL envelope ──
    # These are the visible "mountain tops" in the waveform.
    min_distance = max(1, int(0.05 * sr / hop))  # ~50 ms between peaks
    min_env_prom = env_orig.max() * 0.005
    candidates, _ = find_peaks(env_orig, distance=min_distance,
                               prominence=min_env_prom)

    # Verify each candidate is a true mountain top: it must be strictly
    # above the envelope values ~20 ms to its left AND right.  This
    # rejects flat plateaus and float‑noise artefacts.
    half_check = max(3, int(0.02 * sr / hop))
    orig_peaks = []
    for i in candidates:
        lo = max(0, i - half_check)
        hi = min(len(env_orig) - 1, i + half_check)
        if env_orig[i] > env_orig[lo] and env_orig[i] > env_orig[hi]:
            orig_peaks.append(i)
    orig_peaks = np.array(orig_peaks, dtype=int)

    if progress_cb:
        progress_cb("Scoring peaks…", 70)

    # ── Step 2: score each mountain top by its boosted contrast ──
    #   threshold 0   → ≥ 1.3   (subtle bumps)
    #   threshold 0.5 → ≥ 3.9   (moderate)
    #   threshold 1   → ≥ 14.0  (near-perfect spike from silence to peak)
    min_score = 1.3 + 2.7 * threshold + 10.0 * threshold ** 3

    # At high thresholds also require minimum absolute amplitude —
    # threshold 1 demands the peak be at ≥ 60 % of the loudest point.
    min_amp_frac = 0.6 * threshold ** 2  # 0 at t=0, 0.6 at t=1
    min_amp = env_orig.max() * min_amp_frac

    peak_indices = np.array([i for i in orig_peaks
                             if boosted[i] >= min_score
                             and env_orig[i] >= min_amp])

    if progress_cb:
        progress_cb("Refining peak positions…", 80)

    # ── Step 3: refine to the exact sample in the ORIGINAL signal ──
    times_ms = []
    for idx in peak_indices:
        frame_start = idx * hop
        search_start = max(0, frame_start - hop)
        search_end = min(len(y_original), frame_start + 2 * hop)
        local_abs = np.abs(y_original[search_start:search_end])
        exact_sample = search_start + int(np.argmax(local_abs))
        t_ms = exact_sample / sr * 1000 + time_offset_ms
        times_ms.append(int(round(t_ms)))

    # Enforce minimum gap between beats (keep the stronger peaks)
    if min_gap_ms > 0 and len(times_ms) > 1:
        peak_strengths = []
        for i, idx in enumerate(peak_indices):
            peak_strengths.append((times_ms[i], boosted[idx]))
        # Sort by contrast (strongest first), greedily keep non-overlapping
        peak_strengths.sort(key=lambda x: -x[1])
        kept = []
        for t_ms_val, _ in peak_strengths:
            if all(abs(t_ms_val - k) >= min_gap_ms for k in kept):
                kept.append(t_ms_val)
        times_ms = sorted(kept)

    return times_ms


def _dedup_actions(actions):
    """Remove duplicate same-time actions (keep last) and collapse 3+ same-pos runs."""
    if not actions:
        return actions
    deduped = [actions[0]]
    for a in actions[1:]:
        if a["at"] != deduped[-1]["at"]:
            deduped.append(a)
        else:
            deduped[-1] = a
    # Collapse runs of 3+ same-pos: keep first and last only
    cleaned = [deduped[0]]
    for i in range(1, len(deduped)):
        cur = deduped[i]
        if (cur["pos"] == cleaned[-1]["pos"]
                and len(cleaned) >= 2
                and cleaned[-2]["pos"] == cur["pos"]):
            cleaned[-1] = cur  # replace middle with later timestamp
        else:
            cleaned.append(cur)
    return cleaned


def generate_funscript_peak(beat_times_ms, speed=0.5, min_pos=0, max_pos=100,
                            intensities=None, start_at_beat=False):
    """Peak mode: each beat creates a spike. Speed is always preserved.

    *intensities* is a dict {time_ms: 0.2|0.6|1.0}.  A beat with lower
    intensity produces a proportionally smaller stroke.

    *start_at_beat*: if True, the upward movement starts at the beat time
    (beat = valley) instead of reaching the peak at the beat time.
    """
    if not beat_times_ms:
        return {"version": "1.0", "actions": [], "range": 100}
    if intensities is None:
        intensities = {}

    half_stroke_ms = int(50 + (1.0 - speed) * 450)
    pos_range = max_pos - min_pos
    if pos_range <= 0:
        return {"version": "1.0", "actions": [{"at": 0, "pos": min_pos}], "range": 100}

    speed_per_ms = pos_range / half_stroke_ms

    if start_at_beat:
        # ── Start-at-beat mode: beat marks the START of the upstroke ──
        # Same speed as default mode.  Each beat = valley, then the device
        # ascends to peak and descends back to min at constant speed_per_ms.
        # When beats are close together, the up+down is compressed so that
        # the device is back at min_pos by the next beat, still at the same
        # movement speed (just a shorter stroke).
        actions = [{"at": 0, "pos": min_pos}]

        for i, beat_ms in enumerate(beat_times_ms):
            inten = intensities.get(beat_ms, 1.0)
            beat_max = min_pos + pos_range * inten
            stroke_up = beat_max - min_pos
            full_ascent = stroke_up / speed_per_ms if speed_per_ms > 0 else half_stroke_ms
            full_round = full_ascent * 2  # ascent + descent at same speed

            # Time available until next beat (or unlimited if last beat)
            if i + 1 < len(beat_times_ms):
                avail = beat_times_ms[i + 1] - beat_ms
            else:
                avail = full_round + 1  # plenty of room

            if avail <= 0:
                continue

            # Valley at beat time
            actions.append({"at": beat_ms, "pos": min_pos})

            # Minimum visible stroke: at least 5% of range
            min_stroke = max(2, int(pos_range * 0.05))

            if full_round <= avail:
                # Enough room for full stroke
                peak_t = beat_ms + full_ascent
                valley_t = peak_t + full_ascent
                actions.append({"at": int(round(peak_t)), "pos": int(round(beat_max))})
                actions.append({"at": int(round(valley_t)), "pos": min_pos})
            else:
                # Not enough room — use half the available time for ascent,
                # half for descent, at the same speed (shorter stroke).
                half_t = avail / 2.0
                reached = min(beat_max, min_pos + speed_per_ms * half_t)
                # Ensure minimum visible movement
                reached = max(reached, min_pos + min_stroke)
                reached = min(reached, max_pos)
                peak_t = beat_ms + half_t
                valley_t = beat_ms + avail
                actions.append({"at": int(round(peak_t)), "pos": int(round(reached))})
                actions.append({"at": int(round(valley_t)), "pos": min_pos})

        return {"version": "1.0", "actions": _dedup_actions(actions), "range": 100}

    # ── Default mode: beat marks the PEAK ──
    actions = [{"at": 0, "pos": min_pos}]
    prev_peak_time = None
    prev_peak_pos = min_pos  # position of last peak

    for beat_ms in beat_times_ms:
        inten = intensities.get(beat_ms, 1.0)
        beat_max = min_pos + pos_range * inten  # scaled target for this beat

        if prev_peak_time is None:
            # First beat: ascend from min_pos
            stroke_range = beat_max - min_pos
            stroke_ms = int(stroke_range / speed_per_ms) if speed_per_ms > 0 else half_stroke_ms
            avail = beat_ms
            if avail >= stroke_ms:
                approach = beat_ms - stroke_ms
                if approach > 1:
                    actions.append({"at": int(approach), "pos": min_pos})
                actions.append({"at": beat_ms, "pos": int(round(beat_max))})
                prev_peak_pos = int(round(beat_max))
            elif avail > 0:
                peak = min(beat_max, min_pos + speed_per_ms * avail)
                actions.append({"at": beat_ms, "pos": int(round(peak))})
                prev_peak_pos = int(round(peak))
            else:
                actions.append({"at": beat_ms, "pos": min_pos})
                prev_peak_pos = min_pos
            prev_peak_time = beat_ms
            continue

        gap = beat_ms - prev_peak_time
        if gap <= 0:
            continue

        # Time to fully descend from prev peak to min
        descent_full = (prev_peak_pos - min_pos) / speed_per_ms if prev_peak_pos > min_pos else 0
        # Time to ascend from min to this beat's target
        stroke_range = beat_max - min_pos
        ascent_full = stroke_range / speed_per_ms if speed_per_ms > 0 else half_stroke_ms

        if descent_full + ascent_full <= gap:
            # Full descent, hold at min, full ascent
            desc_end = int(prev_peak_time + descent_full)
            asc_start = int(beat_ms - ascent_full)
            actions.append({"at": desc_end, "pos": min_pos})
            if asc_start > desc_end + 1:
                actions.append({"at": asc_start, "pos": min_pos})
            actions.append({"at": beat_ms, "pos": int(round(beat_max))})
            prev_peak_pos = int(round(beat_max))
        else:
            d_for_max = (gap - (beat_max - prev_peak_pos) / speed_per_ms) / 2
            d = max(0, min(d_for_max, descent_full))

            valley = max(min_pos, prev_peak_pos - speed_per_ms * d)
            ascent_time = gap - d
            peak = min(beat_max, valley + speed_per_ms * ascent_time)

            valley_int = int(round(valley))
            peak_int = int(round(peak))

            if d > 1 and valley_int != prev_peak_pos:
                actions.append({"at": int(prev_peak_time + d), "pos": valley_int})
            actions.append({"at": beat_ms, "pos": peak_int})
            prev_peak_pos = peak_int

        prev_peak_time = beat_ms

    # Final descent to min
    if actions[-1]["pos"] > min_pos:
        last = actions[-1]
        dt = (last["pos"] - min_pos) / speed_per_ms
        actions.append({"at": int(last["at"] + dt), "pos": min_pos})

    return {"version": "1.0", "actions": _dedup_actions(actions), "range": 100}


def generate_funscript_alternating(beat_times_ms, speed=0.5, min_pos=0, max_pos=100,
                                   intensities=None, alt_mode=ALT_FORCE,
                                   start_at_beat=False):
    """Alternating mode: each beat moves the position by the beat's intensity.

    Starting from min_pos, the position jumps up/down by
    ``intensity * pos_range`` on every beat.  The speed setting controls
    how fast the device travels between positions.

    ALT_FORCE – always reverse direction on each beat.
    ALT_KEEP  – only reverse if the resulting position stays within
                [min_pos, max_pos]; otherwise move in the direction
                with more room.

    *start_at_beat*: if True the movement starts at the beat time; if False
    the movement arrives at the target position at the beat time (default).
    """
    if not beat_times_ms:
        return {"version": "1.0", "actions": [], "range": 100}
    if intensities is None:
        intensities = {}

    half_stroke_ms = int(50 + (1.0 - speed) * 450)
    pos_range = max_pos - min_pos
    if pos_range <= 0:
        return {"version": "1.0", "actions": [{"at": 0, "pos": min_pos}], "range": 100}

    speed_per_ms = pos_range / half_stroke_ms

    # Build target positions for each beat
    cur_pos = float(min_pos)
    direction = 1   # +1 = up, -1 = down
    targets = []    # list of (beat_ms, target_pos)

    for beat_ms in beat_times_ms:
        inten = intensities.get(beat_ms, 1.0)
        move_amount = pos_range * inten

        if alt_mode == ALT_FORCE:
            # Always move in current direction
            new_pos = cur_pos + direction * move_amount
            new_pos = max(min_pos, min(max_pos, new_pos))
            targets.append((beat_ms, new_pos))
            cur_pos = new_pos
            direction = -direction  # flip for next beat
        else:
            # ALT_KEEP: try current direction; if it would exceed bounds,
            # go in the direction with more room
            candidate = cur_pos + direction * move_amount
            if min_pos <= candidate <= max_pos:
                # Fits — use current direction
                new_pos = candidate
            else:
                # Doesn't fit — check which direction has more space
                room_up = max_pos - cur_pos
                room_down = cur_pos - min_pos
                if room_up >= room_down:
                    new_pos = min(max_pos, cur_pos + move_amount)
                    direction = 1
                else:
                    new_pos = max(min_pos, cur_pos - move_amount)
                    direction = -1
                # Flip for next beat (we just chose a direction)
            targets.append((beat_ms, new_pos))
            cur_pos = new_pos
            direction = -direction

    # Now generate the actual funscript actions with proper speed
    actions = [{"at": 0, "pos": min_pos}]
    prev_pos = float(min_pos)
    prev_t = 0.0

    for beat_ms, target_pos in targets:
        dist = abs(target_pos - prev_pos)
        travel_ms = dist / speed_per_ms if speed_per_ms > 0 else half_stroke_ms

        if start_at_beat:
            # Movement starts at the beat time
            actions.append({"at": beat_ms, "pos": int(round(prev_pos))})
            arrive_t = beat_ms + travel_ms
            actions.append({"at": int(round(arrive_t)), "pos": int(round(target_pos))})
        else:
            # Movement arrives at the beat time (default)
            depart_t = beat_ms - travel_ms
            if depart_t > prev_t + 1:
                # Hold at previous position until departure
                actions.append({"at": int(round(depart_t)), "pos": int(round(prev_pos))})
            elif depart_t < prev_t:
                # Not enough time — just move as fast as possible (straight line)
                pass
            actions.append({"at": beat_ms, "pos": int(round(target_pos))})

        prev_pos = target_pos
        prev_t = beat_ms

    return {"version": "1.0", "actions": _dedup_actions(actions), "range": 100}


def _add_idle_noise(actions, noise_amount, min_pos, max_pos):
    """Add slow random movement in sections without any beat activity.

    Finds gaps where the device would sit still and injects gentle random
    movements.  *noise_amount* (0‒1) controls the amplitude of the noise.
    """
    if noise_amount <= 0 or not actions or len(actions) < 2:
        return actions

    # Noise stroke amplitude: fraction of the position range
    pos_range = max_pos - min_pos
    noise_amp = pos_range * noise_amount * 0.4  # max ±40% of range at noise=1
    # Noise stroke interval: one movement every ~600‒1200 ms
    noise_interval = 800  # ms between noise points
    # Minimum gap to consider "idle" (no existing movement)
    idle_threshold = noise_interval * 1.2

    rng = np.random.RandomState(42)  # deterministic for reproducibility

    result = [actions[0]]
    for i in range(1, len(actions)):
        prev = result[-1]
        curr = actions[i]
        gap = curr["at"] - prev["at"]

        if gap >= idle_threshold and prev["pos"] == curr["pos"]:
            # This is an idle section — inject noise points
            t = prev["at"] + noise_interval
            last_noise_pos = prev["pos"]
            while t < curr["at"] - noise_interval // 2:
                # Random target within noise_amp of the idle position,
                # clamped to min/max
                base = prev["pos"]
                offset = rng.uniform(-noise_amp, noise_amp)
                new_pos = int(round(max(min_pos, min(max_pos, base + offset))))
                # Avoid tiny movements
                if abs(new_pos - last_noise_pos) < 3:
                    new_pos = last_noise_pos + (3 if rng.random() > 0.5 else -3)
                    new_pos = int(round(max(min_pos, min(max_pos, new_pos))))
                result.append({"at": int(t), "pos": new_pos})
                last_noise_pos = new_pos
                t += noise_interval
            # Ease back to the original position before the next beat action
            if result[-1]["pos"] != curr["pos"]:
                ease_t = curr["at"] - min(noise_interval // 2, (curr["at"] - result[-1]["at"]) // 2)
                if ease_t > result[-1]["at"]:
                    result.append({"at": int(ease_t), "pos": curr["pos"]})

        result.append(curr)

    return result


def generate_funscript(beat_times_ms, speed=0.5, min_pos=0, max_pos=100,
                       mode=MODE_PEAK, noise_amount=0, intensities=None,
                       alt_mode=ALT_FORCE, start_at_beat=False,
                       reverse_curve=False):
    if mode == MODE_ALTERNATING:
        fs = generate_funscript_alternating(beat_times_ms, speed, min_pos, max_pos,
                                            intensities=intensities,
                                            alt_mode=alt_mode,
                                            start_at_beat=start_at_beat)
    else:
        fs = generate_funscript_peak(beat_times_ms, speed, min_pos, max_pos,
                                     intensities=intensities,
                                     start_at_beat=start_at_beat)

    if noise_amount > 0:
        fs["actions"] = _add_idle_noise(fs["actions"], noise_amount, min_pos, max_pos)

    if reverse_curve and fs.get("actions"):
        for a in fs["actions"]:
            a["pos"] = min_pos + max_pos - a["pos"]

    return fs


def _format_time(seconds, _pos=None):
    if seconds < 0:
        seconds = 0
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:04.1f}"


def _format_time_ms(seconds):
    if seconds < 0:
        seconds = 0
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:06.3f}"


def _sec_to_tc(seconds):
    """Convert seconds to hh:mm:ss.fff timecode string."""
    if seconds < 0:
        seconds = 0
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _tc_to_sec(tc):
    """Parse hh:mm:ss.fff (or mm:ss.fff or ss.fff or plain float) to seconds."""
    tc = tc.strip()
    parts = tc.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        return None


def _compute_envelope(y, sr):
    """Compute a downsampled waveform envelope for the full audio.

    Returns (t_array, envelope_max) covering the full duration,
    downsampled to at most ENVELOPE_MAX_POINTS.  Time values are placed
    at the **centre** of each chunk so that peaks align with the true
    sample position.
    """
    total_samples = len(y)
    if total_samples == 0:
        return np.array([0]), np.array([0])

    chunk = max(1, total_samples // ENVELOPE_MAX_POINTS)
    n_chunks = total_samples // chunk
    if n_chunks < 2:
        t_env = np.arange(total_samples) / sr
        return t_env, np.abs(y)

    y_trimmed = y[:n_chunks * chunk].reshape(n_chunks, chunk)
    env_max = np.max(np.abs(y_trimmed), axis=1)
    # Centre of each chunk → peaks appear at the right x position
    t_env = (np.arange(n_chunks) + 0.5) * chunk / sr
    return t_env, env_max


def _build_envelope_mipmap(t_env, env_max):
    """Build a multi-resolution pyramid for fast LOD waveform drawing.

    Each level halves the number of points by taking the max of each pair.
    Returns a list of (t, env) arrays from finest to coarsest.
    """
    levels = [(t_env, env_max)]
    t, e = t_env, env_max
    while len(t) > 64:
        n = len(t)
        n_even = n - (n % 2)
        if n_even < 2:
            break
        t2 = t[:n_even].reshape(-1, 2)
        e2 = e[:n_even].reshape(-1, 2)
        t_new = t2.mean(axis=1)
        e_new = e2.max(axis=1)
        # Keep the last odd sample if present
        if n % 2:
            t_new = np.append(t_new, t[-1])
            e_new = np.append(e_new, e[-1])
        t, e = t_new, e_new
        levels.append((t, e))
    return levels


class App:
    def __init__(self, root, initial_file=None):
        self.root = root
        root.title("BeatFunCreator")
        root.geometry("1200x800")
        root.minsize(950, 650)

        # Set window icon
        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.isfile(_icon_path):
            try:
                root.iconbitmap(_icon_path)
            except Exception:
                pass

        # Menu bar
        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open File…", command=lambda: self._browse_file(),
                              accelerator="Ctrl+O")
        file_menu.add_command(label="Load Project…", command=lambda: self._load_beats())
        file_menu.add_command(label="Save Project…", command=lambda: self._save_beats())
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=lambda: self._on_close())
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="GitHub Page",
                              command=lambda: webbrowser.open(GITHUB_URL))
        help_menu.add_command(label="Support / Donate",
                              command=lambda: webbrowser.open(DONATE_URL))
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        root.config(menu=menubar)

        self.y = None       # normalized audio (used for display & detection)
        self.y_raw = None   # original unnormalized audio
        self.sr = None
        self.duration = 0
        self.beats_ms = None
        self.beats_intensity = {}   # {time_ms: 0.2 | 0.6 | 1.0}
        self.tmp_audio = None
        self.in_line_w = None
        self.out_line_w = None
        self.in_handle = None
        self.out_handle = None
        self.in_line_f = None
        self.out_line_f = None
        self.in_line_z = None
        self.out_line_z = None
        self._dragging = None
        self._suppress_marker_draw = False  # suppress draw_idle during drag
        self._last_funscript = None
        self._pending_project = None  # project data to restore after media loads
        self._dim_patches = []  # greyed-out overlays outside marker range

        # Full envelope data (computed once on load)
        self._t_env = None
        self._env_max = None
        self._env_mipmap = None  # multi-resolution pyramid

        # LOD artists (removed/recreated on view change)
        self._wave_artists = []
        self._wave_fill = None       # persistent Polygon for waveform envelope
        self._fun_line = None        # persistent Line2D for funscript curve
        self._fun_fill = None        # persistent Polygon for funscript fill
        self._beat_artists = []
        self._fun_artists = []
        self._beat_ym = 1

        # Beat hover/edit state
        self._hover_beat_idx = None      # index into self.beats_ms
        self._hover_line_w = None        # highlight artist on waveform
        self._hover_line_f = None        # highlight artist on funscript
        self._dragging_beat_idx = None   # index of beat being dragged
        self._dragging_beat_orig_ms = None  # original time of dragged beat

        # Multi-select state
        self._selected_beats = set()     # set of time_ms values
        self._selection_start_xy = None  # (x_data, y_pixel) for selection rect
        self._selection_rect_artist = None  # matplotlib Rectangle artist
        self._selecting = False          # True while drawing selection rect
        self._multi_drag_offsets = None  # {time_ms: offset_ms} for group drag
        self._multi_drag_anchor_ms = None  # anchor beat ms for group drag

        # Snap preview
        self._snap_preview_line = None   # artist for ghost marker
        self._multi_beat_preview_artists = []  # ghost markers for shift+hover

        # Middle-mouse pan state
        self._mmb_pan_start_x = None     # pixel x at press
        self._mmb_pan_start_xlim = None  # xlim at press
        self._mmb_pan_click_data = None  # data x at press (for seek on release)
        self._mmb_did_pan = False        # True if mouse moved enough to count as pan

        # Modifier key tracking (independent of canvas focus)
        self._mod_shift = False
        self._mod_ctrl = False

        # Clipboard for copy/paste beats
        self._clipboard_beats = None     # list of (offset_ms, intensity) relative to first beat

        # Override zones
        self._override_zones = []        # list of {"start_ms", "end_ms", "settings": {...}}
        self._selected_zone_idx = None   # index of selected zone or None
        self._zone_artists = []          # LOD artists for zone rectangles
        self._clipboard_zone = None      # copied zone (settings + duration)

        # Zone drag state
        self._zone_drag_start_ms = None  # zone's original start_ms at drag begin
        self._zone_drag_end_ms = None    # zone's original end_ms at drag begin
        self._zone_drag_click_ms = None  # where the user clicked (in ms)

        # Musical beat grid (tempo detection)
        self._tempo_grid_times = None    # numpy array of beat times in seconds
        self._tempo_sub_times = None     # numpy array of sub-beat times
        self._tempo_bpm = None           # detected BPM
        self._tempo_phase = 0.0          # phase offset in seconds (time of first beat)
        self._tempo_time_sig = 4         # beats per measure (numerator)
        self._tempo_grid_artists = []    # LOD artists for grid lines
        self._tap_times = []             # timestamps for tap tempo
        self._tap_recording = False      # True while recording taps

        # Undo/Redo
        self._undo_stack = []            # list of state snapshots (max 50)
        self._redo_stack = []            # list of state snapshots

        # Keyframe editing state
        self._kf_moved = {}              # {orig_at_ms: {"at": new_at, "pos": new_pos}}
        self._kf_deleted = set()         # set of orig_at_ms that were deleted
        self._kf_independent = []        # [{"at": ms, "pos": pos}] manually added
        self._kf_types = {}              # {current_at_ms: "modified"|"independent"} (derived)
        self._kf_reverse = {}            # {current_at_ms: orig_at_ms} for modified kfs (derived)
        self._selected_kf = set()        # set of at_ms for selected keyframes
        self._hover_kf_idx = None        # index in actions list
        self._hover_kf_artist = None     # highlight dot artist
        self._dragging_kf_idx = None     # index of kf being dragged
        self._dragging_kf_orig = None    # {"at", "pos"} at drag start
        self._kf_multi_drag = None       # {at_ms: {"off_ms": .., "off_pos": ..}}
        self._kf_multi_anchor = None     # at_ms of anchor keyframe for group drag
        self._clipboard_kf = None        # [(offset_ms, pos)] for copy/paste
        self._kf_select_start = None     # (x_data, y_data) for selection rect on ax_fun

        # Marker handle hover state
        self._hover_marker = None        # "in" or "out" or None

        # FPS counter state
        self._fps_last_time = _time.time()
        self._fps_frame_count = 0
        self._fps_value = 0.0
        self._fps_text = None  # matplotlib text artist

        # Playback state
        self._playing = False
        self._play_start_pos = 0
        self._play_end_pos = 0
        self._play_stream = None        # sd.OutputStream for sample-accurate position
        self._play_sample_idx = 0       # current sample index in the chunk
        self._play_chunk = None         # audio chunk being played
        self._playhead_pos = 0  # current playhead in seconds (persists after stop)
        self._playback_speed = 1.0  # playback speed multiplier
        self._video_window = None   # Toplevel for video preview
        self._video_cap = None      # cv2.VideoCapture
        self._video_fps = 30.0      # video frame rate
        self._video_label = None    # tk.Label showing video frame
        self._video_last_frame_sec = -1  # avoid redundant seeks
        self._playhead_line_w = None
        self._playhead_line_f = None
        self._playhead_line_z = None
        self._playhead_timer = None

        # ── Left panel (scrollable) ──
        left_outer = ttk.Frame(root, width=380)
        left_outer.pack(side="left", fill="y", padx=(8, 0), pady=8)
        left_outer.pack_propagate(False)

        left_canvas = tk.Canvas(left_outer, highlightthickness=0, bd=0,
                                width=365)
        left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical",
                                       command=left_canvas.yview)
        left = ttk.Frame(left_canvas)

        left.bind("<Configure>",
                  lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        self._left_canvas_win = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        # Keep inner frame width in sync with canvas width
        def _on_canvas_configure(event):
            left_canvas.itemconfig(self._left_canvas_win, width=event.width)
        left_canvas.bind("<Configure>", _on_canvas_configure)

        # Enable mouse wheel scrolling on the left panel
        def _on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel, add=False)
        # Only scroll when mouse is over the left panel
        def _bind_left_scroll(event):
            left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)
        def _unbind_left_scroll(event):
            left_canvas.unbind_all("<MouseWheel>")
        left_outer.bind("<Enter>", _bind_left_scroll)
        left_outer.bind("<Leave>", _unbind_left_scroll)

        # File & Output selection (combined)
        coll_file = CollapsibleFrame(left, text="File / Output")
        coll_file.pack(fill="x", pady=(0, 4))
        frame_file = coll_file.content

        file_row = ttk.Frame(frame_file)
        file_row.pack(fill="x", padx=8, pady=(6, 2))
        self.file_path = tk.StringVar()
        ttk.Label(file_row, text="Input").pack(side="left", padx=(0, 4))
        ttk.Entry(file_row, textvariable=self.file_path).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(file_row, text="Browse…", command=self._browse_file).pack(side="left")

        out_row = ttk.Frame(frame_file)
        out_row.pack(fill="x", padx=8, pady=(2, 6))
        self.out_dir = tk.StringVar()
        ttk.Label(out_row, text="Output").pack(side="left", padx=(0, 4))
        ttk.Entry(out_row, textvariable=self.out_dir).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(out_row, text="Browse…", command=self._browse_outdir).pack(side="left")

        # ── Work Area (In / Out markers) ──
        coll_workarea = CollapsibleFrame(left, text="Work Area")
        coll_workarea.pack(fill="x", pady=4)
        frame_workarea = coll_workarea.content

        marker_row = ttk.Frame(frame_workarea)
        marker_row.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(marker_row, text="In:").pack(side="left", padx=(0, 2))
        self.in_var = tk.StringVar(value=_sec_to_tc(0))
        ttk.Entry(marker_row, textvariable=self.in_var, width=12).pack(
            side="left", fill="x", expand=True, padx=2)
        ttk.Label(marker_row, text="Out:").pack(side="left", padx=(6, 2))
        self.out_var = tk.StringVar(value="—")
        ttk.Entry(marker_row, textvariable=self.out_var, width=12).pack(
            side="left", fill="x", expand=True, padx=2)
        ttk.Button(marker_row, text="Reset", command=self._reset_markers).pack(
            side="left", padx=(4, 0))

        tk.Label(frame_workarea,
                 text="Operations only affect the area between In and Out markers.\n"
                      "Elements outside this range appear dimmed.",
                 fg="gray", font=("Segoe UI", 7), justify="left").pack(
            fill="x", padx=8, pady=(2, 6))

        # ── Normalization section ──
        coll_norm = CollapsibleFrame(left, text="Normalization")
        coll_norm.pack(fill="x", pady=4)
        frame_norm = coll_norm.content

        self.auto_normalize = tk.BooleanVar(value=DEFAULTS["auto_normalize"])
        self.norm_percentile = tk.DoubleVar(value=DEFAULTS["norm_percentile"])

        norm_row0 = ttk.Frame(frame_norm)
        norm_row0.grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(norm_row0, text="Auto-normalize on load",
                        variable=self.auto_normalize).pack(side="left")
        ttk.Button(norm_row0, text="↺", width=2,
                   command=lambda: self.auto_normalize.set(
                       DEFAULTS["auto_normalize"])).pack(side="left", padx=4)

        frame_norm.columnconfigure(1, weight=1)
        ttk.Label(frame_norm, text="Percentile").grid(
            row=1, column=0, sticky="w", padx=8, pady=3)
        ttk.Scale(frame_norm, from_=50.0, to=100.0, variable=self.norm_percentile).grid(
            row=1, column=1, sticky="ew", padx=4, pady=3)
        self._norm_pct_label = ttk.Label(frame_norm, text="100%", width=5)
        self._norm_pct_label.grid(row=1, column=2, padx=(0, 2), pady=3)
        ttk.Button(frame_norm, text="↺", width=2,
                   command=lambda: self.norm_percentile.set(
                       DEFAULTS["norm_percentile"])).grid(
            row=1, column=3, padx=(0, 4), pady=3)
        self.norm_percentile.trace_add("write", lambda *_: self._norm_pct_label.config(
            text=f"{self.norm_percentile.get():.0f}%"))

        ttk.Button(frame_norm, text="Normalize Now",
                   command=self._normalize_now).grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=(2, 6))

        tk.Label(frame_norm,
                 text="Lower percentile = louder overall, peaks clip.\n"
                      "100% = classic peak normalization (no clipping).",
                 fg="gray", font=("Segoe UI", 7), justify="left").grid(
            row=3, column=0, columnspan=4, padx=8, pady=(0, 4))

        # All setting vars
        self.threshold = tk.DoubleVar(value=DEFAULTS["threshold"])
        self.speed = tk.DoubleVar(value=DEFAULTS["speed"])
        self.pos_min = tk.DoubleVar(value=DEFAULTS["pos_min"])
        self.pos_max = tk.DoubleVar(value=DEFAULTS["pos_max"])
        self.min_beat_gap = tk.DoubleVar(value=DEFAULTS["min_beat_gap"])
        self.noise_amount = tk.DoubleVar(value=DEFAULTS["noise_amount"])
        self.bass_only = tk.BooleanVar(value=DEFAULTS["bass_only"])
        self.mode_var = tk.StringVar(value=DEFAULTS["mode"])
        self.peak_start_at_beat = tk.BooleanVar(value=DEFAULTS["peak_start_at_beat"])
        self.reverse_curve = tk.BooleanVar(value=DEFAULTS["reverse_curve"])

        self._slider_vars = {
            "threshold": self.threshold,
            "min_beat_gap": self.min_beat_gap,
            "speed": self.speed,
            "pos_min": self.pos_min,
            "pos_max": self.pos_max,
            "noise_amount": self.noise_amount,
        }

        # Helper to add a slider row with reset button
        def _add_slider(parent, row, label, key, var):
            parent.columnconfigure(1, weight=1)
            ttk.Label(parent, text=label).grid(
                row=row, column=0, sticky="w", padx=8, pady=3)
            ttk.Scale(parent, from_=0.0, to=1.0, variable=var).grid(
                row=row, column=1, sticky="ew", padx=4, pady=3)
            val_label = ttk.Label(parent, text="0.00", width=5)
            val_label.grid(row=row, column=2, padx=(0, 2), pady=3)
            ttk.Button(parent, text="↺", width=2,
                       command=lambda v=var, d=DEFAULTS[key]: v.set(d)).grid(
                row=row, column=3, padx=(0, 4), pady=3)
            if var is self.min_beat_gap:
                var.trace_add("write",
                              lambda *_, v=var, l=val_label: l.config(
                                  text=f"{int(v.get() * 2000)}ms"))
                val_label.config(text=f"{int(var.get() * 2000)}ms")
            else:
                var.trace_add("write",
                              lambda *_, v=var, l=val_label: l.config(text=f"{v.get():.2f}"))
                val_label.config(text=f"{var.get():.2f}")

        # ── Beat Detection section ──
        coll_beat = CollapsibleFrame(left, text="Beat Detection")
        coll_beat.pack(fill="x", pady=4)
        frame_beat = coll_beat.content

        _add_slider(frame_beat, 0, "Beat Threshold", "threshold", self.threshold)
        _add_slider(frame_beat, 1, "Min Beat Gap", "min_beat_gap", self.min_beat_gap)

        # Bass-only checkbox
        bass_frame = ttk.Frame(frame_beat)
        bass_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(bass_frame, text="Bass only (low freq ≤ 200 Hz)",
                        variable=self.bass_only).pack(side="left")
        ttk.Button(frame_beat, text="↺", width=2,
                   command=lambda: self.bass_only.set(DEFAULTS["bass_only"])).grid(
            row=2, column=3, padx=(0, 4), pady=3)

        # Analyze Beats button inside Beat Detection
        self.btn_analyze = ttk.Button(frame_beat, text="Analyze Beats",
                                      command=lambda: self._run("analyze"))
        self.btn_analyze.grid(row=3, column=0, columnspan=4, sticky="ew", padx=8, pady=(4, 3))

        # ── Tempo / Beat Grid section ──
        coll_tempo = CollapsibleFrame(left, text="Tempo / Beat Grid")
        coll_tempo.pack(fill="x", pady=4)
        frame_tempo = coll_tempo.content

        # Make tempo grid columns expand to full width
        frame_tempo.columnconfigure(0, weight=1)

        # Row 0: Detect / Generate / Clear — three buttons side by side
        tempo_row0 = ttk.Frame(frame_tempo)
        tempo_row0.grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=2)
        ttk.Button(tempo_row0, text="Detect",
                   command=self._detect_tempo).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(tempo_row0, text="Generate",
                   command=self._generate_tempo_grid).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(tempo_row0, text="Clear",
                   command=self._clear_tempo).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # Row 1: BPM + Time Sig + Phase on one row
        tempo_row1 = ttk.Frame(frame_tempo)
        tempo_row1.grid(row=1, column=0, columnspan=4, sticky="ew", padx=8, pady=2)
        self.tempo_bpm_var = tk.StringVar(value="")
        ttk.Label(tempo_row1, text="BPM").pack(side="left", padx=(0, 2))
        ttk.Entry(tempo_row1, textvariable=self.tempo_bpm_var, width=6).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        self.tempo_timesig_var = tk.StringVar(value="4/4")
        ttk.Label(tempo_row1, text="Sig.").pack(side="left", padx=(0, 2))
        ttk.Combobox(tempo_row1, textvariable=self.tempo_timesig_var,
                     values=["2/4", "3/4", "4/4", "5/4", "6/8", "7/8"],
                     width=4, state="readonly").pack(side="left", padx=(0, 4))
        self.tempo_phase_var = tk.StringVar(value="0.000")
        ttk.Label(tempo_row1, text="Phase").pack(side="left", padx=(0, 2))
        ttk.Entry(tempo_row1, textvariable=self.tempo_phase_var, width=6).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        self.tempo_subdiv_var = tk.IntVar(value=1)
        ttk.Label(tempo_row1, text="Sub").pack(side="left", padx=(0, 2))
        self._subdiv_spin = ttk.Spinbox(tempo_row1, textvariable=self.tempo_subdiv_var,
                                         from_=1, to=16, width=3)
        self._subdiv_spin.pack(side="left")
        ToolTip(self._subdiv_spin,
                "Sub-beat divisions per beat.\n"
                "1 = no sub-beats, 2 = double,\n"
                "3 = triplets, 4 = 16th notes, etc.\n"
                "Adds extra snap points between beats.")

        # Row 2: Tap Tempo — record toggle + tap button + status
        tempo_row3 = ttk.Frame(frame_tempo)
        tempo_row3.grid(row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=2)
        self._tap_btn = ttk.Button(tempo_row3, text="Record Taps",
                                   command=self._toggle_tap_recording)
        self._tap_btn.pack(side="left", padx=(0, 4))
        self._tap_click_btn = ttk.Button(tempo_row3, text="Tap",
                                         command=self._register_tap,
                                         state="disabled")
        self._tap_click_btn.pack(side="left", padx=(0, 4))
        self._tap_rec_indicator = tk.Label(tempo_row3, text="● REC",
                                           fg="#f7768e", bg="#1a1b26",
                                           font=("Segoe UI", 8, "bold"))
        self._tap_rec_indicator.pack(side="left", padx=(0, 4))
        self._tap_rec_indicator.pack_forget()  # hidden initially
        self._tap_label = ttk.Label(tempo_row3, text="", font=("Segoe UI", 7))
        self._tap_label.pack(side="left", fill="x", expand=True)

        # ── Movement Settings section ──
        coll_move = CollapsibleFrame(left, text="Movement Settings")
        coll_move.pack(fill="x", pady=4)
        frame_move = coll_move.content
        frame_move.columnconfigure(1, weight=1)

        # Mode selector (dropdown)
        ttk.Label(frame_move, text="Mode").grid(
            row=0, column=0, sticky="w", padx=8, pady=3)
        mode_combo_frame = ttk.Frame(frame_move)
        mode_combo_frame.grid(row=0, column=1, columnspan=2, sticky="w", padx=4, pady=3)
        ttk.Combobox(mode_combo_frame, textvariable=self.mode_var,
                     values=[MODE_PEAK, MODE_ALTERNATING], width=16,
                     state="readonly").pack(side="left")
        ttk.Button(frame_move, text="↺", width=2,
                   command=lambda: self.mode_var.set(DEFAULTS["mode"])).grid(
            row=0, column=3, padx=(0, 4), pady=3)

        # ── Start at beat (common, row 1) ──
        start_at_beat_frame = ttk.Frame(frame_move)
        start_at_beat_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(start_at_beat_frame, text="Start movement at beat",
                        variable=self.peak_start_at_beat).pack(side="left")
        ttk.Button(frame_move, text="↺", width=2,
                   command=lambda: self.peak_start_at_beat.set(DEFAULTS["peak_start_at_beat"])).grid(
            row=1, column=3, padx=(0, 4), pady=3)

        # ── Alternating-specific settings (row 2) ──
        self.alt_mode_var = tk.StringVar(value=DEFAULTS["alt_mode"])
        self._alt_options_frame = ttk.Frame(frame_move)
        self._alt_options_frame.grid(row=2, column=0, columnspan=4, sticky="ew")
        alt_inner = ttk.Frame(self._alt_options_frame)
        alt_inner.pack(fill="x")
        ttk.Label(alt_inner, text="Alt. mode").grid(
            row=0, column=0, sticky="w", padx=8, pady=3)
        alt_combo_frame = ttk.Frame(alt_inner)
        alt_combo_frame.grid(row=0, column=1, columnspan=2, sticky="w", padx=4, pady=3)
        ttk.Combobox(alt_combo_frame, textvariable=self.alt_mode_var,
                     values=[ALT_FORCE, ALT_KEEP], width=24,
                     state="readonly").pack(side="left")
        ttk.Button(alt_inner, text="↺", width=2,
                   command=lambda: self.alt_mode_var.set(DEFAULTS["alt_mode"])).grid(
            row=0, column=3, padx=(0, 4), pady=3)

        _add_slider(frame_move, 3, "Movement Speed", "speed", self.speed)
        _add_slider(frame_move, 4, "Position Min", "pos_min", self.pos_min)
        _add_slider(frame_move, 5, "Position Max", "pos_max", self.pos_max)
        _add_slider(frame_move, 6, "Idle Noise", "noise_amount", self.noise_amount)

        # Reverse curve checkbox
        reverse_frame = ttk.Frame(frame_move)
        reverse_frame.grid(row=7, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(reverse_frame, text="Reverse movement curve",
                        variable=self.reverse_curve).pack(side="left")
        ttk.Button(frame_move, text="↺", width=2,
                   command=lambda: self.reverse_curve.set(DEFAULTS["reverse_curve"])).grid(
            row=7, column=3, padx=(0, 4), pady=3)

        # Show/hide mode-specific options based on current mode
        def _update_mode_visibility(*_):
            mode = self.mode_var.get()
            if mode == MODE_ALTERNATING:
                self._alt_options_frame.grid()
            else:
                self._alt_options_frame.grid_remove()
        self._update_mode_visibility = _update_mode_visibility
        self.mode_var.trace_add("write", _update_mode_visibility)
        _update_mode_visibility()  # initial state

        # Buttons — use consistent inner padding via a single padded frame
        frame_btns = ttk.Frame(left)
        frame_btns.pack(fill="x", padx=8, pady=8)

        frame_beats_io = ttk.Frame(frame_btns)
        frame_beats_io.pack(fill="x", pady=2)
        ttk.Button(frame_beats_io, text="Save Project…",
                   command=self._save_beats).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(frame_beats_io, text="Load Project…",
                   command=self._load_beats).pack(side="left", fill="x", expand=True, padx=(2, 0))

        frame_reset_row = ttk.Frame(frame_btns)
        frame_reset_row.pack(fill="x", pady=2)
        self.btn_reset_beats = ttk.Button(frame_reset_row, text="Reset Beats",
                                          command=self._reset_beats_only)
        self.btn_reset_beats.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.btn_reset_movement = ttk.Button(frame_reset_row, text="Reset Movement",
                                             command=self._reset_movement)
        self.btn_reset_movement.pack(side="left", fill="x", expand=True, padx=2)
        self.btn_reset_zones = ttk.Button(frame_reset_row, text="Reset Zones",
                                          command=self._reset_zones)
        self.btn_reset_zones.pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.btn_create = ttk.Button(frame_btns, text="Create FunScript",
                                     command=lambda: self._run("create"))
        self.btn_create.pack(fill="x", pady=2)

        # Progress
        self.progress = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(4, 2))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status, wraplength=350,
                  font=("Segoe UI", 8)).pack(padx=8, pady=(0, 4))

        # ── Right panel ──
        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True, padx=(4, 8), pady=8)

        # Beat editor controls at top of timeline panel
        frame_top_bar = ttk.Frame(right)
        frame_top_bar.pack(fill="x")

        # Row 1: editor controls, snap, intensity
        frame_edit_row1 = ttk.Frame(frame_top_bar)
        frame_edit_row1.pack(fill="x")
        self.beat_edit_mode = tk.BooleanVar(value=True)  # always active
        KF_LOCK_OPTS = ["No restrictions", "Lock all", "Lock beat", "Lock independent"]
        KF_LOCK_TIME_OPTS = KF_LOCK_OPTS + ["Lock order"]
        self.kf_lock_time = tk.StringVar(value="No restrictions")
        ttk.Label(frame_edit_row1, text="Lock time:").pack(side="left", padx=(4, 0))
        ttk.Combobox(frame_edit_row1, textvariable=self.kf_lock_time,
                     values=KF_LOCK_TIME_OPTS, width=14,
                     state="readonly").pack(side="left", padx=(2, 4))
        self.kf_lock_pos = tk.StringVar(value="No restrictions")
        ttk.Label(frame_edit_row1, text="Lock pos:").pack(side="left", padx=(4, 0))
        ttk.Combobox(frame_edit_row1, textvariable=self.kf_lock_pos,
                     values=KF_LOCK_OPTS, width=14,
                     state="readonly").pack(side="left", padx=(2, 4))
        self.beat_intensity_var = tk.IntVar(value=100)
        self._intensity_presets = [20, 40, 60, 80, 100]  # default presets
        ttk.Label(frame_edit_row1, text="Intensity:").pack(side="left", padx=(8, 0))
        self._intensity_label = ttk.Label(frame_edit_row1, text="100%", width=5)
        self._intensity_label.pack(side="left", padx=(0, 0))
        ttk.Scale(frame_edit_row1, from_=0, to=100, variable=self.beat_intensity_var,
                  length=80, command=lambda v: self._on_intensity_slider(float(v))
                  ).pack(side="left", padx=(2, 2))
        self._preset_btns = []
        for i in range(5):
            idx = i
            val = self._intensity_presets[i]
            btn = ttk.Button(frame_edit_row1, text=f"{val}%", width=5,
                             command=lambda ii=idx: self._recall_intensity_preset(ii))
            btn.pack(side="left", padx=0)
            btn.bind("<Button-3>", lambda e, ii=idx: self._on_preset_click(ii))
            ToolTip(btn, f"Click: recall preset #{i+1}\n"
                         f"Right-click: save current intensity\n"
                         f"Key {i+1}: recall preset value")
            self._preset_btns.append(btn)
        # Snap mode: Off / Peaks / Musical Beat
        self.snap_to_spike = tk.BooleanVar(value=False)  # kept for compat
        self.snap_mode = tk.StringVar(value="Off")
        self.snap_first_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_edit_row1, text="1st only",
                        variable=self.snap_first_only).pack(side="right", padx=(0, 2))
        ttk.Combobox(frame_edit_row1, textvariable=self.snap_mode,
                     values=["Off", "Peaks", "Musical Beat"], width=12,
                     state="readonly").pack(side="right", padx=(2, 4))
        ttk.Label(frame_edit_row1, text="Snap:").pack(side="right", padx=(4, 0))
        self.snap_strength = tk.DoubleVar(value=0.5)
        ttk.Scale(frame_edit_row1, from_=0.05, to=1.0, variable=self.snap_strength,
                  length=60).pack(side="right", padx=2)
        ttk.Label(frame_edit_row1, text="Snap radius:").pack(side="right")
        # Sync snap_to_spike bool from snap_mode for backward compat
        def _sync_snap_mode(*_):
            self.snap_to_spike.set(self.snap_mode.get() != "Off")
        self.snap_mode.trace_add("write", _sync_snap_mode)

        # Row 2: shift+click multi-beat params
        frame_edit_row2 = ttk.Frame(frame_top_bar)
        frame_edit_row2.pack(fill="x")
        ttk.Label(frame_edit_row2,
                  text="Shift+click: add").pack(side="left", padx=(4, 2))
        self.multi_beat_count = tk.IntVar(value=4)
        ttk.Spinbox(frame_edit_row2, textvariable=self.multi_beat_count,
                    from_=1, to=100, width=4).pack(side="left", padx=2)
        ttk.Label(frame_edit_row2, text="beats, spacing").pack(side="left", padx=2)
        self.multi_beat_spacing_mode = tk.StringVar(value="Time")
        self._spacing_mode_combo = ttk.Combobox(
            frame_edit_row2, textvariable=self.multi_beat_spacing_mode,
            values=["Time"], width=6, state="readonly")
        self._spacing_mode_combo.pack(side="left", padx=2)
        self.multi_beat_spacing = tk.DoubleVar(value=0.5)
        self._multi_beat_spacing_spin = ttk.Spinbox(
            frame_edit_row2, textvariable=self.multi_beat_spacing,
            from_=0.01, to=10.0, increment=0.01, width=5, format="%.2f")
        self._multi_beat_spacing_spin.pack(side="left", padx=2)
        self._multi_beat_spacing_label = ttk.Label(frame_edit_row2, text="s")
        self._multi_beat_spacing_label.pack(side="left")
        def _on_spacing_mode(*_):
            if self.multi_beat_spacing_mode.get() == "Tempo":
                self._multi_beat_spacing_spin.config(state="disabled")
                self._multi_beat_spacing_label.config(text="")
            else:
                self._multi_beat_spacing_spin.config(state="normal")
                self._multi_beat_spacing_label.config(text="s")
        self.multi_beat_spacing_mode.trace_add("write", _on_spacing_mode)

        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG_COLOR)
        self.fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.08, hspace=0.08)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 2.3, 0.2], hspace=0.08)
        self.ax_wave = self.fig.add_subplot(gs[0])
        self.ax_fun = self.fig.add_subplot(gs[1], sharex=self.ax_wave)
        self.ax_zones = self.fig.add_subplot(gs[2], sharex=self.ax_wave)

        self._style_axes()
        self._set_placeholder_text()

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # FPS counter at top-right corner
        self._fps_text = self.fig.text(
            0.99, 0.97, "0 FPS", fontsize=7, color=TEXT_COLOR,
            alpha=0.5, ha="right", va="top", fontfamily="monospace")

        # Draw callback for FPS counter
        self.canvas.mpl_connect("draw_event", self._on_draw_fps)

        # Horizontal scrollbar for timeline
        self._scrollbar = ttk.Scrollbar(right, orient="horizontal",
                                        command=self._on_scrollbar)
        self._scrollbar.pack(fill="x")
        self._updating_scrollbar = False

        # Timer display — drawn on the canvas (bottom-left of zones axis)
        self.time_var = tk.StringVar(value="0:00:00.000 / 0:00:00.000")
        self._time_text_artist = None  # matplotlib text artist for timer
        self.time_var.trace_add("write", lambda *_: self._update_time_text())

        # Controls row below timeline: video button, speed control, shortcuts hint
        frame_time = ttk.Frame(right)
        frame_time.pack(fill="x")
        tk.Label(frame_time,
                 text="Space = play/pause | Click = seek | ←→ = ±0.1s | Ctrl+←→ = ±1s | Home/End",
                 fg="gray", font=("Segoe UI", 7),
                 anchor="e").pack(side="right", padx=4)
        # Playback speed slider (0.1x – 3.0x)
        self._speed_dvar = tk.DoubleVar(value=1.0)
        self._speed_label_var = tk.StringVar(value="1.0x")
        ttk.Label(frame_time, text="Speed").pack(side="left", padx=(4, 2))
        self._speed_scale = ttk.Scale(
            frame_time, from_=0.1, to=3.0, variable=self._speed_dvar,
            orient="horizontal", length=90,
            command=lambda v: self._on_speed_slider(float(v)))
        self._speed_scale.pack(side="left", padx=(0, 2))
        ttk.Label(frame_time, textvariable=self._speed_label_var, width=4).pack(
            side="left", padx=(0, 6))
        # Video preview button
        self._video_btn = ttk.Button(frame_time, text="Video", command=self._toggle_video_window)
        self._video_btn.pack(side="left", padx=(0, 4))

        # Make canvas focusable
        self.canvas.get_tk_widget().config(takefocus=True)
        self.canvas.get_tk_widget().bind("<Left>", self._on_key_left)
        self.canvas.get_tk_widget().bind("<Right>", self._on_key_right)
        self.canvas.get_tk_widget().bind("<Control-Left>", self._on_key_ctrl_left)
        self.canvas.get_tk_widget().bind("<Control-Right>", self._on_key_ctrl_right)
        self.canvas.get_tk_widget().bind("<Home>", self._on_key_home)
        self.canvas.get_tk_widget().bind("<End>", self._on_key_end)
        self.canvas.get_tk_widget().bind("<Button-1>",
                                         lambda e: self.canvas.get_tk_widget().focus_set(), add="+")

        root.bind("<space>", self._on_space)
        for i in range(5):
            self.canvas.get_tk_widget().bind(
                f"<Key-{i+1}>",
                lambda e, idx=i: self._recall_intensity_preset(idx))
        self.canvas.get_tk_widget().bind("<Delete>", lambda e: self._on_delete_key())
        self.canvas.get_tk_widget().bind("<Control-c>", lambda e: self._copy_all())
        self.canvas.get_tk_widget().bind("<Control-v>", lambda e: self._paste_all())
        self.canvas.get_tk_widget().bind("<Control-z>", lambda e: self._undo())
        self.canvas.get_tk_widget().bind("<Control-y>", lambda e: self._redo())

        # Track modifier keys at root level so they work even before canvas gets focus
        root.bind("<KeyPress-Shift_L>", lambda e: self._set_mod("shift", True))
        root.bind("<KeyPress-Shift_R>", lambda e: self._set_mod("shift", True))
        root.bind("<KeyRelease-Shift_L>", lambda e: self._set_mod("shift", False))
        root.bind("<KeyRelease-Shift_R>", lambda e: self._set_mod("shift", False))
        root.bind("<KeyPress-Control_L>", lambda e: self._set_mod("ctrl", True))
        root.bind("<KeyPress-Control_R>", lambda e: self._set_mod("ctrl", True))
        root.bind("<KeyRelease-Control_L>", lambda e: self._set_mod("ctrl", False))
        root.bind("<KeyRelease-Control_R>", lambda e: self._set_mod("ctrl", False))
        # Also reset modifiers when window loses focus (prevents stuck keys)
        root.bind("<FocusOut>", lambda e: self._reset_mods())

        # Matplotlib events
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self.in_var.trace_add("write", lambda *_: self._update_marker_lines())
        self.out_var.trace_add("write", lambda *_: self._update_marker_lines())
        self.mode_var.trace_add("write", lambda *_: self._on_mode_changed())
        self.alt_mode_var.trace_add("write", lambda *_: self._on_mode_changed())
        self.peak_start_at_beat.trace_add("write", lambda *_: self._on_mode_changed())
        self.reverse_curve.trace_add("write", lambda *_: self._on_mode_changed())
        self.speed.trace_add("write", lambda *_: self._on_mode_changed())
        self.pos_min.trace_add("write", lambda *_: self._on_mode_changed())
        self.pos_max.trace_add("write", lambda *_: self._on_mode_changed())
        self.noise_amount.trace_add("write", lambda *_: self._on_mode_changed())

        # Redraw keyframes when lock options change (color update)
        def _on_lock_changed(*_):
            if self._last_funscript:
                self._draw_fun_lod()
                self.canvas.draw_idle()
        self.kf_lock_time.trace_add("write", _on_lock_changed)
        self.kf_lock_pos.trace_add("write", _on_lock_changed)

        self._setup_dnd()

        # Restore saved settings from last session
        cfg = _load_config()
        self._initial_project = None
        if not initial_file and cfg.get("last_project") and os.path.isfile(cfg["last_project"]):
            self._initial_project = cfg["last_project"]
        elif not initial_file and cfg.get("last_file") and os.path.isfile(cfg["last_file"]):
            initial_file = cfg["last_file"]
        if cfg.get("last_outdir") and os.path.isdir(cfg["last_outdir"]):
            self.out_dir.set(cfg["last_outdir"])
        # Restore slider/checkbox/mode settings
        for key, var in self._slider_vars.items():
            if key in cfg:
                try:
                    var.set(float(cfg[key]))
                except (ValueError, TypeError):
                    pass
        if "bass_only" in cfg:
            self.bass_only.set(bool(cfg["bass_only"]))
        if "mode" in cfg:
            self.mode_var.set(cfg["mode"])
        if "alt_mode" in cfg:
            self.alt_mode_var.set(cfg["alt_mode"])
        if "peak_start_at_beat" in cfg:
            self.peak_start_at_beat.set(bool(cfg["peak_start_at_beat"]))
        if "reverse_curve" in cfg:
            self.reverse_curve.set(bool(cfg["reverse_curve"]))
        if "auto_normalize" in cfg:
            self.auto_normalize.set(bool(cfg["auto_normalize"]))
        if "norm_percentile" in cfg:
            try:
                self.norm_percentile.set(float(cfg["norm_percentile"]))
            except (ValueError, TypeError):
                pass
        # Restore beat editor settings
        if "kf_lock_time" in cfg:
            self.kf_lock_time.set(str(cfg["kf_lock_time"]))
        if "kf_lock_pos" in cfg:
            self.kf_lock_pos.set(str(cfg["kf_lock_pos"]))
        if "snap_mode" in cfg:
            self.snap_mode.set(cfg["snap_mode"])
        elif "snap_to_spike" in cfg and bool(cfg["snap_to_spike"]):
            self.snap_mode.set("Peaks")
        if "snap_first_only" in cfg:
            self.snap_first_only.set(bool(cfg["snap_first_only"]))
        if "snap_strength" in cfg:
            try:
                self.snap_strength.set(float(cfg["snap_strength"]))
            except (ValueError, TypeError):
                pass
        if "beat_intensity" in cfg:
            try:
                v = cfg["beat_intensity"]
                # Handle old format "20%"/"60%"/"100%" and new int format
                if isinstance(v, str):
                    v = {"20%": 20, "60%": 60, "100%": 100}.get(v, 100)
                self.beat_intensity_var.set(int(v))
                self._intensity_label.config(text=f"{int(v)}%")
            except (ValueError, TypeError):
                pass
        if "intensity_presets" in cfg:
            try:
                self._intensity_presets = [int(x) for x in cfg["intensity_presets"]]
                for i, v in enumerate(self._intensity_presets[:5]):
                    if i < len(self._preset_btns):
                        self._preset_btns[i].config(text=f"{v}%")
            except (ValueError, TypeError):
                pass
        if "multi_beat_count" in cfg:
            try:
                self.multi_beat_count.set(int(cfg["multi_beat_count"]))
            except (ValueError, TypeError):
                pass
        if "multi_beat_spacing" in cfg:
            try:
                self.multi_beat_spacing.set(float(cfg["multi_beat_spacing"]))
            except (ValueError, TypeError):
                pass
        if "multi_beat_spacing_mode" in cfg:
            self.multi_beat_spacing_mode.set(cfg["multi_beat_spacing_mode"])

        if "tempo_timesig" in cfg:
            self.tempo_timesig_var.set(cfg["tempo_timesig"])
        if "tempo_subdiv" in cfg:
            try:
                self.tempo_subdiv_var.set(int(cfg["tempo_subdiv"]))
            except (ValueError, TypeError):
                pass

        if self._initial_project:
            # Load last project (which will also load the media file if saved)
            self.root.after(100, lambda: self._load_project_file(self._initial_project))
        elif initial_file and os.path.isfile(initial_file):
            self._set_file(initial_file)

        # Save config on close
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Save current paths and all settings, then exit."""
        cfg = _load_config()
        fp = self.file_path.get().strip()
        od = self.out_dir.get().strip()
        if fp and os.path.isfile(fp):
            cfg["last_file"] = fp
        if od and os.path.isdir(od):
            cfg["last_outdir"] = od
        # Save all slider/checkbox/mode settings
        for key, var in self._slider_vars.items():
            cfg[key] = var.get()
        cfg["bass_only"] = self.bass_only.get()
        cfg["mode"] = self.mode_var.get()
        cfg["alt_mode"] = self.alt_mode_var.get()
        cfg["peak_start_at_beat"] = self.peak_start_at_beat.get()
        cfg["reverse_curve"] = self.reverse_curve.get()
        cfg["auto_normalize"] = self.auto_normalize.get()
        cfg["norm_percentile"] = self.norm_percentile.get()
        # Beat editor settings
        cfg["kf_lock_time"] = self.kf_lock_time.get()
        cfg["kf_lock_pos"] = self.kf_lock_pos.get()
        cfg["snap_mode"] = self.snap_mode.get()
        cfg["snap_first_only"] = self.snap_first_only.get()
        cfg["snap_strength"] = self.snap_strength.get()
        cfg["beat_intensity"] = self.beat_intensity_var.get()
        cfg["intensity_presets"] = list(self._intensity_presets)
        cfg["multi_beat_count"] = self.multi_beat_count.get()
        cfg["multi_beat_spacing"] = self.multi_beat_spacing.get()
        cfg["multi_beat_spacing_mode"] = self.multi_beat_spacing_mode.get()
        cfg["tempo_timesig"] = self.tempo_timesig_var.get()
        cfg["tempo_subdiv"] = self.tempo_subdiv_var.get()
        _save_config(cfg)
        self._stop_playback()
        self.root.destroy()

    def _show_about(self):
        """Show About dialog with version info and donate link."""
        about = tk.Toplevel(self.root)
        about.title("About BeatFunCreator")
        about.resizable(False, False)
        about.transient(self.root)
        about.grab_set()

        frame = ttk.Frame(about, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="BeatFunCreator", font=("Segoe UI", 16, "bold")).pack(pady=(0, 4))
        ttk.Label(frame, text=f"Version {APP_VERSION}").pack()
        ttk.Label(frame, text="Create FunScript files from audio and video beats.",
                  wraplength=300, justify="center").pack(pady=(8, 12))

        link_frame = ttk.Frame(frame)
        link_frame.pack(pady=(0, 8))
        gh_link = tk.Label(link_frame, text="GitHub", fg="#7aa2f7", cursor="hand2",
                           font=("Segoe UI", 9, "underline"))
        gh_link.pack(side="left", padx=8)
        gh_link.bind("<Button-1>", lambda e: webbrowser.open(GITHUB_URL))
        donate_link = tk.Label(link_frame, text="Support / Donate", fg="#9ece6a", cursor="hand2",
                               font=("Segoe UI", 9, "underline"))
        donate_link.pack(side="left", padx=8)
        donate_link.bind("<Button-1>", lambda e: webbrowser.open(DONATE_URL))

        ttk.Button(frame, text="Close", command=about.destroy).pack(pady=(8, 0))

        # Center on parent
        about.update_idletasks()
        w = about.winfo_width()
        h = about.winfo_height()
        x = self.root.winfo_x() + (self.root.winfo_width() - w) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - h) // 2
        about.geometry(f"+{x}+{y}")

    # ── Axes styling ──

    def _style_axes(self):
        for ax in (self.ax_wave, self.ax_fun, self.ax_zones):
            ax.set_facecolor(BG_COLOR)
            ax.tick_params(colors=TEXT_COLOR, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
            ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.5)

        self.ax_wave.yaxis.set_visible(False)
        self.ax_wave.tick_params(axis="x", labelbottom=False)

        self.ax_fun.set_ylabel("Pos", color=TEXT_COLOR, fontsize=9)
        self.ax_fun.set_ylim(-5, 105)
        self.ax_fun.set_yticks([0, 25, 50, 75, 100])
        self.ax_fun.yaxis.set_tick_params(labelsize=7)
        self.ax_fun.xaxis.set_major_formatter(mticker.FuncFormatter(_format_time))
        self.ax_fun.tick_params(axis="x", labelbottom=False)

        self.ax_zones.set_ylim(0, 1)
        self.ax_zones.yaxis.set_visible(False)
        self.ax_zones.xaxis.set_major_formatter(mticker.FuncFormatter(_format_time))
        self.ax_zones.set_xlabel("Time", color=TEXT_COLOR, fontsize=7)
        self.ax_zones.set_ylabel("", color=TEXT_COLOR, fontsize=5,
                                  labelpad=2)

    def _update_time_text(self):
        """Draw or update the timer text below the zones axis, in the figure margin."""
        txt = self.time_var.get()
        if self._time_text_artist is not None:
            try:
                self._time_text_artist.set_text(txt)
            except Exception:
                self._time_text_artist = None
        if self._time_text_artist is None:
            self._time_text_artist = self.fig.text(
                0.035, 0.015, txt,
                ha="left", va="bottom",
                color="#7aa2f7", fontsize=8,
                fontfamily="monospace", alpha=0.9)

    def _set_placeholder_text(self):
        self.ax_wave.text(0.5, 0.5, "Load a file to display waveform",
                          transform=self.ax_wave.transAxes, ha="center", va="center",
                          color=TEXT_COLOR, fontsize=11, alpha=0.5)
        self.ax_fun.text(0.5, 0.5, "FunScript preview will appear here",
                         transform=self.ax_fun.transAxes, ha="center", va="center",
                         color=TEXT_COLOR, fontsize=11, alpha=0.5)

    # ── DnD ──

    def _setup_dnd(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass  # tkinterdnd2 not available, drag & drop disabled

    def _on_drop(self, event):
        raw = event.data
        if raw.startswith("{"):
            paths = [p.strip("{}") for p in raw.split("} {")]
        else:
            paths = raw.split()
        for p in paths:
            p = p.strip()
            if os.path.splitext(p)[1].lower() in ALL_EXTENSIONS and os.path.isfile(p):
                self._set_file(p)
                return
        messagebox.showwarning("Unsupported file", "Please drop an audio or video file.")

    # ── File ──

    def _set_file(self, path):
        self._auto_stop_tap_recording()
        self._stop_playback()
        self._close_video_window()
        self.file_path.set(path)
        if not self.out_dir.get():
            self.out_dir.set(os.path.dirname(path))
        self.y = None
        self.y_raw = None
        self.sr = None
        self.duration = 0
        self.beats_ms = None
        self.beats_intensity = {}
        self._override_zones = []
        self._selected_zone_idx = None
        self._last_funscript = None
        self._t_env = None
        self._env_max = None
        self._env_mipmap = None
        self._wave_fill = None
        self._fun_line = None
        self._fun_fill = None
        self._playhead_pos = 0
        self._undo_stack = []
        self._redo_stack = []
        self._set_buttons_state("disabled")
        self.progress.config(value=0)
        Thread(target=self._load_and_show_waveform, args=(path,), daemon=True).start()

    def _load_and_show_waveform(self, source_path):
        try:
            audio_path = source_path
            if is_video(source_path):
                audio_path = extract_audio_from_video(source_path, self._progress_cb)
                self.tmp_audio = audio_path

            y_raw, self.sr = load_audio(audio_path, self._progress_cb)
            self.y_raw = y_raw
            if self.auto_normalize.get():
                pct = self.norm_percentile.get()
                self.y = normalize_audio(y_raw, percentile=pct)
            else:
                self.y = y_raw.copy()
            self.duration = len(self.y) / self.sr

            if self.tmp_audio:
                try:
                    os.unlink(self.tmp_audio)
                except OSError:
                    pass
                self.tmp_audio = None

            self.root.after(0, lambda: self.in_var.set(_sec_to_tc(0)))
            self.root.after(0, lambda: self.out_var.set(_sec_to_tc(self.duration)))
            self.root.after(0, lambda: self.time_var.set(
                f"0:00:00.000 / {_format_time_ms(self.duration)}"))

            self._progress_cb("Drawing waveform…", 80)
            self.root.after(0, self._draw_waveform_only)
            self._progress_cb("Ready", 100)
        except Exception as e:
            self._progress_cb("Error loading file", 0)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self._set_buttons_state("normal"))

    def _draw_waveform_only(self):
        self._draw_waveform()
        self.ax_fun.clear()
        self._fun_line = None
        self._fun_fill = None
        self._style_axes()
        self.ax_fun.set_ylabel("Pos", color=TEXT_COLOR, fontsize=9)
        self.ax_fun.set_ylim(-5, 105)
        self.ax_fun.set_yticks([0, 25, 50, 75, 100])
        self.ax_fun.xaxis.set_major_formatter(mticker.FuncFormatter(_format_time))
        self._draw_zones_lod()
        self._draw_markers()
        self._draw_playhead_lines()
        self._update_shading()
        self._update_time_text()
        self.fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.08, hspace=0.08)
        self._sync_scrollbar()
        self.canvas.draw()
        # Restore project data if a project was loading
        if self._pending_project is not None:
            data = self._pending_project
            self._pending_project = None
            self._apply_project_data(data)

    def _browse_file(self):
        exts_all = " ".join(f"*{e}" for e in sorted(ALL_EXTENSIONS))
        exts_audio = " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))
        exts_video = " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))
        path = filedialog.askopenfilename(
            title="Select Audio or Video File",
            filetypes=[("All supported", exts_all),
                       ("Audio files", exts_audio),
                       ("Video files", exts_video),
                       ("All files", "*.*")])
        if path:
            self._set_file(path)

    def _browse_outdir(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.out_dir.set(path)

    # ── Playback ──

    def _on_space(self, event):
        if isinstance(event.widget, (ttk.Entry, tk.Entry)):
            return
        if self.y is None:
            return
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self.y is None or self.sr is None:
            return

        # Start from current playhead position
        start_pos = self._playhead_pos

        # Determine end position
        out_sec = _tc_to_sec(self.out_var.get())
        if out_sec is None:
            out_sec = self.duration

        # If playhead is past the out marker, reset to in marker
        if start_pos >= out_sec:
            start_pos = _tc_to_sec(self.in_var.get())
            if start_pos is None:
                start_pos = 0

        start_sample = int(start_pos * self.sr)
        end_sample = int(out_sec * self.sr)
        start_sample = max(0, min(start_sample, len(self.y)))
        end_sample = max(start_sample, min(end_sample, len(self.y)))

        audio_chunk = self.y[start_sample:end_sample]
        if len(audio_chunk) == 0:
            return

        self._playing = True
        self._play_start_pos = start_pos
        self._play_end_pos = out_sec
        self._play_chunk = audio_chunk
        self._play_sample_idx = 0

        # Use OutputStream with callback for sample-accurate position tracking
        def _audio_callback(outdata, frames, time_info, status):
            idx = self._play_sample_idx
            end = idx + frames
            if end <= len(self._play_chunk):
                outdata[:, 0] = self._play_chunk[idx:end]
                self._play_sample_idx = end
            else:
                # Fill remaining with silence, signal stop
                remaining = len(self._play_chunk) - idx
                if remaining > 0:
                    outdata[:remaining, 0] = self._play_chunk[idx:]
                outdata[max(0, remaining):] = 0
                self._play_sample_idx = len(self._play_chunk)
                raise sd.CallbackStop()

        self._play_stream = sd.OutputStream(
            samplerate=int(self.sr * self._playback_speed), channels=1,
            dtype='float32', callback=_audio_callback)
        self._play_stream.start()
        self._tick_playhead()

    def _stop_playback(self):
        if self._playing:
            # Save current position from sample counter
            self._playhead_pos = self._get_playhead_pos()
        self._playing = False
        if self._play_stream is not None:
            try:
                self._play_stream.stop()
                self._play_stream.close()
            except Exception:
                pass
            self._play_stream = None
        self._play_chunk = None
        if self._playhead_timer is not None:
            self.root.after_cancel(self._playhead_timer)
            self._playhead_timer = None

    def _get_marker_bounds(self):
        """Return (in_sec, out_sec) clamped to valid range."""
        in_sec = _tc_to_sec(self.in_var.get())
        if in_sec is None:
            in_sec = 0
        out_sec = _tc_to_sec(self.out_var.get())
        if out_sec is None:
            out_sec = self.duration if self.duration > 0 else 0
        if self.duration > 0:
            out_sec = min(out_sec, self.duration)
        return max(0, in_sec), out_sec

    def _clamp_playhead_to_markers(self):
        """Shift playhead if it's outside the current marker range."""
        in_sec, out_sec = self._get_marker_bounds()
        if self._playhead_pos < in_sec:
            self._seek_to(in_sec)
        elif self._playhead_pos > out_sec:
            self._seek_to(out_sec)

    def _seek_to(self, pos_sec):
        """Seek playhead to a specific position, clamped to marker range."""
        was_playing = self._playing
        if was_playing:
            self._stop_playback()

        in_sec, out_sec = self._get_marker_bounds()
        pos_sec = max(in_sec, min(pos_sec, out_sec))
        self._playhead_pos = pos_sec
        self._update_playhead_position(pos_sec)
        self.time_var.set(
            f"{_format_time_ms(pos_sec)} / {_format_time_ms(self.duration)}")
        self.canvas.draw_idle()

        if was_playing:
            self._start_playback()
        self._update_video_frame(pos_sec)

    def _on_speed_slider(self, val):
        """Handle playback speed slider change."""
        speed = round(max(0.1, min(3.0, val)), 2)
        self._playback_speed = speed
        self._speed_label_var.set(f"{speed:.1f}x")
        # Restart stream at new speed if currently playing
        if self._playing:
            pos = self._get_playhead_pos()
            self._stop_playback()
            self._playhead_pos = pos
            self._start_playback()

    # ── Video preview window ──

    def _toggle_video_window(self):
        """Open or close the video preview window."""
        if self._video_window is not None:
            self._close_video_window()
            return
        fp = self.file_path.get().strip()
        if not fp or not is_video(fp):
            messagebox.showinfo("No video", "Load a video file first.")
            return
        try:
            import cv2
        except ImportError:
            messagebox.showerror("Missing dependency",
                                 "Install opencv-python:\n  pip install opencv-python")
            return
        cap = cv2.VideoCapture(fp)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{fp}")
            return
        self._video_cap = cap
        self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Scale to a reasonable preview size (max 640px wide)
        max_w = 640
        if w > max_w:
            scale = max_w / w
            w = max_w
            h = int(h * scale)
        self._video_size = (w, h)
        self._video_aspect = w / max(1, h)

        win = tk.Toplevel(self.root)
        win.title("Video Preview")
        win.geometry(f"{w}x{h}")
        win.configure(bg="#1a1b26")
        win.protocol("WM_DELETE_WINDOW", self._close_video_window)
        win.resizable(True, True)
        self._video_window = win

        self._video_label = tk.Label(win, bg="#1a1b26")
        self._video_label.pack(fill="both", expand=True)

        self._video_last_frame_sec = -1
        self._update_video_frame(self._playhead_pos)

    def _close_video_window(self):
        """Close the video preview window and release resources."""
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
        if self._video_window is not None:
            self._video_window.destroy()
            self._video_window = None
        self._video_label = None
        self._video_last_frame_sec = -1

    def _update_video_frame(self, pos_sec):
        """Show the video frame at pos_sec in the preview window."""
        if self._video_cap is None or self._video_label is None:
            return
        # Avoid redundant seeks for the same frame
        frame_dur = 1.0 / self._video_fps
        if abs(pos_sec - self._video_last_frame_sec) < frame_dur * 0.4:
            return
        self._video_last_frame_sec = pos_sec
        try:
            import cv2
            from PIL import Image, ImageTk
        except ImportError:
            return
        self._video_cap.set(cv2.CAP_PROP_POS_MSEC, pos_sec * 1000)
        ret, frame = self._video_cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use window size for dynamic resizing, maintain aspect ratio
        try:
            win_w = self._video_window.winfo_width()
            win_h = self._video_window.winfo_height()
            if win_w > 1 and win_h > 1:
                aspect = self._video_aspect
                fit_w = win_w
                fit_h = int(win_w / aspect)
                if fit_h > win_h:
                    fit_h = win_h
                    fit_w = int(win_h * aspect)
                w, h = max(1, fit_w), max(1, fit_h)
            else:
                w, h = self._video_size
        except Exception:
            w, h = self._video_size
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(img)
        self._video_label.config(image=photo)
        self._video_label._photo = photo  # prevent garbage collection

    def _get_playhead_pos(self):
        if not self._playing:
            return self._playhead_pos
        # Sample-accurate position from the audio callback counter
        elapsed_sec = self._play_sample_idx / self.sr
        return min(self._play_start_pos + elapsed_sec, self._play_end_pos)

    def _tick_playhead(self):
        if not self._playing:
            return

        pos = self._get_playhead_pos()

        # Check if stream finished naturally
        stream_done = (self._play_stream is not None
                       and not self._play_stream.active)
        if stream_done or pos >= self._play_end_pos:
            self._playhead_pos = self._play_end_pos
            self._stop_playback()
            self._update_playhead_position(self._playhead_pos)
            self.time_var.set(
                f"{_format_time_ms(self._playhead_pos)} / {_format_time_ms(self.duration)}")
            self.canvas.draw_idle()
            return

        self._playhead_pos = pos
        self._update_playhead_position(pos)
        self.time_var.set(
            f"{_format_time_ms(pos)} / {_format_time_ms(self.duration)}")
        self._update_video_frame(pos)

        # Auto-scroll when playhead exits view
        xlim = self.ax_wave.get_xlim()
        if pos > xlim[1]:
            span = xlim[1] - xlim[0]
            new_left = pos
            new_right = new_left + span
            new_left, new_right = self._clamp_xlim(new_left, new_right)
            self.ax_wave.set_xlim(new_left, new_right)
            self.ax_fun.set_xlim(new_left, new_right)
            self._on_view_changed()
            self._sync_scrollbar()

        self.canvas.draw_idle()
        self._playhead_timer = self.root.after(50, self._tick_playhead)

    def _draw_playhead_lines(self):
        pos = self._playhead_pos
        self._playhead_line_w = self.ax_wave.axvline(
            pos, color=PLAYHEAD_COLOR, linewidth=1.5, alpha=0.9, zorder=15)
        self._playhead_line_f = self.ax_fun.axvline(
            pos, color=PLAYHEAD_COLOR, linewidth=1.5, alpha=0.9, zorder=15)
        self._playhead_line_z = self.ax_zones.axvline(
            pos, color=PLAYHEAD_COLOR, linewidth=1.5, alpha=0.9, zorder=15)

    def _update_playhead_position(self, pos):
        if self._playhead_line_w is not None:
            self._playhead_line_w.set_xdata([pos, pos])
        if self._playhead_line_f is not None:
            self._playhead_line_f.set_xdata([pos, pos])
        if hasattr(self, '_playhead_line_z') and self._playhead_line_z is not None:
            self._playhead_line_z.set_xdata([pos, pos])

    # ── Markers ──

    def _get_in_out_ms(self):
        in_ms = out_ms = None
        v = _tc_to_sec(self.in_var.get())
        if v is not None and v > 0:
            in_ms = v * 1000
        v = _tc_to_sec(self.out_var.get())
        if v is not None:
            out_ms = v * 1000
        return in_ms, out_ms

    def _reset_markers(self):
        self.in_var.set(_sec_to_tc(0))
        if self.duration > 0:
            self.out_var.set(_sec_to_tc(self.duration))
        else:
            self.out_var.set("—")

    def _update_marker_lines(self):
        in_sec = _tc_to_sec(self.in_var.get())
        if in_sec is None:
            in_sec = 0
        out_sec = _tc_to_sec(self.out_var.get())

        for line in (self.in_line_w, self.in_line_f, getattr(self, 'in_line_z', None)):
            if line is not None:
                line.set_xdata([in_sec, in_sec])
        for line in (self.out_line_w, self.out_line_f, getattr(self, 'out_line_z', None)):
            if line is not None and out_sec is not None:
                line.set_xdata([out_sec, out_sec])

        if self.in_handle is not None:
            self.in_handle.set_data([in_sec], [self._handle_y()])
        if self.out_handle is not None and out_sec is not None:
            self.out_handle.set_data([out_sec], [self._handle_y()])

        self._update_shading()
        self._clamp_playhead_to_markers()
        if not self._suppress_marker_draw:
            self.canvas.draw_idle()

    def _handle_y(self):
        """Y position for marker handles: exactly at the top border of the waveform."""
        ylim = self.ax_wave.get_ylim()
        return ylim[1]

    def _update_handle_visibility(self):
        """Hide handles that are outside the visible x-range."""
        xlim = self.ax_wave.get_xlim()
        for handle, var in ((self.in_handle, self.in_var),
                            (self.out_handle, self.out_var)):
            if handle is None:
                continue
            sec = _tc_to_sec(var.get())
            if sec is None:
                sec = 0
            handle.set_visible(xlim[0] <= sec <= xlim[1])

    def _is_near_pin(self, event):
        """Check if mouse event is in the pin handle zone (above the waveform top border)."""
        if event.x is None or event.y is None:
            return False
        try:
            bbox = self.ax_wave.get_window_extent()
            # Pin zone: from a few pixels below the top border to ~30px above it
            return bbox.y1 - 8 <= event.y <= bbox.y1 + 35
        except Exception:
            return False

    def _update_marker_hover(self, x_sec, near_pin=False):
        """Update hover state for in/out marker pin handles.
        Only activates when near_pin is True (mouse is near pin area)."""
        nearest = None
        if near_pin:
            xlim = self.ax_wave.get_xlim()
            tol = (xlim[1] - xlim[0]) * 0.02
            in_sec = _tc_to_sec(self.in_var.get())
            out_sec = _tc_to_sec(self.out_var.get())

            if in_sec is not None and abs(x_sec - in_sec) < tol:
                nearest = "in"
            if out_sec is not None and abs(x_sec - out_sec) < tol:
                if nearest is None or abs(x_sec - out_sec) < abs(x_sec - in_sec):
                    nearest = "out"

        if nearest == self._hover_marker:
            return  # no change

        # Reset previous hover
        if self.in_handle is not None:
            self.in_handle.set_markersize(14)
            self.in_handle.set_markeredgewidth(0.8)
            self.in_handle.set_alpha(0.95)
        if self.out_handle is not None:
            self.out_handle.set_markersize(14)
            self.out_handle.set_markeredgewidth(0.8)
            self.out_handle.set_alpha(0.95)

        # Apply hover effect
        self._hover_marker = nearest
        if nearest == "in" and self.in_handle is not None:
            self.in_handle.set_markersize(17)
            self.in_handle.set_markeredgewidth(1.5)
            self.in_handle.set_alpha(1.0)
        elif nearest == "out" and self.out_handle is not None:
            self.out_handle.set_markersize(17)
            self.out_handle.set_markeredgewidth(1.5)
            self.out_handle.set_alpha(1.0)

        self.canvas.draw_idle()

    def _update_shading(self):
        """Draw semi-transparent dark overlays outside the In/Out marker range."""
        # Remove old patches
        for p in self._dim_patches:
            try:
                p.remove()
            except Exception:
                pass
        self._dim_patches = []

        in_sec = _tc_to_sec(self.in_var.get())
        if in_sec is None:
            in_sec = 0
        out_sec = _tc_to_sec(self.out_var.get())
        if out_sec is None:
            out_sec = self.duration if self.duration > 0 else 0

        # Nothing to dim if markers span the whole timeline
        if in_sec <= 0 and out_sec >= self.duration:
            return

        from matplotlib.patches import Rectangle
        DIM_COLOR = "#1a1b26"
        DIM_ALPHA = 0.55

        for ax in (self.ax_wave, self.ax_fun, self.ax_zones):
            ylim = ax.get_ylim()
            y0 = min(ylim)
            height = max(ylim) - y0
            # Left dim (before In marker)
            if in_sec > 0:
                rect = Rectangle((0, y0), in_sec, height,
                                 facecolor=DIM_COLOR, alpha=DIM_ALPHA,
                                 edgecolor='none', zorder=15)
                ax.add_patch(rect)
                self._dim_patches.append(rect)
            # Right dim (after Out marker)
            dur = self.duration if self.duration > 0 else out_sec
            if out_sec < dur:
                rect = Rectangle((out_sec, y0), dur - out_sec, height,
                                 facecolor=DIM_COLOR, alpha=DIM_ALPHA,
                                 edgecolor='none', zorder=15)
                ax.add_patch(rect)
                self._dim_patches.append(rect)

    def _on_draw_fps(self, event):
        """Update FPS counter on each draw."""
        now = _time.time()
        self._fps_frame_count += 1
        dt = now - self._fps_last_time
        if dt >= 0.5:
            self._fps_value = self._fps_frame_count / dt
            self._fps_frame_count = 0
            self._fps_last_time = now
            if self._fps_text is not None:
                self._fps_text.set_text(f"{self._fps_value:.0f} FPS")

    # ── Scrollbar ──

    def _on_scrollbar(self, *args):
        """Handle scrollbar interaction."""
        if self.duration <= 0 or self._updating_scrollbar:
            return
        cmd = args[0]
        if cmd == "moveto":
            frac = float(args[1])
            xlim = self.ax_wave.get_xlim()
            span = xlim[1] - xlim[0]
            left = frac * self.duration
            right = left + span
            left, right = self._clamp_xlim(left, right)
            self.ax_wave.set_xlim(left, right)
            self._on_view_changed()
            self._sync_scrollbar()
            self.canvas.draw_idle()
        elif cmd == "scroll":
            amount = int(args[1])
            self._pan(amount * 0.05)

    def _sync_scrollbar(self):
        """Update scrollbar thumb to reflect current view."""
        if self.duration <= 0:
            return
        try:
            xlim = self.ax_wave.get_xlim()
            lo = max(0, xlim[0] / self.duration)
            hi = min(1, xlim[1] / self.duration)
            self._updating_scrollbar = True
            self._scrollbar.set(lo, hi)
            self._updating_scrollbar = False
        except Exception:
            self._updating_scrollbar = False

    # ── Navigation ──

    def _clamp_xlim(self, left, right):
        if self.duration <= 0:
            return left, right
        span = right - left
        if span > self.duration:
            span = self.duration
        if left < 0:
            left = 0
            right = left + span
        if right > self.duration:
            right = self.duration
            left = right - span
        if left < 0:
            left = 0
        return left, right

    def _pan(self, fraction):
        xlim = self.ax_wave.get_xlim()
        span = xlim[1] - xlim[0]
        shift = span * fraction
        left, right = self._clamp_xlim(xlim[0] + shift, xlim[1] + shift)
        self.ax_wave.set_xlim(left, right)
        self._on_view_changed()
        self._sync_scrollbar()
        self.canvas.draw_idle()

    def _on_key_left(self, event):
        self._step_playhead(-0.1)

    def _on_key_right(self, event):
        self._step_playhead(0.1)

    def _on_key_ctrl_left(self, event):
        self._step_playhead(-1.0)

    def _on_key_ctrl_right(self, event):
        self._step_playhead(1.0)

    def _on_key_home(self, event):
        in_sec, _ = self._get_marker_bounds()
        self._jump_playhead(in_sec)

    def _on_key_end(self, event):
        _, out_sec = self._get_marker_bounds()
        self._jump_playhead(out_sec)

    def _step_playhead(self, delta_sec):
        """Move playhead by delta_sec and scroll view to keep it visible."""
        if self.y is None:
            return
        new_pos = max(0, min(self.duration, self._playhead_pos + delta_sec))
        self._seek_to(new_pos)
        self._ensure_playhead_visible(new_pos)

    def _jump_playhead(self, pos_sec):
        """Jump playhead to pos_sec and centre view on it."""
        if self.y is None:
            return
        pos_sec = max(0, min(self.duration, pos_sec))
        self._seek_to(pos_sec)
        self._ensure_playhead_visible(pos_sec)

    def _ensure_playhead_visible(self, pos):
        """Scroll the view so pos is visible, preserving zoom level."""
        xlim = self.ax_wave.get_xlim()
        span = xlim[1] - xlim[0]
        if pos < xlim[0] or pos > xlim[1]:
            # Centre the view on the playhead
            left = pos - span / 2
            right = left + span
            left, right = self._clamp_xlim(left, right)
            self.ax_wave.set_xlim(left, right)
            self._on_view_changed()
            self._sync_scrollbar()
            self.canvas.draw_idle()

    def _on_scroll(self, event):
        self._auto_stop_tap_recording()
        if event.inaxes not in (self.ax_wave, self.ax_fun, self.ax_zones):
            return

        # Shift+scroll: adjust multi-beat spacing
        if self._mod_shift:
            step = 0.01 if event.button == "up" else -0.01
            new_val = max(0.01, self.multi_beat_spacing.get() + step)
            self.multi_beat_spacing.set(round(new_val, 3))
            self.status.set(f"Beat spacing: {self.multi_beat_spacing.get():.3f} s")
            # Update preview at current mouse position
            xdata = event.xdata
            if xdata is not None:
                self._update_multi_beat_preview(xdata)
                self.canvas.draw_idle()
            return

        xlim = self.ax_wave.get_xlim()
        xdata = event.xdata
        if xdata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        new_w = (xlim[1] - xlim[0]) * scale
        left = xdata - (xdata - xlim[0]) * scale
        right = left + new_w
        left, right = self._clamp_xlim(left, right)
        self.ax_wave.set_xlim(left, right)
        self._on_view_changed()
        self._sync_scrollbar()
        self.canvas.draw_idle()

    # ── Mouse ──

    def _find_nearest_beat(self, x_sec):
        """Return (index, distance_sec) of the nearest beat to x_sec, or (None, inf)."""
        if not self.beats_ms:
            return None, float('inf')
        import bisect
        x_ms = x_sec * 1000
        pos = bisect.bisect_left(self.beats_ms, x_ms)
        best_idx = None
        best_dist = float('inf')
        for i in (pos - 1, pos):
            if 0 <= i < len(self.beats_ms):
                d = abs(self.beats_ms[i] - x_ms) / 1000.0
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        return best_idx, best_dist

    def _update_beat_hover(self, x_sec):
        """Highlight the nearest beat if within tolerance, else clear."""
        xlim = self.ax_wave.get_xlim()
        tol = (xlim[1] - xlim[0]) * 0.015  # 1.5% of visible span
        idx, dist = self._find_nearest_beat(x_sec)
        if idx is not None and dist <= tol:
            if idx == self._hover_beat_idx:
                return  # already highlighted
            self._clear_beat_hover()
            self._hover_beat_idx = idx
            beat_sec = self.beats_ms[idx] / 1000.0
            ym = self._beat_ym
            self._hover_line_w = self.ax_wave.axvline(
                beat_sec, color="#ffffff", linewidth=2.0, alpha=0.8, zorder=12)
            self._hover_line_f = self.ax_fun.axvline(
                beat_sec, color="#ffffff", linewidth=2.0, alpha=0.8, zorder=12)
            self.canvas.draw_idle()
        else:
            if self._hover_beat_idx is not None:
                self._clear_beat_hover()
                self.canvas.draw_idle()

    def _clear_beat_hover(self):
        """Remove hover highlight artists."""
        for line in (self._hover_line_w, self._hover_line_f):
            if line is not None:
                try:
                    line.remove()
                except Exception:
                    pass
        self._hover_line_w = None
        self._hover_line_f = None
        self._hover_beat_idx = None

    def _get_beat_intensity(self):
        """Return the currently selected beat intensity as a float (0.0–1.0)."""
        return max(0, min(100, self.beat_intensity_var.get())) / 100.0

    def _on_intensity_slider(self, val):
        """Handle intensity slider movement."""
        v = int(round(val))
        self.beat_intensity_var.set(v)
        self._intensity_label.config(text=f"{v}%")
        self._on_intensity_changed()

    def _on_preset_click(self, idx):
        """Click on preset button: store current intensity value in that preset."""
        val = self.beat_intensity_var.get()
        self._intensity_presets[idx] = val
        self._preset_btns[idx].config(text=f"{val}%")

    def _recall_intensity_preset(self, idx):
        """Key 1-5: set slider to the preset value."""
        val = self._intensity_presets[idx]
        self.beat_intensity_var.set(val)
        self._intensity_label.config(text=f"{val}%")
        self._on_intensity_changed()

    def _on_mode_changed(self):
        """Regenerate and redraw funscript when movement mode changes."""
        if not self.beats_ms:
            return
        self._regenerate_funscript()
        self._draw_fun_lod()
        self.canvas.draw_idle()

    # ── Modifier key tracking ──

    def _set_mod(self, key, state):
        if key == "shift":
            self._mod_shift = state
        elif key == "ctrl":
            self._mod_ctrl = state

    def _reset_mods(self):
        self._mod_shift = False
        self._mod_ctrl = False

    # ── Copy / Paste beats ──

    def _copy_beats(self):
        """Copy selected beats to clipboard (as offsets relative to first)."""
        if not self._selected_beats:
            self.status.set("No beats selected to copy")
            return
        sorted_ms = sorted(self._selected_beats)
        first = sorted_ms[0]
        self._clipboard_beats = [
            (ms - first, self.beats_intensity.get(ms, 1.0))
            for ms in sorted_ms]
        self.status.set(f"Copied {len(self._clipboard_beats)} beats")

    def _paste_beats(self):
        """Paste copied beats at the current playhead position."""
        if not self._clipboard_beats:
            self.status.set("Nothing to paste")
            return
        self._push_undo()
        if self.beats_ms is None:
            self.beats_ms = []
        # Paste at playhead position
        base_sec = self._playhead_pos
        if self.snap_to_spike.get():
            base_sec = self._snap_to_spike(base_sec)
        snap_first = self.snap_first_only.get()
        import bisect
        new_selected = set()
        for i, (offset_ms, inten) in enumerate(self._clipboard_beats):
            t_sec = base_sec + offset_ms / 1000.0
            # Snap: if snap_first_only, only snap the first beat
            if self.snap_to_spike.get() and not snap_first:
                if i > 0:
                    t_sec = self._snap_to_spike(t_sec)
            t_ms = int(round(t_sec * 1000))
            if t_ms < 0:
                continue
            if self.duration > 0 and t_ms > int(self.duration * 1000):
                continue
            pos = bisect.bisect_left(self.beats_ms, t_ms)
            if pos < len(self.beats_ms) and self.beats_ms[pos] == t_ms:
                continue  # skip duplicate
            self.beats_ms.insert(pos, t_ms)
            if inten != 1.0:
                self.beats_intensity[t_ms] = inten
            new_selected.add(t_ms)
        self._selected_beats = new_selected
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self.canvas.draw_idle()
        self.status.set(f"Pasted {len(new_selected)} beats at {_format_time_ms(base_sec)}")

    def _on_intensity_changed(self):
        """Apply intensity to all selected beats when slider/key changes."""
        if not self._selected_beats:
            return
        inten = self._get_beat_intensity()
        for b_ms in list(self._selected_beats):
            if b_ms not in (self.beats_ms or []):
                continue
            if inten >= 1.0:
                self.beats_intensity.pop(b_ms, None)
            else:
                self.beats_intensity[b_ms] = inten
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self.canvas.draw_idle()
        self.status.set(f"Set {len(self._selected_beats)} beats to {int(inten * 100)}%")

    def _delete_selected_beats(self):
        """Delete all selected beats."""
        if not self._selected_beats or not self.beats_ms:
            return
        self._push_undo()
        self._clear_beat_hover()
        to_remove = set(self._selected_beats)
        self.beats_ms = [b for b in self.beats_ms if b not in to_remove]
        for b_ms in to_remove:
            self.beats_intensity.pop(b_ms, None)
        count = len(to_remove)
        self._selected_beats.clear()
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self.canvas.draw_idle()
        self.status.set(f"Deleted {count} selected beats — {len(self.beats_ms)} remaining")

    def _clear_selection(self):
        """Clear beat and keyframe selection and remove visual indicators."""
        self._selected_beats.clear()
        self._selected_kf.clear()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self.canvas.draw_idle()

    def _remove_snap_preview(self):
        """Remove snap preview ghost line."""
        if self._snap_preview_line is not None:
            try:
                self._snap_preview_line.remove()
            except Exception:
                pass
            self._snap_preview_line = None

    def _update_snap_preview(self, x_sec):
        """Show a ghost marker where snapping would place the beat."""
        self._remove_snap_preview()
        if self.snap_mode.get() == "Off":
            return
        snapped = self._snap_to_spike(x_sec)
        if abs(snapped - x_sec) < 1e-6:
            return  # no snap offset, no preview needed
        ym = self._beat_ym
        self._snap_preview_line = self.ax_wave.axvline(
            snapped, color="#ffffff", linewidth=1.5, alpha=0.35,
            linestyle="--", zorder=11)

    def _remove_multi_beat_preview(self):
        """Remove shift+hover multi-beat preview markers."""
        for a in self._multi_beat_preview_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._multi_beat_preview_artists = []

    def _get_multi_beat_times(self, start_sec, count):
        """Compute positions for multi-beat placement from start_sec."""
        times = []
        spacing_mode = self.multi_beat_spacing_mode.get()
        snap_first = self.snap_first_only.get()

        if spacing_mode == "Tempo" and self._tempo_grid_times is not None and len(self._tempo_grid_times) > 0:
            # Use tempo grid: find the nearest grid point at or after start_sec,
            # then take the next 'count' grid points (including sub-beats)
            grid = self._tempo_grid_times
            if self._tempo_sub_times is not None and len(self._tempo_sub_times) > 0:
                grid = np.concatenate([grid, self._tempo_sub_times])
                grid.sort()
            idx = np.searchsorted(grid, start_sec - 0.001)
            for i in range(count):
                gi = idx + i
                if gi >= len(grid):
                    break
                t = float(grid[gi])
                if self.duration > 0 and t > self.duration:
                    break
                times.append(t)
        else:
            spacing = self.multi_beat_spacing.get()
            if spacing <= 0:
                return times
            for i in range(count):
                t = start_sec + i * spacing
                if self.duration > 0 and t > self.duration:
                    break
                if self.snap_to_spike.get() and not (snap_first and i > 0):
                    t = self._snap_to_spike(t)
                times.append(t)
        return times

    def _update_multi_beat_preview(self, x_sec):
        """Show ghost markers where shift+click multi-beats would be placed."""
        self._remove_multi_beat_preview()
        count = self.multi_beat_count.get()
        if count <= 0:
            return
        times = self._get_multi_beat_times(x_sec, count)
        for t in times:
            line = self.ax_wave.axvline(
                t, color="#9ece6a", linewidth=1.2, alpha=0.4,
                linestyle=":", zorder=11)
            self._multi_beat_preview_artists.append(line)

    def _start_selection_rect(self, x_data, y_pixel):
        """Begin drawing an RTS-style selection rectangle."""
        self._selection_start_xy = (x_data, y_pixel)
        self._selecting = True

    def _update_selection_rect(self, x_data, y_pixel):
        """Update the selection rectangle as the mouse moves."""
        if self._selection_rect_artist is not None:
            try:
                self._selection_rect_artist.remove()
            except Exception:
                pass
            self._selection_rect_artist = None
        if self._selection_start_xy is None:
            return
        x0, _ = self._selection_start_xy
        x1 = x_data
        left = min(x0, x1)
        right = max(x0, x1)
        # Draw a semi-transparent rectangle across the full waveform height
        ylim = self.ax_wave.get_ylim()
        from matplotlib.patches import Rectangle
        rect = Rectangle((left, ylim[0]), right - left, ylim[1] - ylim[0],
                          facecolor="#7aa2f7", alpha=0.15, edgecolor="#7aa2f7",
                          linewidth=1, linestyle="--", zorder=15)
        self.ax_wave.add_patch(rect)
        self._selection_rect_artist = rect

    def _finish_selection_rect(self, x_data):
        """Finalize selection rectangle: select beats within it."""
        if self._selection_start_xy is None:
            return
        x0, _ = self._selection_start_xy
        x1 = x_data
        left_ms = int(min(x0, x1) * 1000)
        right_ms = int(max(x0, x1) * 1000)
        # Remove the rect artist
        if self._selection_rect_artist is not None:
            try:
                self._selection_rect_artist.remove()
            except Exception:
                pass
            self._selection_rect_artist = None
        self._selection_start_xy = None
        self._selecting = False
        # Select beats in range
        if self.beats_ms:
            self._selected_beats = {
                b for b in self.beats_ms if left_ms <= b <= right_ms}
        if self._selected_beats:
            self.status.set(f"Selected {len(self._selected_beats)} beats")
        self._draw_beat_lod()
        self.canvas.draw_idle()

    def _normalize_now(self):
        """Re-normalize audio within the marker range with current percentile."""
        if self.y_raw is None:
            return
        pct = self.norm_percentile.get()
        in_sec, out_sec = self._get_marker_bounds()
        s0 = int(in_sec * self.sr)
        s1 = int(out_sec * self.sr)
        s0 = max(0, min(s0, len(self.y_raw)))
        s1 = max(s0, min(s1, len(self.y_raw)))
        # Normalize only the marker range, keep rest as-is
        self.y = self.y_raw.copy()
        if s1 > s0:
            self.y[s0:s1] = normalize_audio(self.y_raw[s0:s1], percentile=pct)
        # Invalidate cached envelope so it's recomputed
        self._t_env = None
        self._env_max = None
        self._env_mipmap = None
        # Redraw waveform preserving zoom/pan
        self._draw_waveform(preserve_view=True)
        self._draw_markers()
        self._draw_playhead_lines()
        self._update_shading()
        self._sync_scrollbar()
        self.canvas.draw()
        self.status.set(f"Normalized at {pct:.0f}th percentile")

    def _snap_to_spike(self, x_sec):
        """Snap to nearest peak or musical beat depending on snap_mode.
        Snap strength controls the search radius (0.05=tight, 1.0=wide).
        Holding Ctrl or Shift while dragging temporarily disables snapping."""
        if self._dragging is not None and (self._mod_ctrl or self._mod_shift):
            return x_sec
        mode = self.snap_mode.get()
        if mode == "Off":
            return x_sec

        xlim = self.ax_wave.get_xlim()
        strength = self.snap_strength.get()
        radius = max(0.01, (xlim[1] - xlim[0]) * 0.05 * strength)

        if mode == "Musical Beat":
            if self._tempo_grid_times is None or len(self._tempo_grid_times) == 0:
                return x_sec
            # Combine main grid and sub-beats for snapping
            grid = self._tempo_grid_times
            if self._tempo_sub_times is not None and len(self._tempo_sub_times) > 0:
                grid = np.concatenate([grid, self._tempo_sub_times])
                grid.sort()
            idx = np.searchsorted(grid, x_sec)
            best_dist = float("inf")
            best_t = x_sec
            for ci in (idx - 1, idx):
                if 0 <= ci < len(grid):
                    d = abs(grid[ci] - x_sec)
                    if d < best_dist:
                        best_dist = d
                        best_t = float(grid[ci])
            return best_t if best_dist <= radius else x_sec

        # mode == "Peaks" — original envelope-based snap
        if self._t_env is None or self._env_max is None:
            return x_sec
        t_env = self._t_env
        env = self._env_max
        mask = (t_env >= x_sec - radius) & (t_env <= x_sec + radius)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return x_sec
        best = indices[np.argmax(env[indices])]
        return float(t_env[best])

    def _add_beat_at(self, x_sec, snap=True):
        """Insert a beat at the given time and refresh display."""
        self._push_undo()
        if self.beats_ms is None:
            self.beats_ms = []
        if snap:
            x_sec = self._snap_to_spike(x_sec)
        import bisect
        t_ms = int(round(x_sec * 1000))
        if t_ms < 0:
            return
        # Don't add duplicate
        pos = bisect.bisect_left(self.beats_ms, t_ms)
        if pos < len(self.beats_ms) and self.beats_ms[pos] == t_ms:
            return
        self.beats_ms.insert(pos, t_ms)
        intensity = self._get_beat_intensity()
        if intensity != 1.0:
            self.beats_intensity[t_ms] = intensity
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        pct = int(intensity * 100)
        self.status.set(f"Added beat at {_format_time_ms(x_sec)} ({pct}%) — {len(self.beats_ms)} beats total")
        self.canvas.draw_idle()

    def _add_multi_beats_at(self, start_sec):
        """Add multiple beats starting at start_sec using current spacing mode."""
        self._push_undo()
        count = self.multi_beat_count.get()
        if count <= 0:
            return
        times = self._get_multi_beat_times(start_sec, count)
        for t in times:
            self._add_beat_at(t, snap=False)  # times already snapped/positioned
        self.status.set(f"Added {len(times)} beats from {_format_time_ms(start_sec)}")

    def _move_beat(self, idx, new_sec):
        """Move beat at idx to new_sec, preserving intensity. Returns new index or None."""
        if self.beats_ms is None or idx < 0 or idx >= len(self.beats_ms):
            return None
        old_ms = self.beats_ms[idx]
        new_sec = self._snap_to_spike(new_sec)
        new_ms = int(round(new_sec * 1000))
        new_ms = max(0, new_ms)
        if self.duration > 0:
            new_ms = min(int(self.duration * 1000), new_ms)
        if new_ms == old_ms:
            return idx  # no change
        # Preserve intensity
        inten = self.beats_intensity.pop(old_ms, None)
        # Update selection tracking
        was_selected = old_ms in self._selected_beats
        if was_selected:
            self._selected_beats.discard(old_ms)
        del self.beats_ms[idx]
        import bisect
        pos = bisect.bisect_left(self.beats_ms, new_ms)
        # Avoid duplicates
        if pos < len(self.beats_ms) and self.beats_ms[pos] == new_ms:
            # Put old beat back if target already occupied
            self.beats_ms.insert(idx, old_ms)
            if inten is not None:
                self.beats_intensity[old_ms] = inten
            if was_selected:
                self._selected_beats.add(old_ms)
            return idx
        self.beats_ms.insert(pos, new_ms)
        if inten is not None:
            self.beats_intensity[new_ms] = inten
        if was_selected:
            self._selected_beats.add(new_ms)
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self.canvas.draw_idle()
        return pos

    def _remove_beat(self, idx):
        """Remove the beat at index and refresh display."""
        if self.beats_ms is None or idx < 0 or idx >= len(self.beats_ms):
            return
        self._push_undo()
        removed_ms = self.beats_ms[idx]
        self._clear_beat_hover()
        self._selected_beats.discard(removed_ms)
        self.beats_intensity.pop(removed_ms, None)
        del self.beats_ms[idx]
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self.status.set(f"Removed beat at {_format_time_ms(removed_ms / 1000)} — {len(self.beats_ms)} beats total")
        self.canvas.draw_idle()

    def _reset_beats(self):
        """Clear beats and funscript data within work area, redraw."""
        in_sec, out_sec = self._get_marker_bounds()
        in_ms = int(in_sec * 1000)
        out_ms = int(out_sec * 1000)
        count = sum(1 for b in self.beats_ms if in_ms <= b <= out_ms) if self.beats_ms else 0
        if count > 0:
            if not messagebox.askyesno(
                    "Reset Work Area",
                    f"This will delete {count} beats and movement data "
                    "within the work area.\n\nContinue?"):
                return
        self._push_undo()
        self._clear_beat_hover()
        self._selected_beats.clear()
        # Remove beats within marker range, keep those outside
        if self.beats_ms:
            kept = [b for b in self.beats_ms if b < in_ms or b > out_ms]
            for b in list(self.beats_intensity.keys()):
                if in_ms <= b <= out_ms:
                    self.beats_intensity.pop(b, None)
            self.beats_ms = kept if kept else None
        # Remove zones overlapping marker range
        self._override_zones = [z for z in self._override_zones
                                if z["end_ms"] < in_ms or z["start_ms"] > out_ms]
        self._selected_zone_idx = None
        # Remove funscript actions within range
        if self._last_funscript and self._last_funscript.get("actions"):
            self._last_funscript["actions"] = [
                a for a in self._last_funscript["actions"]
                if a["at"] < in_ms or a["at"] > out_ms]
            if not self._last_funscript["actions"]:
                self._last_funscript = None
        self._draw_beat_lod()
        self._draw_fun_lod()
        self._draw_zones_lod()
        self.status.set("Beats and FunScript reset within work area.")
        self.canvas.draw_idle()

    def _reset_beats_only(self):
        """Clear beats within the marker range but keep funscript/zones."""
        if not self.beats_ms:
            return
        in_sec, out_sec = self._get_marker_bounds()
        in_ms = int(in_sec * 1000)
        out_ms = int(out_sec * 1000)
        count = sum(1 for b in self.beats_ms if in_ms <= b <= out_ms)
        if count == 0:
            return
        if not messagebox.askyesno(
                "Reset Beats",
                f"This will delete {count} beats within the work area.\n\nContinue?"):
            return
        self._push_undo()
        self._clear_beat_hover()
        self._clear_kf_hover()
        self._selected_beats.clear()
        self._selected_kf.clear()
        kept = [b for b in self.beats_ms if b < in_ms or b > out_ms]
        for b in list(self.beats_intensity.keys()):
            if in_ms <= b <= out_ms:
                self.beats_intensity.pop(b, None)
        self.beats_ms = kept if kept else None
        self._kf_moved.clear()
        self._kf_deleted.clear()
        self._kf_independent.clear()
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self.status.set(f"Reset {count} beats within work area.")
        self.canvas.draw_idle()

    def _reset_movement(self):
        """Clear funscript and override zones within marker range but keep beats."""
        in_sec, out_sec = self._get_marker_bounds()
        in_ms = int(in_sec * 1000)
        out_ms = int(out_sec * 1000)
        self._push_undo()
        self._clear_kf_hover()
        self._selected_kf.clear()
        # Remove zones overlapping marker range
        self._override_zones = [z for z in self._override_zones
                                if z["end_ms"] < in_ms or z["start_ms"] > out_ms]
        self._selected_zone_idx = None
        self._kf_moved.clear()
        self._kf_deleted.clear()
        self._kf_independent.clear()
        # Remove funscript actions within range
        if self._last_funscript and self._last_funscript.get("actions"):
            self._last_funscript["actions"] = [
                a for a in self._last_funscript["actions"]
                if a["at"] < in_ms or a["at"] > out_ms]
            if not self._last_funscript["actions"]:
                self._last_funscript = None
        self._draw_fun_lod()
        self._draw_zones_lod()
        self.status.set("Movement data reset within work area.")
        self.canvas.draw_idle()

    def _reset_zones(self):
        """Clear override zones within the marker range but keep beats and movement."""
        in_sec, out_sec = self._get_marker_bounds()
        in_ms = int(in_sec * 1000)
        out_ms = int(out_sec * 1000)
        zones_in_range = [z for z in self._override_zones
                          if not (z["end_ms"] < in_ms or z["start_ms"] > out_ms)]
        if not zones_in_range:
            self.status.set("No zones in work area to reset.")
            return
        self._push_undo()
        self._override_zones = [z for z in self._override_zones
                                if z["end_ms"] < in_ms or z["start_ms"] > out_ms]
        self._selected_zone_idx = None
        self._regenerate_funscript()
        self._draw_fun_lod()
        self._draw_zones_lod()
        self.status.set(f"Reset {len(zones_in_range)} override zones within work area.")
        self.canvas.draw_idle()

    def _save_beats(self):
        """Save current beats to a JSON file."""
        if not self.beats_ms:
            messagebox.showwarning("No beats", "No beats to save. Analyze first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.out_dir.get() or None)
        if not path:
            return
        try:
            # Convert intensity keys to strings for JSON
            int_str = {str(k): v for k, v in self.beats_intensity.items()}
            data = {
                "input_file": self.file_path.get(),
                "output_dir": self.out_dir.get(),
                "beats_ms": self.beats_ms,
                "beats_intensity": int_str,
                "override_zones": self._override_zones,
            }
            if self._tempo_grid_times is not None:
                data["tempo_grid_sec"] = self._tempo_grid_times.tolist()
                data["tempo_bpm"] = self._tempo_bpm
                data["tempo_phase"] = self._tempo_phase
                data["tempo_time_sig"] = self.tempo_timesig_var.get()
            # Save keyframe edits
            if self._kf_moved:
                data["kf_moved"] = {str(k): v for k, v in self._kf_moved.items()}
            if self._kf_deleted:
                data["kf_deleted"] = list(self._kf_deleted)
            if self._kf_independent:
                data["kf_independent"] = self._kf_independent
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            # Remember as last project
            cfg = _load_config()
            cfg["last_project"] = path
            _save_config(cfg)
            self.status.set(f"Saved {len(self.beats_ms)} beats to {os.path.basename(path)}")
        except OSError as e:
            messagebox.showerror("Error", f"Could not save beats:\n{e}")

    def _load_beats(self):
        """Load beats from a JSON file via dialog."""
        path = filedialog.askopenfilename(
            title="Load Project",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.out_dir.get() or None)
        if not path:
            return
        self._load_project_file(path)

    def _load_project_file(self, path):
        """Load a project from the given JSON file path."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            beats = data.get("beats_ms")
            if not isinstance(beats, list):
                messagebox.showerror("Error", "Invalid project file: missing 'beats_ms' array.")
                return
            # Remember as last project
            cfg = _load_config()
            cfg["last_project"] = path
            _save_config(cfg)
            # Restore output dir
            out = data.get("output_dir", "")
            if out:
                self.out_dir.set(out)
            # Store project data to restore after media loads
            inp = data.get("input_file", "")
            if inp and os.path.isfile(inp):
                self._pending_project = data
                self._set_file(inp)  # will clear beats, then _apply_pending_project restores them
            else:
                # No media file — just restore project data directly
                self._apply_project_data(data)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            messagebox.showerror("Error", f"Could not load project:\n{e}")

    def _apply_project_data(self, data):
        """Apply parsed project data (beats, zones, tempo, keyframes) to the app state."""
        beats = data.get("beats_ms", [])
        self._push_undo()
        self._clear_beat_hover()
        self.beats_ms = sorted(int(b) for b in beats)
        # Load intensities
        self.beats_intensity = {}
        raw_int = data.get("beats_intensity", {})
        for k, v in raw_int.items():
            try:
                self.beats_intensity[int(k)] = float(v)
            except (ValueError, TypeError):
                pass
        # Load zones
        self._override_zones = data.get("override_zones", [])
        self._selected_zone_idx = None
        # Load tempo grid
        tg = data.get("tempo_grid_sec")
        if tg:
            self._tempo_grid_times = np.array(tg, dtype=float)
            self._tempo_bpm = data.get("tempo_bpm")
            self._tempo_phase = data.get("tempo_phase", 0.0)
            ts = data.get("tempo_time_sig", "4/4")
            self.tempo_timesig_var.set(ts)
            self._tempo_time_sig = self._parse_time_sig()
            if self._tempo_bpm:
                self.tempo_bpm_var.set(f"{self._tempo_bpm:.1f}")
            self.tempo_phase_var.set(f"{self._tempo_phase:.3f}")
            if hasattr(self, '_spacing_mode_combo'):
                vals = list(self._spacing_mode_combo.cget("values"))
                if "Tempo" not in vals:
                    vals.append("Tempo")
                    self._spacing_mode_combo.config(values=vals)
        else:
            self._tempo_grid_times = None
            self._tempo_sub_times = None
            self._tempo_bpm = None
            if hasattr(self, '_spacing_mode_combo'):
                self._spacing_mode_combo.config(values=["Time"])
                if self.multi_beat_spacing_mode.get() == "Tempo":
                    self.multi_beat_spacing_mode.set("Time")
        # Load keyframe edits
        self._kf_moved = {}
        raw_moved = data.get("kf_moved", {})
        for k, v in raw_moved.items():
            try:
                self._kf_moved[int(k)] = v
            except (ValueError, TypeError):
                pass
        self._kf_deleted = set(data.get("kf_deleted", []))
        self._kf_independent = data.get("kf_independent", [])
        self._selected_kf.clear()
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self._draw_zones_lod()
        self._draw_wave_lod()
        self.canvas.draw_idle()
        n_zones = len(self._override_zones)
        zone_str = f", {n_zones} zones" if n_zones else ""
        tempo_str = f", {self._tempo_bpm:.0f} BPM" if self._tempo_bpm else ""
        self.status.set(f"Loaded {len(self.beats_ms)} beats{zone_str}{tempo_str}")

    def _detect_tempo(self):
        """Detect musical tempo using librosa, fill BPM and phase fields."""
        if self.y is None or self.sr is None:
            messagebox.showwarning("No audio", "Load an audio file first.")
            return
        self.status.set("Detecting tempo…")
        self.root.update_idletasks()
        try:
            # Analyze only the marker range
            in_sec, out_sec = self._get_marker_bounds()
            s0 = max(0, int(in_sec * self.sr))
            s1 = min(len(self.y), int(out_sec * self.sr))
            y_range = self.y[s0:s1] if s1 > s0 else self.y
            tempo, beat_frames = librosa.beat.beat_track(
                y=y_range, sr=self.sr, units="frames")
            if hasattr(tempo, "__len__"):
                tempo = float(tempo[0])
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sr) + in_sec
            self._tempo_bpm = float(tempo)
            self.tempo_bpm_var.set(f"{tempo:.1f}")
            # Use first detected beat as phase
            if len(beat_times) > 0:
                self._tempo_phase = float(beat_times[0])
                self.tempo_phase_var.set(f"{self._tempo_phase:.3f}")
            # Auto-generate the grid
            self._build_tempo_grid()
            self._draw_wave_lod()
            self.canvas.draw_idle()
            n = len(self._tempo_grid_times) if self._tempo_grid_times is not None else 0
            self.status.set(f"Detected tempo: {tempo:.1f} BPM, {n} beats")
        except Exception as e:
            messagebox.showerror("Error", f"Tempo detection failed:\n{e}")
            self.status.set("Tempo detection failed.")

    def _generate_tempo_grid(self):
        """Generate beat grid from the BPM, phase, and time signature fields."""
        self._auto_stop_tap_recording()
        if self.y is None or self.sr is None:
            messagebox.showwarning("No audio", "Load an audio file first.")
            return
        try:
            bpm = float(self.tempo_bpm_var.get())
            if bpm <= 0:
                raise ValueError("BPM must be positive")
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid BPM", "Enter a valid BPM number.")
            return
        try:
            phase = float(self.tempo_phase_var.get())
        except (ValueError, TypeError):
            phase = 0.0
        self._tempo_bpm = bpm
        self._tempo_phase = max(0.0, phase)
        self._build_tempo_grid()
        self._draw_wave_lod()
        self.canvas.draw_idle()
        n = len(self._tempo_grid_times) if self._tempo_grid_times is not None else 0
        self.status.set(f"Generated grid: {bpm:.1f} BPM, {n} beats")

    def _build_tempo_grid(self):
        """Build evenly-spaced beat grid from BPM, phase, and duration."""
        if self._tempo_bpm is None or self._tempo_bpm <= 0:
            self._tempo_grid_times = None
            self._tempo_sub_times = None
            # Remove "Tempo" spacing mode
            if hasattr(self, '_spacing_mode_combo'):
                self._spacing_mode_combo.config(values=["Time"])
                if self.multi_beat_spacing_mode.get() == "Tempo":
                    self.multi_beat_spacing_mode.set("Time")
            return
        interval = 60.0 / self._tempo_bpm  # seconds per beat
        duration = self.duration if self.duration > 0 else 300.0
        phase = self._tempo_phase
        # Restrict grid to marker range
        in_sec, out_sec = self._get_marker_bounds()
        grid_start = in_sec
        grid_end = out_sec if out_sec > 0 else duration
        # Parse time signature for measure grouping
        self._tempo_time_sig = self._parse_time_sig()
        # Generate beats from phase within marker range
        times = []
        t = phase
        while t <= grid_end:
            if t >= grid_start:
                times.append(t)
            t += interval
        # Also go backwards from phase
        if phase > interval:
            t = phase - interval
            while t >= grid_start:
                times.append(t)
                t -= interval
        times.sort()
        self._tempo_grid_times = np.array(times, dtype=float)
        # Generate sub-beats between main beats
        subdiv = max(1, self.tempo_subdiv_var.get())
        if subdiv > 1 and len(times) >= 2:
            sub_times = []
            main_set = set(round(t, 6) for t in times)
            sub_interval = interval / subdiv
            for main_t in times:
                for s in range(1, subdiv):
                    st = main_t + s * sub_interval
                    if grid_start <= st <= grid_end and round(st, 6) not in main_set:
                        sub_times.append(st)
            sub_times.sort()
            self._tempo_sub_times = np.array(sub_times, dtype=float)
        else:
            self._tempo_sub_times = None
        # Enable "Tempo" spacing mode now that grid exists
        if hasattr(self, '_spacing_mode_combo'):
            vals = list(self._spacing_mode_combo.cget("values"))
            if "Tempo" not in vals:
                vals.append("Tempo")
                self._spacing_mode_combo.config(values=vals)

    def _parse_time_sig(self):
        """Parse time signature string like '4/4' → beats per measure (numerator)."""
        sig = self.tempo_timesig_var.get()
        try:
            num = int(sig.split("/")[0])
            return max(1, num)
        except (ValueError, IndexError):
            return 4

    def _toggle_tap_recording(self):
        """Toggle tap recording on/off."""
        if self._tap_recording:
            self._stop_tap_recording()
        else:
            self._start_tap_recording()

    def _start_tap_recording(self):
        """Start recording taps. Return key or Tap button registers a tap."""
        self._tap_recording = True
        self._tap_times = []
        self._tap_btn.config(text="Stop Recording")
        self._tap_click_btn.config(state="normal")
        self._tap_rec_indicator.pack(side="left", padx=(0, 4), before=self._tap_label)
        self._tap_label.config(text="Press Return or click Tap…")
        self.root.bind("<Return>", self._on_tap_key)

    def _stop_tap_recording(self):
        """Stop recording taps."""
        self._tap_recording = False
        self._tap_btn.config(text="Record Taps")
        self._tap_click_btn.config(state="disabled")
        self._tap_rec_indicator.pack_forget()
        self.root.unbind("<Return>")

    def _auto_stop_tap_recording(self):
        """Stop tap recording if active. Called on non-exempt actions."""
        if self._tap_recording:
            self._stop_tap_recording()

    def _on_tap_key(self, event=None):
        """Handle a tap (Return key or button)."""
        if not self._tap_recording:
            return
        self._register_tap()

    def _register_tap(self):
        """Record a tap and calculate BPM from tap intervals."""
        now = _time.time()
        self._tap_times.append(now)
        n = len(self._tap_times)
        if n < 2:
            self._tap_label.config(text=f"1 tap… keep pressing Return")
            return
        intervals = [self._tap_times[i] - self._tap_times[i - 1]
                     for i in range(1, n)]
        avg_interval = sum(intervals) / len(intervals)
        bpm = 60.0 / avg_interval
        self._tempo_bpm = bpm
        self.tempo_bpm_var.set(f"{bpm:.1f}")
        # Set phase from playhead position at first tap
        if self._playing and self._playhead_pos > 0:
            first_tap_playhead = self._playhead_pos - (now - self._tap_times[0])
            self._tempo_phase = max(0.0, first_tap_playhead)
            self.tempo_phase_var.set(f"{self._tempo_phase:.3f}")
        self._tap_label.config(text=f"{n} taps → {bpm:.1f} BPM")

    def _clear_tempo(self):
        """Remove the musical beat grid but keep BPM/phase fields."""
        self._tempo_grid_times = None
        self._tempo_sub_times = None
        # Remove "Tempo" spacing mode
        if hasattr(self, '_spacing_mode_combo'):
            self._spacing_mode_combo.config(values=["Time"])
            if self.multi_beat_spacing_mode.get() == "Tempo":
                self.multi_beat_spacing_mode.set("Time")
        self._draw_wave_lod()
        self.canvas.draw_idle()
        self.status.set("Tempo grid cleared.")

    def _draw_tempo_grid_on_ax(self, ax, xlim=None):
        """Draw musical beat grid lines — three levels:
        downbeat (bright blue), normal beat (mid blue), sub-beat (dark blue)."""
        if self._tempo_grid_times is None or len(self._tempo_grid_times) == 0:
            return
        if xlim is None:
            xlim = ax.get_xlim()
        grid = self._tempo_grid_times
        beats_per_measure = self._tempo_time_sig if self._tempo_time_sig > 0 else 4

        # Find the index of the phase beat (beat 1) in the grid
        phase = self._tempo_phase
        if len(grid) > 0:
            phase_idx = int(np.argmin(np.abs(grid - phase)))
        else:
            phase_idx = 0

        i0 = max(0, int(np.searchsorted(grid, xlim[0])) - 1)
        i1 = min(len(grid), int(np.searchsorted(grid, xlim[1])) + 1)
        indices = np.arange(i0, i1)

        # Vectorized downbeat/normal separation
        beat_mod = (indices - phase_idx) % beats_per_measure
        db_mask = beat_mod == 0
        downbeat_times = grid[indices[db_mask]]
        normal_times = grid[indices[~db_mask]]

        # Sub-beats via searchsorted
        sub_times_arr = None
        if self._tempo_sub_times is not None and len(self._tempo_sub_times) > 0:
            sg = self._tempo_sub_times
            si0 = max(0, int(np.searchsorted(sg, xlim[0])) - 1)
            si1 = min(len(sg), int(np.searchsorted(sg, xlim[1])) + 1)
            sub_times_arr = sg[si0:si1]

        # Thin to pixel budget (numpy version)
        width_px = self._get_axes_width_px(ax)
        max_items = max(10, width_px * 30 // 100)
        def _thin(arr):
            if arr is None or len(arr) <= max_items:
                return arr
            step = len(arr) / max_items
            return arr[(np.arange(max_items) * step).astype(int)]
        sub_times_arr = _thin(sub_times_arr)
        normal_times = _thin(normal_times)
        downbeat_times = _thin(downbeat_times)

        # Build a single LineCollection with per-segment styling
        all_segs = []
        all_colors = []
        all_lw = []
        all_alpha = []
        if sub_times_arr is not None and len(sub_times_arr) > 0:
            for t in sub_times_arr:
                all_segs.append([(t, -1.0), (t, 1.0)])
                all_colors.append("#3d59a1")
                all_lw.append(0.6)
                all_alpha.append(0.55)
        if len(normal_times) > 0:
            for t in normal_times:
                all_segs.append([(t, -1.0), (t, 1.0)])
                all_colors.append("#5c7bd9")
                all_lw.append(0.9)
                all_alpha.append(0.65)
        if len(downbeat_times) > 0:
            for t in downbeat_times:
                all_segs.append([(t, -1.0), (t, 1.0)])
                all_colors.append("#89b4fa")
                all_lw.append(1.5)
                all_alpha.append(0.8)
        if all_segs:
            lc = LineCollection(all_segs, colors=all_colors,
                                linewidths=all_lw, zorder=0)
            lc.set_alpha(all_alpha)
            ax.add_collection(lc)
            self._wave_artists.append(lc)

    def _regenerate_funscript(self):
        """Regenerate funscript from current beats and settings.

        If override zones exist, beats inside a zone use that zone's settings;
        beats outside any zone use the global settings.
        """
        if not self.beats_ms:
            self._last_funscript = None
            # Still apply independent keyframes even without beats
            if self._kf_independent:
                self._last_funscript = {"version": "1.0", "actions": [], "range": 100}
                self._apply_kf_edits()
            return

        # Global settings as defaults
        global_settings = {
            "speed": self.speed.get(),
            "pos_min": self.pos_min.get(),
            "pos_max": self.pos_max.get(),
            "mode": self.mode_var.get(),
            "alt_mode": self.alt_mode_var.get(),
            "noise_amount": self.noise_amount.get(),
            "peak_start_at_beat": self.peak_start_at_beat.get(),
            "reverse_curve": self.reverse_curve.get(),
        }

        if not self._override_zones:
            # No zones — simple path
            min_p = int(round(global_settings["pos_min"] * 100))
            max_p = int(round(global_settings["pos_max"] * 100))
            if min_p >= max_p:
                min_p, max_p = 0, 100
            self._last_funscript = generate_funscript(
                self.beats_ms,
                speed=global_settings["speed"],
                min_pos=min_p, max_pos=max_p,
                mode=global_settings["mode"],
                noise_amount=global_settings["noise_amount"],
                intensities=self.beats_intensity,
                alt_mode=global_settings["alt_mode"],
                start_at_beat=global_settings["peak_start_at_beat"],
                reverse_curve=global_settings["reverse_curve"],
            )
            self._apply_kf_edits()
            return

        # Sort zones by start time
        sorted_zones = sorted(self._override_zones, key=lambda z: z["start_ms"])

        # For each beat, determine which settings to use.
        # Zone settings only override keys listed in _overrides; the rest
        # fall through to global settings.
        _merged_cache = {}  # zone id -> merged dict

        def _settings_for_beat(beat_ms):
            for z in sorted_zones:
                if z["start_ms"] <= beat_ms <= z["end_ms"]:
                    zid = id(z)
                    if zid not in _merged_cache:
                        zs = z["settings"]
                        active = set(zs.get("_overrides", list(zs.keys())))
                        active.discard("_overrides")
                        merged = dict(global_settings)
                        for k in active:
                            if k in zs:
                                merged[k] = zs[k]
                        _merged_cache[zid] = merged
                    return _merged_cache[zid]
            return global_settings

        # Group consecutive beats with the same settings object identity
        groups = []
        current_settings = None
        current_beats = []
        for b_ms in self.beats_ms:
            s = _settings_for_beat(b_ms)
            s_key = id(s)
            if s_key != id(current_settings):
                if current_beats:
                    groups.append((current_settings, list(current_beats)))
                current_settings = s
                current_beats = [b_ms]
            else:
                current_beats.append(b_ms)
        if current_beats:
            groups.append((current_settings, list(current_beats)))

        # Generate funscript for each group and merge
        all_actions = []
        for settings, beats in groups:
            min_p = int(round(settings["pos_min"] * 100))
            max_p = int(round(settings["pos_max"] * 100))
            if min_p >= max_p:
                min_p, max_p = 0, 100
            fs = generate_funscript(
                beats,
                speed=settings["speed"],
                min_pos=min_p, max_pos=max_p,
                mode=settings["mode"],
                noise_amount=settings["noise_amount"],
                intensities=self.beats_intensity,
                alt_mode=settings.get("alt_mode", ALT_FORCE),
                start_at_beat=settings.get("peak_start_at_beat", False),
                reverse_curve=settings.get("reverse_curve", False),
            )
            all_actions.extend(fs.get("actions", []))

        # Sort by time and deduplicate
        all_actions.sort(key=lambda a: a["at"])
        all_actions = _dedup_actions(all_actions)
        self._last_funscript = {"version": "1.0", "actions": all_actions, "range": 100}
        self._apply_kf_edits()

    def _apply_kf_edits(self):
        """Layer keyframe edits (moved / deleted / independent) on top of
        the base actions produced by _regenerate_funscript.  Rebuilds
        _kf_types and _kf_reverse mappings used for drawing & interaction."""
        if self._last_funscript is None:
            self._kf_types.clear()
            self._kf_reverse.clear()
            return

        actions = self._last_funscript["actions"]
        new_actions = []
        self._kf_types = {}
        self._kf_reverse = {}

        # Build set of base at_ms values for quick lookup
        base_at_set = {a["at"] for a in actions}

        # Clean up stale edits: remove moved/deleted entries whose
        # original at_ms no longer exists in the base
        stale_moved = [k for k in self._kf_moved if k not in base_at_set]
        for k in stale_moved:
            self._kf_moved.pop(k, None)
        self._kf_deleted = {k for k in self._kf_deleted if k in base_at_set}

        rev = self.reverse_curve.get()
        for a in actions:
            at = a["at"]
            if at in self._kf_deleted:
                continue  # skip deleted keyframes
            if at in self._kf_moved:
                m = self._kf_moved[at]
                pos = (100 - m["pos"]) if rev else m["pos"]
                new_actions.append({"at": m["at"], "pos": pos})
                self._kf_types[m["at"]] = "modified"
                self._kf_reverse[m["at"]] = at
            else:
                new_actions.append(dict(a))

        # Add independent keyframes
        for kf in self._kf_independent:
            pos = (100 - kf["pos"]) if rev else kf["pos"]
            new_actions.append({"at": kf["at"], "pos": pos})
            self._kf_types[kf["at"]] = "independent"

        new_actions.sort(key=lambda a: a["at"])
        # Skip dedup while dragging to prevent keyframes from disappearing
        if self._dragging in ("kf", "multi_kf"):
            # Only dedup same-time (keep last), skip same-pos run collapse
            seen = {}
            for a in new_actions:
                seen[a["at"]] = a
            self._last_funscript["actions"] = sorted(seen.values(), key=lambda a: a["at"])
        else:
            self._last_funscript["actions"] = _dedup_actions(new_actions)

    # ── Undo / Redo ──

    def _snapshot_state(self):
        """Return a deep copy of the current editable state."""
        return {
            "beats_ms": list(self.beats_ms) if self.beats_ms else None,
            "beats_intensity": dict(self.beats_intensity),
            "override_zones": copy.deepcopy(self._override_zones),
            "selected_beats": set(self._selected_beats),
            "kf_moved": copy.deepcopy(self._kf_moved),
            "kf_deleted": set(self._kf_deleted),
            "kf_independent": copy.deepcopy(self._kf_independent),
            "selected_kf": set(self._selected_kf),
        }

    def _restore_state(self, snapshot):
        """Apply a previously saved state snapshot."""
        self._clear_beat_hover()
        self._clear_kf_hover()
        self.beats_ms = list(snapshot["beats_ms"]) if snapshot["beats_ms"] is not None else None
        self.beats_intensity = dict(snapshot["beats_intensity"])
        self._override_zones = copy.deepcopy(snapshot["override_zones"])
        self._selected_beats = set(snapshot["selected_beats"])
        self._kf_moved = copy.deepcopy(snapshot.get("kf_moved", {}))
        self._kf_deleted = set(snapshot.get("kf_deleted", set()))
        self._kf_independent = copy.deepcopy(snapshot.get("kf_independent", []))
        self._selected_kf = set(snapshot.get("selected_kf", set()))
        self._selected_zone_idx = None
        self._regenerate_funscript()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self._draw_zones_lod()
        self.canvas.draw_idle()

    def _push_undo(self):
        """Capture current state and push to undo stack, clear redo stack."""
        self._auto_stop_tap_recording()
        self._undo_stack.append(self._snapshot_state())
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self):
        """Pop from undo stack, push current to redo, restore state."""
        if not self._undo_stack:
            self.status.set("Nothing to undo")
            return
        self._redo_stack.append(self._snapshot_state())
        snapshot = self._undo_stack.pop()
        self._restore_state(snapshot)
        self.status.set("Undo")

    def _redo(self):
        """Pop from redo stack, push current to undo, restore state."""
        if not self._redo_stack:
            self.status.set("Nothing to redo")
            return
        self._undo_stack.append(self._snapshot_state())
        snapshot = self._redo_stack.pop()
        self._restore_state(snapshot)
        self.status.set("Redo")

    # ── Keyframe helpers ──

    def _find_nearest_kf(self, x_sec, y_pos):
        """Find nearest keyframe action index near (x_sec, y_pos) in ax_fun.
        Returns index into _last_funscript['actions'] or None."""
        if self._last_funscript is None:
            return None
        actions = self._last_funscript["actions"]
        if not actions:
            return None
        xlim = self.ax_fun.get_xlim()
        ylim = self.ax_fun.get_ylim()
        x_tol = (xlim[1] - xlim[0]) * 0.015
        y_tol = (ylim[1] - ylim[0]) * 0.06
        if x_tol <= 0 or y_tol <= 0:
            return None
        import bisect
        target_ms = x_sec * 1000
        # Binary search to find nearby candidates
        at_values = [a["at"] for a in actions]
        idx = bisect.bisect_left(at_values, target_ms)
        best_idx = None
        best_dist = float("inf")
        for i in range(max(0, idx - 3), min(len(actions), idx + 4)):
            dx = abs(actions[i]["at"] / 1000.0 - x_sec) / x_tol
            dy = abs(actions[i]["pos"] - y_pos) / y_tol
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < best_dist and dist < 1.5:
                best_dist = dist
                best_idx = i
        return best_idx

    def _kf_color_for(self, at_ms):
        """Return the display color for a keyframe by its type."""
        kf_type = self._kf_types.get(at_ms)
        # Check if fully locked
        lock_t = self.kf_lock_time.get()
        lock_p = self.kf_lock_pos.get()
        t_locked = (lock_t == "Lock all"
                    or (lock_t == "Lock beat" and kf_type != "independent")
                    or (lock_t == "Lock independent" and kf_type == "independent")
                    or lock_t == "Lock order")
        p_locked = (lock_p == "Lock all"
                    or (lock_p == "Lock beat" and kf_type != "independent")
                    or (lock_p == "Lock independent" and kf_type == "independent"))
        if t_locked and p_locked:
            return "#565f89"
        if kf_type == "modified":
            return KF_COLOR_MODIFIED
        elif kf_type == "independent":
            return KF_COLOR_INDEPENDENT
        return KF_COLOR_ORIGINAL

    def _update_kf_hover(self, x_sec, y_pos):
        """Update keyframe hover highlight on ax_fun. Skipped during drag."""
        if self._dragging in ("kf", "multi_kf"):
            return
        idx = self._find_nearest_kf(x_sec, y_pos)
        if idx == self._hover_kf_idx:
            return
        self._clear_kf_hover()
        self._hover_kf_idx = idx
        if idx is not None:
            a = self._last_funscript["actions"][idx]
            c = self._kf_color_for(a["at"])
            dot, = self.ax_fun.plot(
                a["at"] / 1000.0, a["pos"], marker="o", markersize=8,
                markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=2, alpha=0.85, zorder=15, linewidth=0)
            self._hover_kf_artist = dot
            self.canvas.draw_idle()

    def _clear_kf_hover(self):
        """Remove keyframe hover highlight."""
        if self._hover_kf_artist is not None:
            try:
                self._hover_kf_artist.remove()
            except Exception:
                pass
            self._hover_kf_artist = None
        self._hover_kf_idx = None

    def _add_independent_kf(self, x_sec, y_pos):
        """Add an independent keyframe at (x_sec, y_pos)."""
        self._push_undo()
        at_ms = int(round(x_sec * 1000))
        pos = int(round(max(0, min(100, y_pos))))
        at_ms = max(0, at_ms)
        if self.duration > 0:
            at_ms = min(int(self.duration * 1000), at_ms)
        # Store in un-reversed space
        store_p = (100 - pos) if self.reverse_curve.get() else pos
        self._kf_independent.append({"at": at_ms, "pos": store_p})
        self._apply_kf_edits_and_redraw()
        self.status.set(f"Added keyframe at {_format_time_ms(x_sec)}, pos {pos}")

    def _delete_kf_at(self, idx):
        """Delete keyframe at given index — only independent keyframes can be removed."""
        if self._last_funscript is None:
            return
        actions = self._last_funscript["actions"]
        if idx < 0 or idx >= len(actions):
            return
        at_ms = actions[idx]["at"]
        kf_type = self._kf_types.get(at_ms)
        if kf_type != "independent":
            self.status.set("Beat-generated keyframes cannot be removed")
            return
        self._push_undo()
        self._kf_independent = [
            k for k in self._kf_independent if k["at"] != at_ms]
        self._selected_kf.discard(at_ms)
        self._clear_kf_hover()
        self._apply_kf_edits_and_redraw()
        self.status.set("Deleted keyframe")

    def _delete_selected_kf(self):
        """Delete selected keyframes — only independent ones are removed."""
        if not self._selected_kf or self._last_funscript is None:
            return
        # Filter to only independent keyframes
        to_delete = [ms for ms in self._selected_kf
                     if self._kf_types.get(ms) == "independent"]
        if not to_delete:
            self.status.set("Beat-generated keyframes cannot be removed")
            return
        self._push_undo()
        del_set = set(to_delete)
        self._kf_independent = [
            k for k in self._kf_independent if k["at"] not in del_set]
        count = len(to_delete)
        self._selected_kf -= del_set
        self._clear_kf_hover()
        self._apply_kf_edits_and_redraw()
        self.status.set(f"Deleted {count} independent keyframe(s)")

    def _reset_and_delete_selected_kf(self):
        """Right-click on selection: reset modified keyframes, delete independent ones."""
        if not self._selected_kf or self._last_funscript is None:
            return
        to_reset = [ms for ms in self._selected_kf
                    if self._kf_types.get(ms) == "modified"]
        to_delete = [ms for ms in self._selected_kf
                     if self._kf_types.get(ms) == "independent"]
        if not to_reset and not to_delete:
            self.status.set("Original beat keyframes cannot be removed or reset")
            return
        self._push_undo()
        # Reset modified
        for ms in to_reset:
            orig = self._kf_reverse.get(ms)
            if orig is not None:
                self._kf_moved.pop(orig, None)
            self._selected_kf.discard(ms)
        # Delete independent
        del_set = set(to_delete)
        self._kf_independent = [
            k for k in self._kf_independent if k["at"] not in del_set]
        self._selected_kf -= del_set
        self._clear_kf_hover()
        self._apply_kf_edits_and_redraw()
        parts = []
        if to_reset:
            parts.append(f"reset {len(to_reset)}")
        if to_delete:
            parts.append(f"deleted {len(to_delete)}")
        self.status.set(f"Keyframes: {', '.join(parts)}")

    def _reset_kf_to_original(self, idx):
        """Reset a modified keyframe back to its beat-generated position."""
        if self._last_funscript is None:
            return
        actions = self._last_funscript["actions"]
        if idx < 0 or idx >= len(actions):
            return
        at_ms = actions[idx]["at"]
        if self._kf_types.get(at_ms) != "modified":
            return  # only reset modified keyframes
        orig = self._kf_reverse.get(at_ms)
        if orig is None:
            return
        self._push_undo()
        self._kf_moved.pop(orig, None)
        self._selected_kf.discard(at_ms)
        self._clear_kf_hover()
        self._apply_kf_edits_and_redraw()
        self.status.set("Reset keyframe to original position")

    def _move_kf(self, idx, new_sec, new_pos):
        """Move keyframe at idx to new time/position. Returns new index."""
        if self._last_funscript is None:
            return None
        actions = self._last_funscript["actions"]
        if idx < 0 or idx >= len(actions):
            return None
        at_ms = actions[idx]["at"]
        kf_type = self._kf_types.get(at_ms)
        # Check lock time
        lock_t = self.kf_lock_time.get()
        time_locked = (lock_t == "Lock all"
                       or (lock_t == "Lock beat" and kf_type != "independent")
                       or (lock_t == "Lock independent" and kf_type == "independent"))
        if time_locked:
            new_at = at_ms
        else:
            new_at = int(round(new_sec * 1000))
            new_at = max(0, new_at)
            if self.duration > 0:
                new_at = min(int(self.duration * 1000), new_at)
            # Lock order: clamp so keyframe can't move past neighbors
            if lock_t == "Lock order":
                if idx > 0:
                    new_at = max(new_at, actions[idx - 1]["at"] + 1)
                if idx < len(actions) - 1:
                    new_at = min(new_at, actions[idx + 1]["at"] - 1)
        # Check lock position
        lock_p = self.kf_lock_pos.get()
        pos_locked = (lock_p == "Lock all"
                      or (lock_p == "Lock beat" and kf_type != "independent")
                      or (lock_p == "Lock independent" and kf_type == "independent"))
        if pos_locked:
            new_p = actions[idx]["pos"]
        else:
            new_p = int(round(max(0, min(100, new_pos))))
        # Store in un-reversed space so _apply_kf_edits can apply reverse
        store_p = (100 - new_p) if self.reverse_curve.get() else new_p
        if kf_type == "independent":
            # Update in _kf_independent
            for kf in self._kf_independent:
                if kf["at"] == at_ms:
                    kf["at"] = new_at
                    kf["pos"] = store_p
                    break
        elif kf_type == "modified":
            # Already modified — update the moved entry
            orig = self._kf_reverse.get(at_ms)
            if orig is not None:
                self._kf_moved[orig] = {"at": new_at, "pos": store_p}
        else:
            # Original — create a moved entry
            self._kf_moved[at_ms] = {"at": new_at, "pos": store_p}
        # Update selection tracking
        if at_ms in self._selected_kf:
            self._selected_kf.discard(at_ms)
            self._selected_kf.add(new_at)
        # Rebuild actions
        self._apply_kf_edits_and_redraw()
        # Find new index
        actions = self._last_funscript["actions"]
        for i, a in enumerate(actions):
            if a["at"] == new_at:
                return i
        return None

    def _copy_kf(self):
        """Copy selected keyframes as relative offsets."""
        if not self._selected_kf or self._last_funscript is None:
            return
        actions = self._last_funscript["actions"]
        sel_actions = [a for a in actions if a["at"] in self._selected_kf]
        if not sel_actions:
            return
        sel_actions.sort(key=lambda a: a["at"])
        first_at = sel_actions[0]["at"]
        self._clipboard_kf = [
            (a["at"] - first_at, a["pos"]) for a in sel_actions]
        self.status.set(f"Copied {len(self._clipboard_kf)} keyframes")

    def _paste_kf(self):
        """Paste keyframes at playhead position as independent keyframes."""
        if not self._clipboard_kf:
            return
        self._push_undo()
        base_ms = int(self._playhead_pos * 1000)
        new_sel = set()
        rev = self.reverse_curve.get()
        for offset_ms, pos in self._clipboard_kf:
            at_ms = base_ms + offset_ms
            at_ms = max(0, at_ms)
            if self.duration > 0:
                at_ms = min(int(self.duration * 1000), at_ms)
            store_p = (100 - pos) if rev else pos
            self._kf_independent.append({"at": at_ms, "pos": store_p})
            new_sel.add(at_ms)
        self._selected_kf = new_sel
        self._apply_kf_edits_and_redraw()
        self.status.set(f"Pasted {len(self._clipboard_kf)} keyframes at playhead")

    def _apply_kf_edits_and_redraw(self):
        """Re-apply keyframe edits on the current base and redraw."""
        # Re-run full regeneration so base is fresh, then edits layered on top
        self._regenerate_funscript()
        self._draw_fun_lod()
        self.canvas.draw_idle()

    def _clear_kf_selection(self):
        """Clear keyframe selection and redraw."""
        self._selected_kf.clear()
        self._draw_fun_lod()
        self.canvas.draw_idle()

    # ── Zone helpers ──

    def _zone_at(self, click_ms):
        """Return index of zone containing click_ms, or None."""
        for i, z in enumerate(self._override_zones):
            if z["start_ms"] <= click_ms <= z["end_ms"]:
                return i
        return None

    def _clamp_zone_no_overlap(self, start_ms, end_ms, exclude_idx=None):
        """Clamp start_ms/end_ms so the zone doesn't overlap any other zone.
        exclude_idx: skip this zone index (used when moving/resizing an existing zone).
        Returns (clamped_start, clamped_end)."""
        for i, z in enumerate(self._override_zones):
            if i == exclude_idx:
                continue
            zs, ze = z["start_ms"], z["end_ms"]
            # If our zone overlaps this one, shrink to fit
            if start_ms < ze and end_ms > zs:
                # Overlap detected – decide which side to clamp
                # If center of proposed zone is left of center of existing zone, clamp our right
                mid_new = (start_ms + end_ms) / 2
                mid_existing = (zs + ze) / 2
                if mid_new < mid_existing:
                    end_ms = min(end_ms, zs)
                else:
                    start_ms = max(start_ms, ze)
        return int(start_ms), int(end_ms)

    def _clamp_zone_move_no_overlap(self, start_ms, end_ms, exclude_idx):
        """Clamp a zone being moved so it doesn't overlap any other zone.
        Keeps the zone width constant. Returns (clamped_start, clamped_end)."""
        width = end_ms - start_ms
        for i, z in enumerate(self._override_zones):
            if i == exclude_idx:
                continue
            zs, ze = z["start_ms"], z["end_ms"]
            if start_ms < ze and end_ms > zs:
                # Overlap – push in the direction with less penetration
                push_right = ze - start_ms
                push_left = end_ms - zs
                if push_right <= push_left:
                    start_ms = ze
                    end_ms = start_ms + width
                else:
                    end_ms = zs
                    start_ms = end_ms - width
        return int(start_ms), int(end_ms)

    def _draw_zones_lod(self):
        """Draw override zone rectangles on the zones axes."""
        ax = self.ax_zones
        for a in self._zone_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._zone_artists = []

        if not self._override_zones:
            # Show hint text when no zones exist
            hint = ax.text(
                0.5, 0.5,
                "Click to create override zones",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6, color=TEXT_COLOR, alpha=0.35, style="italic")
            self._zone_artists.append(hint)
            return

        from matplotlib.patches import Rectangle

        xlim = ax.get_xlim()
        for i, zone in enumerate(self._override_zones):
            start_sec = zone["start_ms"] / 1000.0
            end_sec = zone["end_ms"] / 1000.0
            # Skip if completely outside view
            if end_sec < xlim[0] or start_sec > xlim[1]:
                continue
            width = end_sec - start_sec
            is_selected = (i == self._selected_zone_idx)
            ec = "white" if is_selected else ZONE_COLOR
            lw = 1.5 if is_selected else 0.5
            rect = Rectangle((start_sec, 0.05), width, 0.9,
                              facecolor=ZONE_COLOR, alpha=0.6,
                              edgecolor=ec, linewidth=lw, zorder=5)
            ax.add_patch(rect)
            self._zone_artists.append(rect)

            # Resize handle indicators on edges
            handle_w = max(0.002 * (xlim[1] - xlim[0]), 0.01)
            for edge_x in (start_sec, end_sec):
                handle = Rectangle((edge_x - handle_w / 2, 0.0), handle_w, 1.0,
                                    facecolor="white", alpha=0.4 if is_selected else 0.2,
                                    edgecolor="none", zorder=7)
                ax.add_patch(handle)
                self._zone_artists.append(handle)

            # Label — use zone name if set, otherwise show index
            width_px = self._get_axes_width_px(ax)
            span = xlim[1] - xlim[0]
            zone_px = (width / span) * width_px if span > 0 else 0
            if zone_px > 30:
                name = zone.get("name", "Overriding Zone")
                label = name if name else "Overriding Zone"
                txt = ax.text(start_sec + width / 2, 0.5, label,
                              ha="center", va="center", fontsize=6,
                              color="white", alpha=0.8, zorder=8,
                              clip_box=rect.get_bbox(),
                              clip_on=True)
                # Clip text to the zone rectangle bounds
                txt.set_clip_path(rect)
                self._zone_artists.append(txt)

    def _edit_zone(self, idx):
        """Open a modal dialog to edit zone settings.
        Each setting has an 'enabled' checkbox — only enabled settings override
        the global values.  Disabled settings fall through to global."""
        if idx < 0 or idx >= len(self._override_zones):
            return
        zone = self._override_zones[idx]
        settings = zone["settings"]
        # Which keys are actually overridden (default: all, for backward compat)
        overrides = set(settings.get("_overrides", list(settings.keys())))
        overrides.discard("_overrides")

        # Snapshot for cancel-revert
        settings_backup = copy.deepcopy(settings)
        name_backup = zone.get("name", "Overriding Zone")

        dlg = tk.Toplevel(self.root)
        dlg.title(f"Overriding Zone  {_format_time(zone['start_ms'] / 1000)} – {_format_time(zone['end_ms'] / 1000)}")
        dlg.geometry("430x620")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        # Zone name
        name_frame = ttk.Frame(dlg)
        name_frame.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(name_frame, text="Name:").pack(side="left", padx=(0, 4))
        v_name = tk.StringVar(value=zone.get("name", "Overriding Zone"))
        ttk.Entry(name_frame, textvariable=v_name, width=40).pack(side="left", fill="x", expand=True)

        # Shared helpers for enable/disable rows
        _row_widgets = {}  # key -> list of widgets

        def _set_row_state(row, enabled):
            state = "normal" if enabled else "disabled"
            for w in _row_widgets.get(row, []):
                try:
                    w.config(state=state)
                except (tk.TclError, AttributeError):
                    pass

        def _make_enable_cb(parent, row, en_var, key=None):
            rk = key if key is not None else row
            cb = ttk.Checkbutton(parent, variable=en_var,
                                 command=lambda: _set_row_state(rk, en_var.get()))
            cb.grid(row=row, column=0, padx=(4, 0), pady=3)
            return cb

        def _add_zone_slider(parent, row, label, var, default_val, en_var, key=None):
            rk = key if key is not None else row
            _make_enable_cb(parent, row, en_var, key=rk)
            lbl = ttk.Label(parent, text=label)
            lbl.grid(row=row, column=1, sticky="w", padx=4, pady=3)
            scl = ttk.Scale(parent, from_=0.0, to=1.0, variable=var, length=140)
            scl.grid(row=row, column=2, padx=4, pady=3)
            val_lbl = ttk.Label(parent, text=f"{var.get():.2f}", width=5)
            val_lbl.grid(row=row, column=3, padx=(0, 2), pady=3)
            rst = ttk.Button(parent, text="↺", width=2,
                       command=lambda v=var, d=default_val: v.set(d))
            rst.grid(row=row, column=4, padx=(0, 4), pady=3)
            var.trace_add("write",
                          lambda *_, v=var, l=val_lbl: l.config(text=f"{v.get():.2f}"))
            _row_widgets[rk] = [lbl, scl, val_lbl, rst]

        # ════════════════════════════════════════════════
        # Beat Detection Override
        # ════════════════════════════════════════════════
        frame_bd = ttk.LabelFrame(dlg, text="Beat Detection Override")
        frame_bd.pack(fill="x", padx=8, pady=(4, 4))

        tk.Label(frame_bd,
                 text="Checked settings override global beat detection when re-analyzing.",
                 fg="gray", font=("Segoe UI", 7)).grid(
            row=0, column=0, columnspan=5, padx=8, pady=(4, 2), sticky="w")

        # Beat detection variables
        v_bd_threshold = tk.DoubleVar(value=settings.get("bd_threshold", self.threshold.get()))
        v_bd_gap = tk.DoubleVar(value=settings.get("bd_min_beat_gap", self.min_beat_gap.get()))
        v_bd_bass = tk.BooleanVar(value=settings.get("bd_bass_only", self.bass_only.get()))

        en_bd_threshold = tk.BooleanVar(value="bd_threshold" in overrides)
        en_bd_gap = tk.BooleanVar(value="bd_min_beat_gap" in overrides)
        en_bd_bass = tk.BooleanVar(value="bd_bass_only" in overrides)

        _add_zone_slider(frame_bd, 1, "Beat Threshold", v_bd_threshold,
                         DEFAULTS["threshold"], en_bd_threshold, key="bd1")
        _add_zone_slider(frame_bd, 2, "Min Beat Gap", v_bd_gap,
                         DEFAULTS["min_beat_gap"], en_bd_gap, key="bd2")

        # Bass-only checkbox row
        _make_enable_cb(frame_bd, 3, en_bd_bass, key="bd3")
        bd_bass_frame = ttk.Frame(frame_bd)
        bd_bass_frame.grid(row=3, column=1, columnspan=3, sticky="w", padx=4, pady=3)
        bd_bass_cb = ttk.Checkbutton(bd_bass_frame, text="Bass only (≤ 200 Hz)",
                                     variable=v_bd_bass)
        bd_bass_cb.pack(side="left")
        rst_bd_bass = ttk.Button(frame_bd, text="↺", width=2,
                                 command=lambda: v_bd_bass.set(DEFAULTS["bass_only"]))
        rst_bd_bass.grid(row=3, column=4, padx=(0, 4), pady=3)
        _row_widgets["bd3"] = [bd_bass_cb, rst_bd_bass]

        _set_row_state("bd1", en_bd_threshold.get())
        _set_row_state("bd2", en_bd_gap.get())
        _set_row_state("bd3", en_bd_bass.get())

        # ════════════════════════════════════════════════
        # Movement Settings Override
        # ════════════════════════════════════════════════
        frame = ttk.LabelFrame(dlg, text="Movement Settings Override")
        frame.pack(fill="x", padx=8, pady=(4, 4))

        tk.Label(frame, text="Check a setting to override the global value for this zone.",
                 fg="gray", font=("Segoe UI", 7)).grid(
            row=0, column=0, columnspan=5, padx=8, pady=(4, 2), sticky="w")

        # Movement variables pre-filled with zone or global fallback
        v_mode = tk.StringVar(value=settings.get("mode", self.mode_var.get()))
        v_peak_start = tk.BooleanVar(value=settings.get("peak_start_at_beat",
                                                         self.peak_start_at_beat.get()))
        v_alt_mode = tk.StringVar(value=settings.get("alt_mode", self.alt_mode_var.get()))
        v_speed = tk.DoubleVar(value=settings.get("speed", self.speed.get()))
        v_pos_min = tk.DoubleVar(value=settings.get("pos_min", self.pos_min.get()))
        v_pos_max = tk.DoubleVar(value=settings.get("pos_max", self.pos_max.get()))
        v_noise = tk.DoubleVar(value=settings.get("noise_amount", self.noise_amount.get()))
        v_reverse = tk.BooleanVar(value=settings.get("reverse_curve", self.reverse_curve.get()))

        en_mode = tk.BooleanVar(value="mode" in overrides)
        en_start = tk.BooleanVar(value="peak_start_at_beat" in overrides)
        en_alt = tk.BooleanVar(value="alt_mode" in overrides)
        en_speed = tk.BooleanVar(value="speed" in overrides)
        en_pos_min = tk.BooleanVar(value="pos_min" in overrides)
        en_pos_max = tk.BooleanVar(value="pos_max" in overrides)
        en_noise = tk.BooleanVar(value="noise_amount" in overrides)
        en_reverse = tk.BooleanVar(value="reverse_curve" in overrides)

        # ── Row 11: Mode ──  (use 10+ rows to avoid key collision with beat detection)
        _make_enable_cb(frame, 1, en_mode, key="m1")
        lbl_mode = ttk.Label(frame, text="Mode")
        lbl_mode.grid(row=1, column=1, sticky="w", padx=4, pady=3)
        mode_combo_frame = ttk.Frame(frame)
        mode_combo_frame.grid(row=1, column=2, columnspan=2, sticky="w", padx=4, pady=3)
        mode_combo = ttk.Combobox(mode_combo_frame, textvariable=v_mode,
                     values=[MODE_PEAK, MODE_ALTERNATING], width=16,
                     state="readonly")
        mode_combo.pack(side="left")
        rst_mode = ttk.Button(frame, text="↺", width=2,
                   command=lambda: v_mode.set(DEFAULTS["mode"]))
        rst_mode.grid(row=1, column=4, padx=(0, 4), pady=3)
        _row_widgets["m1"] = [lbl_mode, mode_combo, rst_mode]

        # ── Row 12: Start at beat ──
        _make_enable_cb(frame, 2, en_start, key="m2")
        start_frame = ttk.Frame(frame)
        start_frame.grid(row=2, column=1, columnspan=3, sticky="w", padx=4, pady=3)
        start_cb = ttk.Checkbutton(start_frame, text="Start movement at beat",
                        variable=v_peak_start)
        start_cb.pack(side="left")
        rst_start = ttk.Button(frame, text="↺", width=2,
                   command=lambda: v_peak_start.set(DEFAULTS["peak_start_at_beat"]))
        rst_start.grid(row=2, column=4, padx=(0, 4), pady=3)
        _row_widgets["m2"] = [start_cb, rst_start]

        # ── Row 13: Alternating-specific options ──
        dlg_alt_frame = ttk.Frame(frame)
        dlg_alt_frame.grid(row=3, column=0, columnspan=5, sticky="ew")
        alt_inner = ttk.Frame(dlg_alt_frame)
        alt_inner.pack(fill="x")
        _make_enable_cb(alt_inner, 0, en_alt)
        lbl_alt = ttk.Label(alt_inner, text="Alt. mode")
        lbl_alt.grid(row=0, column=1, sticky="w", padx=4, pady=3)
        alt_combo_frame = ttk.Frame(alt_inner)
        alt_combo_frame.grid(row=0, column=2, columnspan=2, sticky="w", padx=4, pady=3)
        alt_combo = ttk.Combobox(alt_combo_frame, textvariable=v_alt_mode,
                     values=[ALT_FORCE, ALT_KEEP], width=24,
                     state="readonly")
        alt_combo.pack(side="left")
        rst_alt = ttk.Button(alt_inner, text="↺", width=2,
                   command=lambda: v_alt_mode.set(DEFAULTS["alt_mode"]))
        rst_alt.grid(row=0, column=4, padx=(0, 4), pady=3)
        _row_widgets["alt"] = [lbl_alt, alt_combo, rst_alt]

        # ── Rows 14-17: Sliders ──
        _add_zone_slider(frame, 4, "Movement Speed", v_speed, DEFAULTS["speed"], en_speed)
        _add_zone_slider(frame, 5, "Position Min", v_pos_min, DEFAULTS["pos_min"], en_pos_min)
        _add_zone_slider(frame, 6, "Position Max", v_pos_max, DEFAULTS["pos_max"], en_pos_max)
        _add_zone_slider(frame, 7, "Idle Noise", v_noise, DEFAULTS["noise_amount"], en_noise)

        # ── Row 18: Reverse curve ──
        _make_enable_cb(frame, 8, en_reverse, key="m8")
        rev_frame = ttk.Frame(frame)
        rev_frame.grid(row=8, column=1, columnspan=3, sticky="w", padx=4, pady=3)
        rev_cb = ttk.Checkbutton(rev_frame, text="Reverse movement curve",
                        variable=v_reverse)
        rev_cb.pack(side="left")
        rst_rev = ttk.Button(frame, text="↺", width=2,
                   command=lambda: v_reverse.set(DEFAULTS["reverse_curve"]))
        rst_rev.grid(row=8, column=4, padx=(0, 4), pady=3)
        _row_widgets["m8"] = [rev_cb, rst_rev]

        # Apply initial enabled/disabled state for movement rows
        _set_row_state("m1", en_mode.get())
        _set_row_state("m2", en_start.get())
        _set_row_state("alt", en_alt.get())
        _set_row_state(4, en_speed.get())
        _set_row_state(5, en_pos_min.get())
        _set_row_state(6, en_pos_max.get())
        _set_row_state(7, en_noise.get())
        _set_row_state("m8", en_reverse.get())

        # Show/hide mode-specific options
        def _update_dlg_mode_vis(*_):
            mode = v_mode.get()
            if mode == MODE_ALTERNATING:
                dlg_alt_frame.grid()
            else:
                dlg_alt_frame.grid_remove()
        v_mode.trace_add("write", _update_dlg_mode_vis)
        _update_dlg_mode_vis()

        def _apply_live(*_):
            """Apply current dialog state to the zone and refresh display."""
            zone["name"] = v_name.get().strip()
            # Build overrides list — only save enabled settings
            active = []
            _pairs = [
                # Beat detection overrides
                (en_bd_threshold, "bd_threshold", v_bd_threshold),
                (en_bd_gap, "bd_min_beat_gap", v_bd_gap),
                (en_bd_bass, "bd_bass_only", v_bd_bass),
                # Movement overrides
                (en_mode, "mode", v_mode),
                (en_start, "peak_start_at_beat", v_peak_start),
                (en_alt, "alt_mode", v_alt_mode),
                (en_speed, "speed", v_speed),
                (en_pos_min, "pos_min", v_pos_min),
                (en_pos_max, "pos_max", v_pos_max),
                (en_noise, "noise_amount", v_noise),
                (en_reverse, "reverse_curve", v_reverse),
            ]
            for en, key, var in _pairs:
                if en.get():
                    settings[key] = var.get()
                    active.append(key)
                else:
                    settings.pop(key, None)
            settings["_overrides"] = active
            self._regenerate_funscript()
            self._draw_fun_lod()
            self._draw_zones_lod()
            self.canvas.draw_idle()

        # Wire up live preview on every variable change
        for var in (v_mode, v_alt_mode, v_name):
            var.trace_add("write", _apply_live)
        for var in (v_peak_start, v_reverse, v_bd_bass,
                    en_mode, en_start, en_alt, en_speed,
                    en_pos_min, en_pos_max, en_noise, en_reverse,
                    en_bd_threshold, en_bd_gap, en_bd_bass):
            var.trace_add("write", _apply_live)
        for var in (v_speed, v_pos_min, v_pos_max, v_noise,
                    v_bd_threshold, v_bd_gap):
            var.trace_add("write", _apply_live)

        def _ok():
            self._push_undo()
            # Settings are already applied live — just close
            dlg.destroy()

        def _cancel():
            # Revert to backup
            zone["name"] = name_backup
            settings.clear()
            settings.update(settings_backup)
            self._regenerate_funscript()
            self._draw_fun_lod()
            self._draw_zones_lod()
            self.canvas.draw_idle()
            dlg.destroy()

        dlg.protocol("WM_DELETE_WINDOW", _cancel)

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="OK", command=_ok).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Cancel", command=_cancel).pack(side="left", padx=8)

    def _on_delete_key(self):
        """Handle Delete key: zones > keyframes > beats priority."""
        if self._selected_zone_idx is not None:
            self._push_undo()
            del self._override_zones[self._selected_zone_idx]
            self._selected_zone_idx = None
            self._draw_zones_lod()
            self._regenerate_funscript()
            self._draw_fun_lod()
            self.canvas.draw_idle()
            self.status.set("Deleted zone")
            return
        if self._selected_kf:
            self._delete_selected_kf()
            return
        self._delete_selected_beats()

    def _copy_all(self):
        """Copy selected zone, keyframes, or beats."""
        if self._selected_zone_idx is not None:
            zone = self._override_zones[self._selected_zone_idx]
            duration = zone["end_ms"] - zone["start_ms"]
            self._clipboard_zone = {
                "duration_ms": duration,
                "name": zone.get("name", ""),
                "settings": copy.deepcopy(zone["settings"]),
            }
            self._clipboard_beats = None
            self._clipboard_kf = None
            self.status.set("Copied zone settings")
        elif self._selected_kf:
            self._clipboard_zone = None
            self._clipboard_beats = None
            self._copy_kf()
        else:
            self._clipboard_zone = None
            self._clipboard_kf = None
            self._copy_beats()

    def _paste_zone_at_playhead(self):
        """Paste a zone from clipboard at playhead, with overlap prevention."""
        self._push_undo()
        base_ms = int(self._playhead_pos * 1000)
        dur = self._clipboard_zone["duration_ms"]
        new_start = base_ms
        new_end = base_ms + dur
        if self.duration > 0:
            new_end = min(new_end, int(self.duration * 1000))
        # Clamp to avoid overlapping existing zones
        new_start, new_end = self._clamp_zone_no_overlap(new_start, new_end, exclude_idx=None)
        if new_end - new_start < ZONE_MIN_MS:
            self.status.set("Not enough room to paste zone here")
            return
        new_zone = {
            "start_ms": new_start,
            "end_ms": new_end,
            "name": self._clipboard_zone.get("name", ""),
            "settings": copy.deepcopy(self._clipboard_zone["settings"]),
        }
        self._override_zones.append(new_zone)
        self._selected_zone_idx = len(self._override_zones) - 1
        self._draw_zones_lod()
        self._regenerate_funscript()
        self._draw_fun_lod()
        self.canvas.draw_idle()
        self.status.set("Pasted zone at playhead")

    def _paste_all(self):
        """Paste zone, keyframes, or beats at playhead."""
        if self._clipboard_zone is not None:
            self._paste_zone_at_playhead()
        elif self._clipboard_kf is not None:
            self._paste_kf()
        elif self._clipboard_beats is not None:
            self._paste_beats()

    def _on_press(self, event):
        # ── Pin handle click — highest priority for left-click ──
        # Must run before any xdata/inaxes guards since pins are above axes
        if event.button == 1 and self._is_near_pin(event):
            pin_x = event.xdata
            if pin_x is None and event.x is not None:
                try:
                    bbox = self.ax_wave.get_window_extent()
                    xlim = self.ax_wave.get_xlim()
                    frac = (event.x - bbox.x0) / max(1, bbox.x1 - bbox.x0)
                    pin_x = xlim[0] + frac * (xlim[1] - xlim[0])
                except Exception:
                    pin_x = None
            if pin_x is not None:
                xlim = self.ax_wave.get_xlim()
                tol = (xlim[1] - xlim[0]) * 0.03
                in_sec = _tc_to_sec(self.in_var.get())
                out_sec = _tc_to_sec(self.out_var.get())
                in_dist = abs(pin_x - in_sec) if in_sec is not None else float('inf')
                out_dist = abs(pin_x - out_sec) if out_sec is not None else float('inf')
                if in_dist < tol or out_dist < tol:
                    if in_dist <= out_dist:
                        self._dragging = "in"
                    else:
                        self._dragging = "out"
                    return

        # Allow handle drag even when click is slightly outside axes (near top border)
        x_data = event.xdata
        if x_data is None and event.x is not None and event.button == 1:
            try:
                bbox = self.ax_wave.get_window_extent()
                if bbox.x0 <= event.x <= bbox.x1:
                    xlim = self.ax_wave.get_xlim()
                    frac = (event.x - bbox.x0) / (bbox.x1 - bbox.x0)
                    x_data = xlim[0] + frac * (xlim[1] - xlim[0])
            except Exception:
                pass

        if event.inaxes not in (self.ax_wave, self.ax_fun, self.ax_zones) and x_data is None:
            return

        # Middle mouse button
        if event.button == 2:
            # Start pan drag (seek on release if no drag occurred)
            if event.xdata is not None:
                self._mmb_pan_start_x = event.x
                self._mmb_pan_start_xlim = self.ax_wave.get_xlim()
                self._mmb_pan_click_data = event.xdata
                self._mmb_did_pan = False
                self._dragging = "mmb_pan"
            return

        # Right click: reset/delete keyframe or beat
        if event.button == 3:
            if self.beat_edit_mode.get():
                # Keyframe actions in ax_fun
                if event.inaxes == self.ax_fun:
                    if self._selected_kf and self._last_funscript:
                        # Process all selected: reset modified, delete independent
                        self._reset_and_delete_selected_kf()
                    elif self._hover_kf_idx is not None and self._last_funscript:
                        a = self._last_funscript["actions"][self._hover_kf_idx]
                        kf_type = self._kf_types.get(a["at"])
                        if kf_type == "modified":
                            self._reset_kf_to_original(self._hover_kf_idx)
                        elif kf_type == "independent":
                            self._delete_kf_at(self._hover_kf_idx)
                    return
                # Beat delete in ax_wave
                if self._selected_beats:
                    self._delete_selected_beats()
                elif self._hover_beat_idx is not None:
                    self._remove_beat(self._hover_beat_idx)
            return

        if event.button != 1:
            return

        if x_data is None:
            return

        # Zone interactions (left-click in zones axes)
        if event.inaxes == self.ax_zones and event.button == 1:
            click_ms = x_data * 1000
            zone_idx = self._zone_at(click_ms)
            edge_tol_ms = (self.ax_wave.get_xlim()[1] - self.ax_wave.get_xlim()[0]) * 1000 * 0.01

            if event.dblclick and zone_idx is not None:
                self._edit_zone(zone_idx)
                return

            if zone_idx is not None:
                self._selected_zone_idx = zone_idx
                zone = self._override_zones[zone_idx]
                # Check if near edge for resize
                if abs(click_ms - zone["start_ms"]) < edge_tol_ms:
                    self._push_undo()
                    self._dragging = "zone_resize_left"
                    self._zone_drag_start_ms = zone["start_ms"]
                    self._zone_drag_end_ms = zone["end_ms"]
                elif abs(click_ms - zone["end_ms"]) < edge_tol_ms:
                    self._push_undo()
                    self._dragging = "zone_resize_right"
                    self._zone_drag_start_ms = zone["start_ms"]
                    self._zone_drag_end_ms = zone["end_ms"]
                else:
                    self._push_undo()
                    self._dragging = "zone_move"
                    self._zone_drag_click_ms = click_ms
                    self._zone_drag_start_ms = zone["start_ms"]
                    self._zone_drag_end_ms = zone["end_ms"]
                self._draw_zones_lod()
                self.canvas.draw_idle()
                self.status.set("Zone selected — double-click to edit settings, Delete to remove")
            else:
                # Before creating a new zone, check if click is near any zone edge
                # (prefer resize when double-arrow cursor is visible)
                near_edge_zone_idx = None
                near_edge_side = None
                for zi, z in enumerate(self._override_zones):
                    if abs(click_ms - z["end_ms"]) < edge_tol_ms:
                        near_edge_zone_idx = zi
                        near_edge_side = "right"
                        break
                    if abs(click_ms - z["start_ms"]) < edge_tol_ms:
                        near_edge_zone_idx = zi
                        near_edge_side = "left"
                        break
                if near_edge_zone_idx is not None:
                    self._selected_zone_idx = near_edge_zone_idx
                    z = self._override_zones[near_edge_zone_idx]
                    self._push_undo()
                    if near_edge_side == "left":
                        self._dragging = "zone_resize_left"
                    else:
                        self._dragging = "zone_resize_right"
                    self._zone_drag_start_ms = z["start_ms"]
                    self._zone_drag_end_ms = z["end_ms"]
                    self._draw_zones_lod()
                    self.canvas.draw_idle()
                    self.status.set("Zone selected — double-click to edit settings, Delete to remove")
                    return

                # Click on empty area: create a new zone (10% of visible width)
                xlim = self.ax_zones.get_xlim()
                vis_span_ms = (xlim[1] - xlim[0]) * 1000
                zone_width_ms = max(ZONE_MIN_MS, int(vis_span_ms * 0.10))
                new_start = int(click_ms)
                new_end = int(click_ms + zone_width_ms)
                if self.duration > 0:
                    new_end = min(new_end, int(self.duration * 1000))
                # Clamp to avoid overlapping existing zones
                new_start, new_end = self._clamp_zone_no_overlap(
                    new_start, new_end, exclude_idx=None)
                if new_end - new_start < ZONE_MIN_MS:
                    return  # not enough room
                self._push_undo()
                new_zone = {
                    "start_ms": new_start, "end_ms": new_end,
                    "name": "Overriding Zone",
                    "settings": {"_overrides": []},
                }
                self._override_zones.append(new_zone)
                self._selected_zone_idx = len(self._override_zones) - 1
                self._draw_zones_lod()
                self._regenerate_funscript()
                self._draw_fun_lod()
                self.canvas.draw_idle()
                self.status.set(f"Created zone at {_format_time_ms(click_ms / 1000)}")
            return

        if event.inaxes not in (self.ax_wave, self.ax_fun):
            return

        click_pos = max(0, x_data)
        if self.duration > 0:
            click_pos = min(click_pos, self.duration)

        if self.beat_edit_mode.get() and self.y is not None:
            # ── Keyframe editing in ax_fun ──
            if event.inaxes == self.ax_fun:
                click_y = event.ydata if event.ydata is not None else 50

                # Ctrl+click → toggle keyframe in selection
                if self._mod_ctrl:
                    if self._hover_kf_idx is not None:
                        a = self._last_funscript["actions"][self._hover_kf_idx]
                        at_ms = a["at"]
                        if at_ms in self._selected_kf:
                            self._selected_kf.discard(at_ms)
                        else:
                            self._selected_kf.add(at_ms)
                        self._draw_fun_lod()
                        self.canvas.draw_idle()
                        self.status.set(f"{len(self._selected_kf)} keyframes selected")
                    return

                # Click on a hovered keyframe
                if self._hover_kf_idx is not None:
                    a = self._last_funscript["actions"][self._hover_kf_idx]
                    at_ms = a["at"]
                    if at_ms in self._selected_kf:
                        # Already selected → start group drag
                        self._push_undo()
                        self._dragging = "multi_kf"
                        self._clear_kf_hover()
                        self._kf_multi_anchor = at_ms
                        # Store initial absolute positions for all selected kfs
                        self._kf_multi_drag = {}
                        actions = self._last_funscript["actions"]
                        for sel_ms in self._selected_kf:
                            for sa in actions:
                                if sa["at"] == sel_ms:
                                    self._kf_multi_drag[sel_ms] = {
                                        "off_ms": sel_ms - at_ms,
                                        "start_pos": sa["pos"],
                                    }
                                    break
                        self._dragging_kf_orig = {"at": at_ms, "pos": a["pos"]}
                        return
                    # Not in selection → select and start single drag
                    self._push_undo()
                    self._selected_kf = {at_ms}
                    self._dragging_kf_idx = self._hover_kf_idx
                    self._dragging_kf_orig = {"at": at_ms, "pos": a["pos"]}
                    self._clear_kf_hover()
                    self._dragging = "kf"
                    self._draw_fun_lod()
                    self.canvas.draw_idle()
                    return

                # Click in empty area of ax_fun
                if not self._mod_shift:
                    # Start selection rectangle on funscript axis
                    self._selected_kf.clear()
                    self._kf_select_start = (click_pos, click_y)
                    self._dragging = "kf_select_rect"
                return

            # ── Beat editing in ax_wave ──
            # Remove previews on click
            self._remove_snap_preview()
            self._remove_multi_beat_preview()

            # Ctrl+click → toggle beat in selection
            if self._mod_ctrl:
                if self._hover_beat_idx is not None:
                    b_ms = self.beats_ms[self._hover_beat_idx]
                    if b_ms in self._selected_beats:
                        self._selected_beats.discard(b_ms)
                    else:
                        self._selected_beats.add(b_ms)
                    self._draw_beat_lod()
                    self.canvas.draw_idle()
                    self.status.set(f"{len(self._selected_beats)} beats selected")
                return

            # Click on a hovered beat
            if self._hover_beat_idx is not None:
                hovered_ms = self.beats_ms[self._hover_beat_idx]
                if hovered_ms in self._selected_beats:
                    # Already selected → start group drag
                    self._push_undo()
                    self._dragging = "multi_beat"
                    self._multi_drag_anchor_ms = hovered_ms
                    self._multi_drag_offsets = {
                        b_ms: b_ms - hovered_ms
                        for b_ms in self._selected_beats}
                    return
                # Not in selection → select just this beat and prepare drag
                self._push_undo()
                self._selected_beats = {hovered_ms}
                self._dragging_beat_idx = self._hover_beat_idx
                self._dragging_beat_orig_ms = hovered_ms
                self._dragging = "beat"
                self._draw_beat_lod()
                self.canvas.draw_idle()
                return

            # Click in empty area → start selection rectangle
            # (only if not shift-clicking for multi-add)
            if self._mod_shift:
                self._add_multi_beats_at(click_pos)
            else:
                # Start selection rectangle drag
                self._selected_beats.clear()
                self._start_selection_rect(click_pos, event.y if event.y else 0)
                self._dragging = "select_rect"
        else:
            # Normal mode: click seeks playhead
            self._seek_to(click_pos)

    def _pixel_to_data_y(self, event):
        """Convert pixel y to data y on ax_fun, clamped to 0-100."""
        if event.ydata is not None and event.inaxes == self.ax_fun:
            return max(0, min(100, event.ydata))
        try:
            bbox = self.ax_fun.get_window_extent()
            ylim = self.ax_fun.get_ylim()
            if event.y is not None and (bbox.y1 - bbox.y0) > 0:
                frac = (event.y - bbox.y0) / (bbox.y1 - bbox.y0)
                return max(0, min(100, ylim[0] + frac * (ylim[1] - ylim[0])))
        except Exception:
            pass
        return 50

    def _on_motion(self, event):
        # ── Helper to compute data x from pixel, even outside axes ──
        def _pixel_to_data_x():
            if event.xdata is not None:
                return event.xdata
            try:
                bbox = self.ax_wave.get_window_extent()
                xlim = self.ax_wave.get_xlim()
                if event.x is not None:
                    if event.x < bbox.x0:
                        return xlim[0]
                    elif event.x > bbox.x1:
                        return xlim[1]
                    else:
                        frac = (event.x - bbox.x0) / max(1, bbox.x1 - bbox.x0)
                        return xlim[0] + frac * (xlim[1] - xlim[0])
            except Exception:
                pass
            return 0.0

        # ── Handle active drag first (highest priority) ──
        if self._dragging is not None:
            val = _pixel_to_data_x()
            val = max(0.0, val)
            if self.duration > 0:
                val = min(val, self.duration)

            if self._dragging == "beat":
                if self._dragging_beat_idx is not None:
                    new_idx = self._move_beat(self._dragging_beat_idx, val)
                    if new_idx is not None:
                        self._dragging_beat_idx = new_idx
                return

            if self._dragging == "multi_beat":
                # Move all selected beats by the same delta
                # Snap applies to the anchor beat only (always, or when snap_first_only)
                if self._multi_drag_offsets and self._multi_drag_anchor_ms is not None:
                    if self.snap_to_spike.get():
                        val = self._snap_to_spike(val)
                    new_anchor_ms = int(round(val * 1000))
                    # Clamp anchor so no beat goes out of bounds
                    min_off = min(self._multi_drag_offsets.values())
                    max_off = max(self._multi_drag_offsets.values())
                    new_anchor_ms = max(-min_off, new_anchor_ms)
                    dur_ms = int(self.duration * 1000) if self.duration > 0 else 0
                    if dur_ms > 0:
                        new_anchor_ms = min(dur_ms - max_off, new_anchor_ms)
                    # Compute new positions for all selected beats
                    import bisect
                    new_positions = {}
                    for old_ms, offset in self._multi_drag_offsets.items():
                        new_positions[old_ms] = new_anchor_ms + offset
                    # Check no duplicates among new positions
                    if len(set(new_positions.values())) < len(new_positions):
                        return
                    # Remove old, insert new, preserving intensity
                    new_selected = set()
                    for old_ms, new_ms in new_positions.items():
                        if old_ms in self.beats_ms:
                            self.beats_ms.remove(old_ms)
                        inten = self.beats_intensity.pop(old_ms, None)
                        pos = bisect.bisect_left(self.beats_ms, new_ms)
                        if pos < len(self.beats_ms) and self.beats_ms[pos] == new_ms:
                            continue  # skip collision with non-selected beat
                        self.beats_ms.insert(pos, new_ms)
                        if inten is not None:
                            self.beats_intensity[new_ms] = inten
                        new_selected.add(new_ms)
                    self._selected_beats = new_selected
                    self._multi_drag_anchor_ms = new_anchor_ms
                    self._multi_drag_offsets = {
                        b_ms: b_ms - new_anchor_ms
                        for b_ms in self._selected_beats}
                    self._regenerate_funscript()
                    self._draw_beat_lod()
                    self._draw_fun_lod()
                    self.canvas.draw_idle()
                return

            if self._dragging == "kf":
                # Single keyframe drag (x = time, y = position)
                if self._dragging_kf_idx is not None:
                    y_pos = self._pixel_to_data_y(event)
                    new_idx = self._move_kf(self._dragging_kf_idx, val, y_pos)
                    if new_idx is not None:
                        self._dragging_kf_idx = new_idx
                return

            if self._dragging == "multi_kf":
                # Multi-keyframe drag — preserve relative deltas
                if self._kf_multi_drag and self._kf_multi_anchor is not None:
                    y_pos = self._pixel_to_data_y(event)
                    new_anchor_ms = int(round(val * 1000))
                    new_anchor_ms = max(0, new_anchor_ms)
                    dur_ms = int(self.duration * 1000) if self.duration > 0 else 0

                    # Clamp anchor so no keyframe goes out of time bounds
                    if dur_ms > 0:
                        min_off = min(o["off_ms"] for o in self._kf_multi_drag.values())
                        max_off = max(o["off_ms"] for o in self._kf_multi_drag.values())
                        new_anchor_ms = max(-min_off, new_anchor_ms)
                        new_anchor_ms = min(dur_ms - max_off, new_anchor_ms)

                    # Compute y delta from drag start, clamped so no kf exceeds 0-100
                    raw_y_delta = y_pos - self._dragging_kf_orig["pos"]
                    min_start = min(o["start_pos"] for o in self._kf_multi_drag.values())
                    max_start = max(o["start_pos"] for o in self._kf_multi_drag.values())
                    # Clamp: min_start + delta >= 0 AND max_start + delta <= 100
                    y_delta = max(-min_start, min(100 - max_start, raw_y_delta))

                    # Build target map: old_sel_ms -> (target_ms, start_pos, off_ms)
                    targets = {}
                    actions = self._last_funscript["actions"] if self._last_funscript else []
                    for sel_ms, offsets in list(self._kf_multi_drag.items()):
                        target_ms = new_anchor_ms + offsets["off_ms"]
                        new_p = offsets["start_pos"] + y_delta
                        targets[sel_ms] = (target_ms, new_p, offsets["off_ms"], offsets["start_pos"])

                    # Move each selected keyframe using absolute start positions
                    for sel_ms, (target_ms, new_p, _, _) in targets.items():
                        for i, a in enumerate(actions):
                            if a["at"] == sel_ms:
                                self._move_kf(i, target_ms / 1000.0, new_p)
                                break
                        # Re-fetch actions after each move (indices shift)
                        actions = self._last_funscript["actions"] if self._last_funscript else []

                    # Rebuild tracking: match by expected target_ms
                    new_multi = {}
                    new_selected = set()
                    actions = self._last_funscript["actions"] if self._last_funscript else []
                    action_set = {a["at"] for a in actions}
                    for _, (target_ms, _, off_ms, start_pos) in targets.items():
                        # Find the actual at_ms (may differ due to lock/clamp)
                        actual = target_ms if target_ms in action_set else None
                        if actual is None:
                            # Fallback: find closest in selected_kf
                            for at in self._selected_kf:
                                if at not in new_selected:
                                    actual = at
                                    break
                        if actual is not None and actual in action_set:
                            new_multi[actual] = {
                                "off_ms": off_ms,
                                "start_pos": start_pos,
                            }
                            new_selected.add(actual)
                    self._kf_multi_drag = new_multi
                    self._kf_multi_anchor = new_anchor_ms
                    self._selected_kf = new_selected
                return

            if self._dragging == "kf_select_rect":
                # Update selection rectangle on funscript axis
                if self._kf_select_start is not None:
                    y_pos = self._pixel_to_data_y(event)
                    x0, y0 = self._kf_select_start
                    x1, y1 = val, y_pos
                    # Select keyframes within rectangle
                    left = min(x0, x1)
                    right = max(x0, x1)
                    bottom = min(y0, y1)
                    top = max(y0, y1)
                    self._selected_kf.clear()
                    if self._last_funscript:
                        for a in self._last_funscript["actions"]:
                            t = a["at"] / 1000.0
                            p = a["pos"]
                            if left <= t <= right and bottom <= p <= top:
                                self._selected_kf.add(a["at"])
                    # Draw selection rectangle
                    if self._selection_rect_artist is not None:
                        try:
                            self._selection_rect_artist.remove()
                        except Exception:
                            pass
                    from matplotlib.patches import Rectangle
                    self._selection_rect_artist = Rectangle(
                        (left, bottom), right - left, top - bottom,
                        fill=True, facecolor="#7aa2f7", alpha=0.15,
                        edgecolor="#7aa2f7", linewidth=1, zorder=20)
                    self.ax_fun.add_patch(self._selection_rect_artist)
                    self._draw_fun_lod()
                    self.canvas.draw_idle()
                return

            if self._dragging == "zone_move":
                if self._selected_zone_idx is not None:
                    zone = self._override_zones[self._selected_zone_idx]
                    delta_ms = val * 1000 - self._zone_drag_click_ms
                    new_start = int(self._zone_drag_start_ms + delta_ms)
                    new_end = int(self._zone_drag_end_ms + delta_ms)
                    if new_start < 0:
                        new_end -= new_start
                        new_start = 0
                    if self.duration > 0 and new_end > int(self.duration * 1000):
                        overshoot = new_end - int(self.duration * 1000)
                        new_end = int(self.duration * 1000)
                        new_start -= overshoot
                    # Prevent overlap with other zones
                    new_start, new_end = self._clamp_zone_move_no_overlap(
                        new_start, new_end, exclude_idx=self._selected_zone_idx)
                    new_start = max(0, new_start)
                    if self.duration > 0:
                        new_end = min(new_end, int(self.duration * 1000))
                    zone["start_ms"] = new_start
                    zone["end_ms"] = new_end
                    self._draw_zones_lod()
                    self.canvas.draw_idle()
                return

            if self._dragging == "zone_resize_left":
                if self._selected_zone_idx is not None:
                    zone = self._override_zones[self._selected_zone_idx]
                    new_start = int(val * 1000)
                    if self.snap_to_spike.get():
                        new_start = int(self._snap_to_spike(val) * 1000)
                    new_start = max(0, min(new_start, zone["end_ms"] - ZONE_MIN_MS))
                    # Prevent overlap: don't extend past neighboring zone's end
                    for i, z in enumerate(self._override_zones):
                        if i == self._selected_zone_idx:
                            continue
                        if z["end_ms"] > new_start and z["start_ms"] < zone["end_ms"]:
                            new_start = max(new_start, z["end_ms"])
                    zone["start_ms"] = new_start
                    self._draw_zones_lod()
                    self.canvas.draw_idle()
                return

            if self._dragging == "zone_resize_right":
                if self._selected_zone_idx is not None:
                    zone = self._override_zones[self._selected_zone_idx]
                    new_end = int(val * 1000)
                    if self.snap_to_spike.get():
                        new_end = int(self._snap_to_spike(val) * 1000)
                    new_end = max(zone["start_ms"] + ZONE_MIN_MS, new_end)
                    if self.duration > 0:
                        new_end = min(new_end, int(self.duration * 1000))
                    # Prevent overlap: don't extend past neighboring zone's start
                    for i, z in enumerate(self._override_zones):
                        if i == self._selected_zone_idx:
                            continue
                        if z["start_ms"] < new_end and z["end_ms"] > zone["start_ms"]:
                            new_end = min(new_end, z["start_ms"])
                    zone["end_ms"] = new_end
                    self._draw_zones_lod()
                    self.canvas.draw_idle()
                return

            if self._dragging == "select_rect":
                self._update_selection_rect(val, event.y if event.y else 0)
                self.canvas.draw_idle()
                return

            if self._dragging == "mmb_pan":
                if self._mmb_pan_start_x is not None and event.x is not None:
                    dx_px = abs(event.x - self._mmb_pan_start_x)
                    if dx_px > 3:  # threshold to distinguish click from drag
                        self._mmb_did_pan = True
                    if self._mmb_did_pan:
                        try:
                            bbox = self.ax_wave.get_window_extent()
                            px_width = bbox.x1 - bbox.x0
                            if px_width > 0:
                                orig_xlim = self._mmb_pan_start_xlim
                                data_span = orig_xlim[1] - orig_xlim[0]
                                dx_data = -(event.x - self._mmb_pan_start_x) / px_width * data_span
                                new_left = orig_xlim[0] + dx_data
                                new_right = orig_xlim[1] + dx_data
                                new_left, new_right = self._clamp_xlim(new_left, new_right)
                                self.ax_wave.set_xlim(new_left, new_right)
                                self._on_view_changed()
                                self._sync_scrollbar()
                                self.canvas.draw_idle()
                        except Exception:
                            pass
                return

            # In/out marker drag
            self._suppress_marker_draw = True
            if self._dragging == "in":
                self.in_var.set(_sec_to_tc(val))
            else:
                self.out_var.set(_sec_to_tc(val))
            self._suppress_marker_draw = False
            self.canvas.draw_idle()
            return

        # ── Pin handle hover — must run BEFORE the xdata-is-None early return ──
        near_pin = self._is_near_pin(event)
        if near_pin:
            hover_x = event.xdata
            if hover_x is None:
                try:
                    bbox = self.ax_wave.get_window_extent()
                    xlim = self.ax_wave.get_xlim()
                    frac = (event.x - bbox.x0) / max(1, bbox.width)
                    hover_x = xlim[0] + frac * (xlim[1] - xlim[0])
                except Exception:
                    hover_x = None
            self._update_marker_hover(hover_x if hover_x is not None else -9999,
                                      near_pin=True)
            # Clear non-pin hovers, stay in pin zone
            if self._hover_beat_idx is not None:
                self._clear_beat_hover()
            self._remove_snap_preview()
            self._remove_multi_beat_preview()
            self.canvas.get_tk_widget().config(cursor="")
            return

        if event.xdata is None:
            # Mouse left axes — clear hovers
            if self._hover_beat_idx is not None:
                self._clear_beat_hover()
                self.canvas.draw_idle()
            if self._hover_marker is not None:
                self._update_marker_hover(-9999)
            self._remove_snap_preview()
            self._remove_multi_beat_preview()
            self.canvas.get_tk_widget().config(cursor="")
            return

        # Zone edge cursor (double-arrow on resize handles)
        if event.inaxes == self.ax_zones and self._override_zones:
            click_ms = event.xdata * 1000
            xlim = self.ax_zones.get_xlim()
            span_ms = (xlim[1] - xlim[0]) * 1000
            edge_tol_ms = max(span_ms * 0.008, 20)
            on_edge = False
            for zone in self._override_zones:
                if (abs(click_ms - zone["start_ms"]) < edge_tol_ms or
                        abs(click_ms - zone["end_ms"]) < edge_tol_ms):
                    on_edge = True
                    break
            self.canvas.get_tk_widget().config(
                cursor="sb_h_double_arrow" if on_edge else "")
        else:
            self.canvas.get_tk_widget().config(cursor="")

        # Marker pin hover for mouse inside axes
        self._update_marker_hover(event.xdata if event.xdata is not None else -9999,
                                  near_pin=False)

        # Hover highlights (only in beat editor mode)
        need_redraw = False
        if self.beat_edit_mode.get():
            if event.inaxes == self.ax_fun:
                # Keyframe hover in funscript axis
                y_pos = event.ydata if event.ydata is not None else 50
                self._update_kf_hover(event.xdata, y_pos)
                # Clear beat hover when in funscript axis
                if self._hover_beat_idx is not None:
                    self._clear_beat_hover()
                    need_redraw = True
                if self._multi_beat_preview_artists:
                    self._remove_multi_beat_preview()
                    need_redraw = True
                if self._snap_preview_line is not None:
                    self._remove_snap_preview()
                    need_redraw = True
                if need_redraw:
                    self.canvas.draw_idle()
            elif event.inaxes == self.ax_wave:
                # Beat hover in waveform axis
                self._update_beat_hover(event.xdata)
                # Clear keyframe hover when in waveform axis
                if self._hover_kf_idx is not None:
                    self._clear_kf_hover()
                    need_redraw = True
                # Shift held → show multi-beat placement preview
                if self._mod_shift:
                    self._remove_snap_preview()
                    self._update_multi_beat_preview(event.xdata)
                    need_redraw = True
                else:
                    if self._multi_beat_preview_artists:
                        self._remove_multi_beat_preview()
                        need_redraw = True
                    if self._hover_beat_idx is None and self.snap_to_spike.get():
                        self._update_snap_preview(event.xdata)
                        need_redraw = True
                    else:
                        if self._snap_preview_line is not None:
                            self._remove_snap_preview()
                            need_redraw = True
                if need_redraw:
                    self.canvas.draw_idle()
            else:
                # Not in wave or fun axes
                if self._multi_beat_preview_artists:
                    self._remove_multi_beat_preview()
                    need_redraw = True
                if self._hover_kf_idx is not None:
                    self._clear_kf_hover()
                    need_redraw = True
                if need_redraw:
                    self.canvas.draw_idle()
        else:
            if self._multi_beat_preview_artists:
                self._remove_multi_beat_preview()
                need_redraw = True
            if self._hover_beat_idx is not None:
                self._clear_beat_hover()
                need_redraw = True
            if self._hover_kf_idx is not None:
                self._clear_kf_hover()
                need_redraw = True
            if need_redraw:
                self.canvas.draw_idle()

    def _on_release(self, event):
        if self._dragging == "select_rect":
            # Finalize selection rectangle
            val = event.xdata
            if val is None:
                try:
                    bbox = self.ax_wave.get_window_extent()
                    xlim = self.ax_wave.get_xlim()
                    if event.x is not None:
                        frac = (event.x - bbox.x0) / max(1, bbox.x1 - bbox.x0)
                        val = xlim[0] + frac * (xlim[1] - xlim[0])
                except Exception:
                    val = 0
            if val is not None and self._selection_start_xy is not None:
                x0 = self._selection_start_xy[0]
                # Only treat as selection if dragged at least a small distance
                if abs(val - x0) > (self.ax_wave.get_xlim()[1] - self.ax_wave.get_xlim()[0]) * 0.003:
                    self._finish_selection_rect(val)
                else:
                    # Tiny drag = add beat at click position
                    if self._selection_rect_artist is not None:
                        try:
                            self._selection_rect_artist.remove()
                        except Exception:
                            pass
                        self._selection_rect_artist = None
                    self._selection_start_xy = None
                    self._selecting = False
                    self._add_beat_at(x0)
            else:
                self._selection_start_xy = None
                self._selecting = False
        if self._dragging == "kf_select_rect":
            # Finalize keyframe selection rectangle
            if self._selection_rect_artist is not None:
                try:
                    self._selection_rect_artist.remove()
                except Exception:
                    pass
                self._selection_rect_artist = None
            if self._kf_select_start is not None:
                val = event.xdata
                y_val = self._pixel_to_data_y(event)
                if val is not None:
                    x0, y0 = self._kf_select_start
                    dist = ((val - x0) ** 2 + (y_val - y0) ** 2) ** 0.5
                    xlim = self.ax_fun.get_xlim()
                    if dist < (xlim[1] - xlim[0]) * 0.003:
                        # Tiny drag = add independent keyframe
                        self._add_independent_kf(x0, y0)
                # else: selection already applied during drag
                self._kf_select_start = None
            self._draw_fun_lod()
            self.canvas.draw_idle()
        if self._dragging in ("zone_move", "zone_resize_left", "zone_resize_right"):
            # Finalize zone drag - regenerate funscript
            self._regenerate_funscript()
            self._draw_fun_lod()
            self.canvas.draw_idle()
        if self._dragging == "mmb_pan":
            if not self._mmb_did_pan and self._mmb_pan_click_data is not None:
                # No drag occurred — seek playhead to click position (with snapping)
                click_pos = max(0, self._mmb_pan_click_data)
                if self.duration > 0:
                    click_pos = min(click_pos, self.duration)
                if self.snap_to_spike.get():
                    click_pos = self._snap_to_spike(click_pos)
                self._seek_to(click_pos)
            self._mmb_pan_start_x = None
            self._mmb_pan_start_xlim = None
            self._mmb_pan_click_data = None
            self._mmb_did_pan = False
        self._dragging = None
        self._dragging_beat_idx = None
        self._dragging_beat_orig_ms = None
        self._multi_drag_offsets = None
        self._multi_drag_anchor_ms = None
        self._dragging_kf_idx = None
        self._dragging_kf_orig = None
        self._kf_multi_drag = None
        self._kf_multi_anchor = None
        self._zone_drag_start_ms = None
        self._zone_drag_end_ms = None
        self._zone_drag_click_ms = None

    # ── Actions ──

    def _set_buttons_state(self, state):
        for b in (self.btn_analyze, self.btn_create):
            b.config(state=state)

    def _progress_cb(self, msg, pct):
        self.root.after(0, lambda: self.status.set(msg))
        self.root.after(0, lambda: self.progress.config(value=pct))

    def _run(self, mode):
        source = self.file_path.get().strip()
        out = self.out_dir.get().strip()

        if not source or not os.path.isfile(source):
            messagebox.showerror("Error", "Please select a valid source file.")
            return
        if mode in ("create", "both"):
            if not out or not os.path.isdir(out):
                messagebox.showerror("Error", "Please select a valid output directory.")
                return
        if mode == "create" and self.beats_ms is None:
            messagebox.showerror("Error", "Run Analyze first to detect beats.")
            return

        # Warn if analyze would override existing beats
        if mode in ("analyze", "both") and self.beats_ms:
            if not messagebox.askyesno(
                    "Override Beats",
                    f"Analyzing will replace beats within the In/Out range. "
                    f"Beats outside the range will be preserved.\n\nContinue?"):
                return

        # Push undo before analyze modifies beats
        if mode in ("analyze", "both"):
            self._push_undo()

        self._stop_playback()
        self._set_buttons_state("disabled")
        self.progress.config(value=0)
        Thread(target=self._worker, args=(mode, source, out), daemon=True).start()

    def _worker(self, mode, source_path, out_dir):
        try:
            need_analyze = mode in ("analyze", "both")
            need_create = mode in ("create", "both")

            if need_analyze:
                if self.y is None:
                    audio_path = source_path
                    if is_video(source_path):
                        audio_path = extract_audio_from_video(source_path, self._progress_cb)
                        self.tmp_audio = audio_path
                    y_raw, self.sr = load_audio(audio_path, self._progress_cb)
                    self.y_raw = y_raw
                    if self.auto_normalize.get():
                        self.y = normalize_audio(y_raw, self.norm_percentile.get())
                    else:
                        self.y = y_raw.copy()
                    self.duration = len(self.y) / self.sr
                    if self.tmp_audio:
                        try:
                            os.unlink(self.tmp_audio)
                        except OSError:
                            pass
                        self.tmp_audio = None
                    self.root.after(0, lambda: self.out_var.set(_sec_to_tc(self.duration)))

                in_ms, out_ms = self._get_in_out_ms()
                min_gap = int(self.min_beat_gap.get() * 2000)  # slider 0‒1 → 0‒2000 ms
                new_beats = detect_beats(
                    self.y, self.sr,
                    threshold=self.threshold.get(),
                    in_ms=in_ms, out_ms=out_ms,
                    bass_only=self.bass_only.get(),
                    min_gap_ms=min_gap,
                    progress_cb=self._progress_cb,
                )

                # Preserve beats outside the in/out range
                range_start = int(in_ms) if in_ms is not None else 0
                range_end = int(out_ms) if out_ms is not None else int(self.duration * 1000)
                if self.beats_ms:
                    kept = [b for b in self.beats_ms
                            if b < range_start or b > range_end]
                    # Also remove intensities for beats that will be replaced
                    for b in self.beats_ms:
                        if range_start <= b <= range_end:
                            self.beats_intensity.pop(b, None)
                else:
                    kept = []
                self.beats_ms = sorted(kept + new_beats)

                # Re-analyze zones with beat detection overrides
                bd_keys = {"bd_threshold", "bd_min_beat_gap", "bd_bass_only"}
                for zone in self._override_zones:
                    zs = zone["settings"]
                    active = set(zs.get("_overrides", []))
                    if not active.intersection(bd_keys):
                        continue  # no beat detection overrides
                    z_start = zone["start_ms"]
                    z_end = zone["end_ms"]
                    # Only re-analyze if zone overlaps in/out range
                    if z_end < range_start or z_start > range_end:
                        continue
                    z_threshold = zs.get("bd_threshold", self.threshold.get()) if "bd_threshold" in active else self.threshold.get()
                    z_gap = int((zs.get("bd_min_beat_gap", self.min_beat_gap.get()) if "bd_min_beat_gap" in active else self.min_beat_gap.get()) * 2000)
                    z_bass = (zs.get("bd_bass_only", self.bass_only.get()) if "bd_bass_only" in active else self.bass_only.get())
                    zone_beats = detect_beats(
                        self.y, self.sr,
                        threshold=z_threshold,
                        in_ms=z_start, out_ms=z_end,
                        bass_only=z_bass,
                        min_gap_ms=z_gap,
                    )
                    # Remove existing beats in zone range and replace
                    self.beats_ms = [b for b in self.beats_ms
                                     if b < z_start or b > z_end]
                    for b in list(self.beats_intensity.keys()):
                        if z_start <= b <= z_end:
                            self.beats_intensity.pop(b, None)
                    self.beats_ms = sorted(self.beats_ms + zone_beats)

                min_p = int(round(self.pos_min.get() * 100))
                max_p = int(round(self.pos_max.get() * 100))
                if min_p >= max_p:
                    min_p, max_p = 0, 100

                noise = self.noise_amount.get()
                # Clear keyframe edits when re-analyzing (base changed)
                self._kf_moved.clear()
                self._kf_deleted.clear()
                self._kf_independent.clear()
                self._selected_kf.clear()

                self._last_funscript = generate_funscript(
                    self.beats_ms,
                    speed=self.speed.get(),
                    min_pos=min_p,
                    max_pos=max_p,
                    mode=self.mode_var.get(),
                    noise_amount=noise,
                    intensities=self.beats_intensity,
                    alt_mode=self.alt_mode_var.get(),
                    start_at_beat=self.peak_start_at_beat.get(),
                    reverse_curve=self.reverse_curve.get(),
                )

                self._progress_cb("Drawing…", 85)
                self.root.after(0, self._draw_all)

            if need_create:
                self._progress_cb("Writing funscript…", 90)
                if self._last_funscript is None:
                    min_p = int(round(self.pos_min.get() * 100))
                    max_p = int(round(self.pos_max.get() * 100))
                    if min_p >= max_p:
                        min_p, max_p = 0, 100
                    noise = self.noise_amount.get()
                    self._last_funscript = generate_funscript(
                        self.beats_ms,
                        speed=self.speed.get(),
                        min_pos=min_p,
                        max_pos=max_p,
                        mode=self.mode_var.get(),
                        noise_amount=noise,
                        intensities=self.beats_intensity,
                        alt_mode=self.alt_mode_var.get(),
                        start_at_beat=self.peak_start_at_beat.get(),
                        reverse_curve=self.reverse_curve.get(),
                    )
                # Filter actions to marker range
                in_sec, out_sec = self._get_marker_bounds()
                in_ms_f = in_sec * 1000
                out_ms_f = out_sec * 1000
                filtered_actions = [a for a in self._last_funscript["actions"]
                                    if in_ms_f <= a["at"] <= out_ms_f]
                export_fs = {"version": "1.0",
                             "actions": filtered_actions,
                             "range": self._last_funscript.get("range", 100)}

                base = os.path.splitext(os.path.basename(source_path))[0]
                out_path = os.path.join(out_dir, base + ".funscript")
                with open(out_path, "w") as f:
                    json.dump(export_fs, f, indent=2)

                n_actions = len(filtered_actions)
                n_beats = len(self.beats_ms)
                self._progress_cb(f"Done — {n_actions} actions saved.", 100)
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"Saved {out_path}\n\n{n_actions} actions from {n_beats} beats."))
            else:
                n = len(self.beats_ms) if self.beats_ms else 0
                self._progress_cb(f"Analysis done — {n} beats detected.", 100)

        except Exception as e:
            self._progress_cb("Error", 0)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self._set_buttons_state("normal"))

    # ── Drawing ──

    def _draw_all(self):
        # Preserve current zoom/pan when redrawing after analysis
        old_xlim = self.ax_wave.get_xlim()
        has_view = old_xlim[1] > old_xlim[0] and old_xlim[1] > 0

        self._draw_waveform(preserve_view=True)
        self._draw_funscript()
        self._draw_zones_lod()

        # Sync funscript x-axis with waveform view
        if has_view:
            self.ax_fun.set_xlim(old_xlim)

        self._draw_markers()
        self._draw_playhead_lines()
        self._update_shading()
        self.fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.08, hspace=0.08)
        self._sync_scrollbar()
        self.canvas.draw()

    def _draw_waveform(self, preserve_view=False):
        ax = self.ax_wave
        # Save current view before clearing so we can restore it
        old_xlim = ax.get_xlim()
        has_valid_view = old_xlim[1] > old_xlim[0] and old_xlim[1] > 0

        ax.clear()
        self._style_axes()

        if self.y is None or self.sr is None:
            return

        duration = self.duration

        # Compute envelope once and cache it
        if self._t_env is None:
            self._t_env, self._env_max = _compute_envelope(self.y, self.sr)
            self._env_mipmap = _build_envelope_mipmap(self._t_env, self._env_max)
        # Reset persistent artists (axes were cleared)
        self._wave_fill = None

        if preserve_view and has_valid_view:
            ax.set_xlim(old_xlim)
        else:
            ax.set_xlim(0, duration)

        # Since audio is normalized to [-1, 1], use fixed ylim so waveform
        # fills the full panel height.
        ym = 1.0
        ax.set_ylim(-ym, ym)
        ax.yaxis.set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)

        self._beat_ym = ym

        # Draw waveform and beats via LOD
        self._draw_wave_lod()
        self._draw_beat_lod()

        fname = os.path.basename(self.file_path.get())
        dur_str = _format_time(duration)
        n_beats = len(self.beats_ms) if self.beats_ms else 0
        mode_str = self.mode_var.get()
        pct = self.norm_percentile.get()
        norm_str = f"norm {pct:.0f}%" if self.auto_normalize.get() or self.y_raw is not self.y else "raw"
        ax.set_title(f"{fname}   [{dur_str}]   {n_beats} beats   ({mode_str})   [{norm_str}]",
                     color=TEXT_COLOR, fontsize=9, loc="left", pad=14)

    def _wave_polygon_xy(self, t_draw, e_draw):
        """Build closed polygon vertices: top envelope forward, bottom reversed."""
        # top half (left→right), then bottom half (right→left)
        xy = np.empty((len(t_draw) * 2, 2), dtype=np.float64)
        n = len(t_draw)
        xy[:n, 0] = t_draw
        xy[:n, 1] = e_draw
        xy[n:, 0] = t_draw[::-1]
        xy[n:, 1] = -e_draw[::-1]
        return xy

    def _draw_wave_lod(self):
        """Draw waveform envelope for the current view using mipmap LOD."""
        ax = self.ax_wave
        # Remove tempo grid artists only (waveform uses persistent polygon)
        for a in self._wave_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._wave_artists = []

        if self._t_env is None or self._env_max is None:
            return

        xlim = ax.get_xlim()
        width_px = self._get_axes_width_px(ax)
        target_pts = max(10, width_px * 2)  # ~2 points per pixel

        # Pick the coarsest mipmap level that has enough points for the view
        mip = self._env_mipmap
        if mip is None:
            mip = [(self._t_env, self._env_max)]
        t_env, env_max = mip[0]  # default: finest
        for t_lev, e_lev in mip:
            i0 = max(0, int(np.searchsorted(t_lev, xlim[0])) - 1)
            i1 = min(len(t_lev), int(np.searchsorted(t_lev, xlim[1])) + 2)
            n_vis = i1 - i0
            t_env, env_max = t_lev, e_lev
            if n_vis <= target_pts:
                break

        # Slice to visible range with margin
        i0 = max(0, int(np.searchsorted(t_env, xlim[0])) - 1)
        i1 = min(len(t_env), int(np.searchsorted(t_env, xlim[1])) + 2)
        t_draw = t_env[i0:i1]
        e_draw = env_max[i0:i1]

        if len(t_draw) == 0:
            return

        xy = self._wave_polygon_xy(t_draw, e_draw)

        # Update persistent Polygon or create it once
        if self._wave_fill is not None and self._wave_fill.axes is not None:
            self._wave_fill.set_xy(xy)
        else:
            self._wave_fill = MplPolygon(
                xy, closed=True, facecolor=WAVE_COLOR, edgecolor='none',
                alpha=0.6, zorder=1)
            ax.add_patch(self._wave_fill)

        # Draw musical beat grid lines behind waveform
        self._draw_tempo_grid_on_ax(ax, xlim=ax.get_xlim())

    def _draw_funscript(self):
        ax = self.ax_fun
        ax.clear()
        self._fun_line = None
        self._fun_fill = None
        self._style_axes()

        ax.set_ylabel("Pos", color=TEXT_COLOR, fontsize=9)
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_time))

        if self._last_funscript is None or not self._last_funscript["actions"]:
            ax.text(0.5, 0.5, "No funscript data",
                    transform=ax.transAxes, ha="center", va="center",
                    color=TEXT_COLOR, fontsize=10, alpha=0.4)
            return

        for yy in (0, 50, 100):
            ax.axhline(yy, color=GRID_COLOR, linewidth=0.5, alpha=0.4)

        self._draw_fun_lod()

    def _get_axes_width_px(self, ax):
        """Return the width of an axes in pixels."""
        try:
            bbox = ax.get_window_extent()
            return max(1, int(bbox.width))
        except Exception:
            return 800  # fallback

    def _thin_to_pixel_budget(self, values, ax, per_100px=4):
        """Thin a sorted list of time values so there are at most
        *per_100px* items per 100 pixels of screen space."""
        if not values:
            return values
        width_px = self._get_axes_width_px(ax)
        max_items = max(10, width_px * per_100px // 100)
        if len(values) <= max_items:
            return values
        step = len(values) / max_items
        return [values[int(i * step)] for i in range(max_items)]

    def _draw_beat_lod(self):
        """Draw beat markers for the current view (LOD), colored by intensity."""
        ax = self.ax_wave
        for a in self._beat_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._beat_artists = []

        if not self.beats_ms:
            return
        ym = getattr(self, '_beat_ym', 1)
        xlim = ax.get_xlim()

        # Use numpy for fast visible-range filtering via searchsorted
        all_ms = np.asarray(self.beats_ms, dtype=np.int64)
        lo = int(xlim[0] * 1000)
        hi = int(xlim[1] * 1000)
        i0 = np.searchsorted(all_ms, lo, side='left')
        i1 = np.searchsorted(all_ms, hi, side='right')
        vis_ms = all_ms[i0:i1]

        if len(vis_ms) == 0:
            return

        # Pixel-budget thinning
        width_px = self._get_axes_width_px(ax)
        max_items = max(10, width_px * 50 // 100)
        if len(vis_ms) > max_items:
            step = len(vis_ms) / max_items
            indices = np.arange(0, max_items) * step
            vis_ms = vis_ms[indices.astype(int)]

        vis_sec = vis_ms / 1000.0
        sel_set = self._selected_beats
        bi = self.beats_intensity

        # Build segments and colors for a single LineCollection
        segments = []
        colors = []
        sel_segments = []
        for j in range(len(vis_ms)):
            b_ms = int(vis_ms[j])
            b_sec = float(vis_sec[j])
            seg = [(b_sec, -ym), (b_sec, ym)]
            if b_ms in sel_set:
                sel_segments.append(seg)
            else:
                segments.append(seg)
                inten = bi.get(b_ms, 1.0)
                colors.append(_beat_intensity_color(inten))

        if segments:
            from matplotlib.collections import LineCollection as _LC
            lc = _LC(segments, colors=colors, linewidths=2.5, alpha=0.55)
            ax.add_collection(lc)
            self._beat_artists.append(lc)

        if sel_segments:
            from matplotlib.collections import LineCollection as _LC
            lc = _LC(sel_segments, colors="#ffffff", linewidths=3.0,
                     alpha=0.85, zorder=10)
            ax.add_collection(lc)
            self._beat_artists.append(lc)

    def _draw_fun_lod(self):
        """Draw funscript line/markers for the current view (LOD)."""
        ax = self.ax_fun
        for a in self._fun_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._fun_artists = []

        if self._last_funscript is None or not self._last_funscript["actions"]:
            # Hide persistent artists if they exist
            if self._fun_fill is not None and self._fun_fill.axes is not None:
                self._fun_fill.set_xy(np.zeros((1, 2)))
            if self._fun_line is not None and self._fun_line.axes is not None:
                self._fun_line.set_data([], [])
            return

        actions = self._last_funscript["actions"]

        # Build numpy arrays once for the full action list
        n_total = len(actions)
        all_at = np.empty(n_total + 1, dtype=np.float64)
        all_pos = np.empty(n_total + 1, dtype=np.float64)
        for i, a in enumerate(actions):
            all_at[i] = a["at"]
            all_pos[i] = a["pos"]

        # Extend line to end of timeline so it never stops short
        dur_ms = self.duration * 1000 if self.duration > 0 else 0
        if n_total > 0 and all_at[n_total - 1] < dur_ms:
            all_at[n_total] = dur_ms
            all_pos[n_total] = all_pos[n_total - 1]
            n_total += 1
        all_at = all_at[:n_total]
        all_pos = all_pos[:n_total]

        xlim = ax.get_xlim()
        vis_start_ms = xlim[0] * 1000
        vis_end_ms = xlim[1] * 1000

        first_idx = max(0, int(np.searchsorted(all_at, vis_start_ms, side='left')) - 1)
        last_idx = min(n_total - 1,
                       int(np.searchsorted(all_at, vis_end_ms, side='right')))
        n_vis = last_idx - first_idx + 1
        if n_vis <= 0:
            return

        t_arr = all_at[first_idx:last_idx + 1] / 1000.0
        p_arr = all_pos[first_idx:last_idx + 1]

        # LOD thinning with vectorized numpy min/max bucketing
        width_px = self._get_axes_width_px(ax)
        n_buckets = max(5, width_px)

        if n_vis > n_buckets * 2:
            bucket_idx = np.linspace(0, n_vis, n_buckets + 1, dtype=int)
            t_parts = []
            p_parts = []
            for b in range(n_buckets):
                s, e = bucket_idx[b], bucket_idx[b + 1]
                if s >= e:
                    continue
                bp = p_arr[s:e]
                mi = s + int(np.argmin(bp))
                ma = s + int(np.argmax(bp))
                if mi == ma:
                    t_parts.append(t_arr[mi])
                    p_parts.append(p_arr[mi])
                elif mi < ma:
                    t_parts.extend((t_arr[mi], t_arr[ma]))
                    p_parts.extend((p_arr[mi], p_arr[ma]))
                else:
                    t_parts.extend((t_arr[ma], t_arr[mi]))
                    p_parts.extend((p_arr[ma], p_arr[mi]))
            t_draw = np.array(t_parts)
            p_draw = np.array(p_parts)
            if len(t_draw) > 0:
                if t_draw[0] != t_arr[0]:
                    t_draw = np.concatenate([[t_arr[0]], t_draw])
                    p_draw = np.concatenate([[p_arr[0]], p_draw])
                if t_draw[-1] != t_arr[-1]:
                    t_draw = np.concatenate([t_draw, [t_arr[-1]]])
                    p_draw = np.concatenate([p_draw, [p_arr[-1]]])
        else:
            t_draw = t_arr
            p_draw = p_arr

        # Persistent Polygon fill (update vertices instead of recreating)
        fill_xy = np.column_stack([
            np.concatenate([t_draw, t_draw[::-1]]),
            np.concatenate([p_draw, np.zeros(len(p_draw))])
        ])
        if self._fun_fill is not None and self._fun_fill.axes is not None:
            self._fun_fill.set_xy(fill_xy)
        else:
            self._fun_fill = MplPolygon(
                fill_xy, closed=True, facecolor=FUNSCRIPT_COLOR,
                edgecolor='none', alpha=0.15, zorder=1)
            ax.add_patch(self._fun_fill)

        # Persistent Line2D (update data instead of recreating)
        if self._fun_line is not None and self._fun_line.axes is not None:
            self._fun_line.set_data(t_draw, p_draw)
        else:
            self._fun_line, = ax.plot(
                t_draw, p_draw, color=FUNSCRIPT_COLOR, linewidth=1.5, zorder=2)

        # Show dot markers only when zoomed in enough, colored by type
        vis_actions = actions[first_idx:last_idx + 1]
        max_dot_pts = max(10, width_px * 10 // 100)
        if n_vis <= max_dot_pts:
            # Helper: check if keyframe is fully locked (both time and pos)
            lock_t = self.kf_lock_time.get()
            lock_p = self.kf_lock_pos.get()
            def _is_fully_locked(kf_type):
                t_locked = (lock_t == "Lock all"
                            or (lock_t == "Lock beat" and kf_type != "independent")
                            or (lock_t == "Lock independent" and kf_type == "independent")
                            or lock_t == "Lock order")
                p_locked = (lock_p == "Lock all"
                            or (lock_p == "Lock beat" and kf_type != "independent")
                            or (lock_p == "Lock independent" and kf_type == "independent"))
                return t_locked and p_locked

            LOCKED_COLOR = "#565f89"  # dark muted color for fully locked kf
            def _kf_color(at_ms):
                kf_type = self._kf_types.get(at_ms)
                if _is_fully_locked(kf_type):
                    return LOCKED_COLOR
                if kf_type == "modified":
                    return KF_COLOR_MODIFIED
                elif kf_type == "independent":
                    return KF_COLOR_INDEPENDENT
                return KF_COLOR_ORIGINAL

            # Group dots by type for coloring
            by_color = {}  # color -> ([t], [p])
            selected_by_color = {}  # color -> ([t], [p])
            for ti, pi, a in zip(t_arr, p_arr, vis_actions):
                at_ms = a["at"]
                c = _kf_color(at_ms)
                if at_ms in self._selected_kf:
                    selected_by_color.setdefault(c, ([], []))
                    selected_by_color[c][0].append(ti)
                    selected_by_color[c][1].append(pi)
                else:
                    by_color.setdefault(c, ([], []))
                    by_color[c][0].append(ti)
                    by_color[c][1].append(pi)
            for c, (ct, cp) in by_color.items():
                dots, = ax.plot(ct, cp, color=c, linewidth=0,
                                marker="o", markersize=4,
                                markerfacecolor=c, markeredgewidth=0)
                self._fun_artists.append(dots)
            # Selected keyframes: larger, same color, white edge
            for c, (ct, cp) in selected_by_color.items():
                sel, = ax.plot(ct, cp, color=c, linewidth=0,
                               marker="o", markersize=6,
                               markerfacecolor=c, markeredgecolor="white",
                               markeredgewidth=1.5, alpha=0.95, zorder=10)
                self._fun_artists.append(sel)

    def _on_view_changed(self):
        """Redraw LOD elements after zoom/pan."""
        self._draw_wave_lod()
        self._draw_beat_lod()
        self._draw_fun_lod()
        self._draw_zones_lod()
        self._update_handle_visibility()
        self._update_shading()

    def _draw_markers(self):
        in_sec = _tc_to_sec(self.in_var.get())
        if in_sec is None:
            in_sec = 0
        out_sec = _tc_to_sec(self.out_var.get())

        if self.y is None:
            return
        dur = self.duration
        if out_sec is None:
            out_sec = dur

        self.in_line_w = self.ax_wave.axvline(in_sec, color=IN_COLOR, linewidth=1.5, alpha=0.9)
        self.in_line_f = self.ax_fun.axvline(in_sec, color=IN_COLOR, linewidth=1.5, alpha=0.9)
        self.in_line_z = self.ax_zones.axvline(in_sec, color=IN_COLOR, linewidth=1.5, alpha=0.9)
        self.out_line_w = self.ax_wave.axvline(out_sec, color=OUT_COLOR, linewidth=1.5, alpha=0.9)
        self.out_line_f = self.ax_fun.axvline(out_sec, color=OUT_COLOR, linewidth=1.5, alpha=0.9)
        self.out_line_z = self.ax_zones.axvline(out_sec, color=OUT_COLOR, linewidth=1.5, alpha=0.9)

        # Pin handles sitting on the top border of the waveform panel.
        # The pin's spike tip sits exactly on the top border.
        # clip_on=False so the pin body above the axes is visible.
        hy = self._handle_y()
        self.in_handle, = self.ax_wave.plot(
            [in_sec], [hy], marker=PIN_MARKER, markersize=14,
            color=IN_COLOR, markeredgecolor="white", markeredgewidth=0.8,
            alpha=0.95, zorder=20, clip_on=False)
        self.out_handle, = self.ax_wave.plot(
            [out_sec], [hy], marker=PIN_MARKER, markersize=14,
            color=OUT_COLOR, markeredgecolor="white", markeredgewidth=0.8,
            alpha=0.95, zorder=20, clip_on=False)
        self._hover_marker = None  # track which handle is hovered
        self._update_handle_visibility()


if __name__ == "__main__":
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()

    initial = sys.argv[1] if len(sys.argv) > 1 else None
    App(root, initial_file=initial)
    root.mainloop()
