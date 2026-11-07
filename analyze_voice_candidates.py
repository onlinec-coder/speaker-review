#!/usr/bin/env python3
"""
Objective (numeric-only) analysis of a long speech recording to find candidate
segments for voice-clone training samples.

IMPORTANT: nothing here identifies a speaker or an emotion. All speaker/emotion
judgements must be made by a human listening to the segments.

Needs: numpy, scipy.   Usage: python analyze_voice_candidates.py input.wav outdir
All measurements use a 10 ms hop. Time resolution of boundaries is about +/-25 ms.
"""
import sys, os, csv, json
import numpy as np
import scipy.io.wavfile as wf
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.signal import butter, sosfilt, welch

IN = sys.argv[1] if len(sys.argv) > 1 else "Husbands_audio.wav"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- parameters
H, WE, WF, NFFT = 160, 400, 800, 2048      # hop, energy win (25ms), F0 win (50ms), fft
ACTIVE_OFF = 8.0       # dB above local noise floor -> "speech-like activity"
PAUSE_OFF = 6.0        # dB above local floor -> counted as pause if sustained
PAUSE_MIN, PAUSE_PRIMARY = 12, 25          # frames: 0.12 s secondary, 0.25 s primary
SPLIT_LONG = 1500      # frames (15 s): phrases longer than this get split at dips
PAD = 12               # frames of pause kept as padding on each side (0.12 s)
V_THR = 0.55           # normalized-autocorrelation voicing threshold
OCT = 0.88             # octave-error guard: prefer shortest lag with peak >= 0.88*max
LOWBAND_HZ = 115.0     # only a descriptive split of the F0 distribution, NOT a gender rule
BINS = [(5, 10), (10, 20), (20, 40), (40, 60)]

sr, x = wf.read(IN)
assert x.ndim == 1 and x.dtype == np.int16, "expects mono int16 wav"
xf = x.astype(np.float32) / 32768.0
dur = len(x) / sr
n = (len(xf) - WF) // H + 1
tc = lambda i: i * H / sr + (WE / 2) / sr   # frame -> centre time (s)

# --------------------------------------------------- per-frame energy/peak/clip
c = np.concatenate([[0], np.cumsum(xf.astype(np.float64) ** 2)])
ix = np.arange(n) * H
db = (10 * np.log10((c[ix + WE] - c[ix]) / WE + 1e-12)).astype(np.float32)
ax = np.abs(x.astype(np.int32))
peak = np.zeros(n, np.float32); clip = np.zeros(n, np.int32)
for s in range(0, n, 20000):
    e = min(n, s + 20000)
    w = swv(ax[s * H:(e - 1) * H + WF], WF)[::H][:e - s]
    peak[s:e] = w.max(axis=1) / 32768.0
    clip[s:e] = (w >= 32760).sum(axis=1)

# ------------------------------------------------ F0 (high-passed copy), bands
xh = sosfilt(butter(4, 70, btype="highpass", fs=sr, output="sos"), xf).astype(np.float32)
hann = np.hanning(WF).astype(np.float32)
wacf = np.fft.irfft(np.abs(np.fft.rfft(hann, NFFT)) ** 2, NFFT); wacf = (wacf / wacf[0]).astype(np.float32)
lag_min, lag_max = int(sr / 450), int(sr / 70)
f0 = np.full(n, np.nan, np.float32); vstr = np.zeros(n, np.float32)
lf = np.zeros(n, np.float32); hf = np.zeros(n, np.float32)
fr = np.fft.rfftfreq(NFFT, 1 / sr); bL = fr < 300; bH = fr >= 2000
for s in range(0, n, 4000):
    e = min(n, s + 4000)
    X = np.fft.rfft(swv(xf[s * H:(e - 1) * H + WF], WF)[::H][:e - s] * hann, NFFT)
    P = X.real ** 2 + X.imag ** 2; tot = P.sum(axis=1) + 1e-12
    lf[s:e] = 10 * np.log10(P[:, bL].sum(axis=1) / tot + 1e-12)
    hf[s:e] = 10 * np.log10(P[:, bH].sum(axis=1) / tot + 1e-12)
    Xh = np.fft.rfft(swv(xh[s * H:(e - 1) * H + WF], WF)[::H][:e - s] * hann, NFFT)
    ac = np.fft.irfft(Xh.real ** 2 + Xh.imag ** 2, NFFT, axis=1)
    r = ac[:, :lag_max + 3] / (ac[:, :1] + 1e-12) / wacf[:lag_max + 3]
    seg = r[:, lag_min:lag_max + 1]; mx = seg.max(axis=1)
    lm = np.zeros_like(seg, bool); lm[:, 1:-1] = (seg[:, 1:-1] >= seg[:, :-2]) & (seg[:, 1:-1] > seg[:, 2:])
    good = lm & (seg >= OCT * mx[:, None])
    k = np.where(good.any(axis=1), good.argmax(axis=1), seg.argmax(axis=1)) + lag_min
    a = np.arange(len(k)); ok = (k > lag_min) & (k < lag_max)
    ym, y0, yp = r[a, k - 1], r[a, k], r[a, k + 1]
    den = ym - 2 * y0 + yp
    dl = np.where(np.abs(den) > 1e-9, 0.5 * (ym - yp) / np.where(np.abs(den) > 1e-9, den, 1), 0.0)
    f0[s:e] = np.where(ok, sr / (k + np.clip(dl, -1, 1)), np.nan); vstr[s:e] = r[a, k]

# --------------------------------------- local noise floor / activity / pauses
env = uniform_filter1d(db, 5)
dec = env[::5]
floor = np.interp(np.arange(n), np.arange(len(dec)) * 5,
                  percentile_filter(dec, 10, size=120, mode="nearest"))   # ~6 s window
active = env > floor + ACTIVE_OFF
voiced = active & (vstr >= V_THR) & np.isfinite(f0)
# 5-frame median smoothing of F0 over voiced neighbours
f0v = np.where(voiced, f0, np.nan)
pad = np.pad(f0v, 2, constant_values=np.nan)
with np.errstate(all="ignore"):
    f0s = np.where(voiced, np.nanmedian(swv(pad, 5), axis=1), np.nan)
voiced &= np.isfinite(f0s)
st = np.where(voiced, 12 * np.log2(np.where(voiced, f0s, 1.0)), np.nan)   # semitones re 1 Hz
low_band = voiced & (f0s < LOWBAND_HZ)

def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)       # start, end(exclusive)

ps_, pe_ = runs_of(env < floor + PAUSE_OFF)
pauses = [(a, b) for a, b in zip(ps_, pe_) if b - a >= PAUSE_MIN]
prim = [p for p in pauses if p[1] - p[0] >= PAUSE_PRIMARY]
sec = [p for p in pauses if p[1] - p[0] < PAUSE_PRIMARY]

bounds = list(prim)
def split(a, b):
    if b - a <= SPLIT_LONG: return
    cand = [p for p in sec if p[0] > a + 200 and p[1] < b - 200]
    if not cand: return
    mid = (a + b) / 2
    best = max(cand, key=lambda p: (p[1] - p[0]) - 0.002 * abs((p[0] + p[1]) / 2 - mid))
    bounds.append(best); split(a, best[0]); split(best[1], b)
edges = [0] + [p for pp in sorted(prim) for p in pp] + [n]
for a, b in zip(edges[::2], edges[1::2]): split(a, b)
bounds = sorted(set(bounds))

# phrases = regions between consecutive boundary pauses
ph = []
prev_end, prev_len = 0, 0
for (a, b) in bounds + [(n, n)]:
    if a - prev_end > 5:
        ph.append((prev_end, a, prev_len, b - a))
    prev_end, prev_len = b, b - a
phs = np.array([(p[0], p[1]) for p in ph])
P = len(ph)
ph_f0 = np.full(P, np.nan); ph_nv = np.zeros(P, int); ph_lvl = np.full(P, np.nan)
for k, (a, b, _, _) in enumerate(ph):
    m = voiced[a:b]; ph_nv[k] = m.sum()
    if m.sum() >= 15: ph_f0[k] = np.median(st[a:b][m])
    if active[a:b].sum() >= 10: ph_lvl[k] = np.median(db[a:b][active[a:b]])

# ------------------------------------------------------------- cumulative sums
cum = lambda v: np.concatenate([[0], np.cumsum(v)])
c_act, c_voi, c_low, c_clip = cum(active), cum(voiced), cum(low_band), cum(clip)
stz = np.where(voiced, st, 0.0)
c_st, c_st2 = cum(stz), cum(stz ** 2)
pow_ = 10 ** (db / 10); c_pow = cum(pow_)
clipf = lambda v, lo=0.0, hi=1.0: float(min(max(v, lo), hi))

def window_metrics(i, j):
    a0, a1 = ph[i][0], ph[j][1]
    pa, pb = ph[i][2], ph[j][3]          # pause lengths before/after
    a = a0 - min(PAD, ph[i][2] // 2); b = a1 + min(PAD, ph[j][3] // 2)
    a = max(a, 0); b = min(b, n)
    nf = b - a
    d_s = nf * H / sr
    na = c_act[b] - c_act[a]; nv = c_voi[b] - c_voi[a]
    seg_db = db[a:b]; seg_env = env[a:b]
    if na < 20 or nv < 30: return None
    act_lvl = float(np.median(seg_db[active[a:b]]))
    rms_all = 10 * np.log10((c_pow[b] - c_pow[a]) / nf + 1e-12)
    noise = float(np.percentile(seg_env, 10))
    snr = act_lvl - noise
    pk = 20 * np.log10(peak[a:b].max() + 1e-9)
    nclip = int(c_clip[b] - c_clip[a])
    m = voiced[a:b]; ff = f0s[a:b][m]
    mean_st = (c_st[b] - c_st[a]) / nv
    std_st = float(np.sqrt(max((c_st2[b] - c_st2[a]) / nv - mean_st ** 2, 0)))
    f_med = float(np.median(ff)); f_p10, f_p90 = np.percentile(ff, [10, 90])
    dyn_std = float(np.std(seg_db[active[a:b]]))
    lowshare = (c_low[b] - c_low[a]) / nv
    mix = float(min(lowshare, 1 - lowshare) / 0.5)           # 0 = one band, 1 = even mix
    pm = ph_f0[i:j + 1]; pm = pm[np.isfinite(pm)]
    prange = float(pm.max() - pm.min()) if len(pm) >= 2 else 0.0
    lv = ph_lvl[i:j + 1]; lv = lv[np.isfinite(lv)]
    lrange = float(lv.max() - lv.min()) if len(lv) >= 2 else 0.0
    npause = j - i
    rate = npause / d_s * 10
    lpause = max([ph[k][3] for k in range(i, j)], default=0) * H / sr
    voic_act = nv / na
    # ---- objective quality score (documented in the txt report)
    q_snr = clipf((snr - 5) / 20)
    q_voi = clipf((voic_act - 0.35) / 0.40)
    q_f0 = 1 - max(clipf(prange / 8), mix)
    dist = max((-26) - act_lvl, act_lvl - (-14), 0)   # dB outside the -26..-14 dBFS target band
    q_lvl = clipf(1 - dist / 12)
    q_pau = clipf(rate) if rate < 1 else (1.0 if rate <= 5 else clipf(1 - (rate - 5) / 5))
    score = 100 * (0.35 * q_snr + 0.20 * q_voi + 0.20 * q_f0 + 0.15 * q_lvl + 0.10 * q_pau)
    if nclip: score -= 5
    warns = []
    if mix > 0.5: warns.append("F0 distribution spans two bands (possible different voices/register or pitch-tracking errors)")
    if prange > 5: warns.append("phrase-median F0 differs by >5 semitones between phrases")
    if lrange > 12: warns.append("phrase level differs by >12 dB (possible speaker/mic change)")
    if snr < 12: warns.append("low estimated SNR (<12 dB)")
    if nclip: warns.append("%d clipped samples" % nclip)
    if voic_act < 0.45: warns.append("low voiced share (unvoiced/whisper-like or noisy)")
    if rate > 5: warns.append("many short pauses")
    if rate < 0.5 and d_s > 15: warns.append("almost no pauses")
    band = "low(<%d Hz)" % LOWBAND_HZ if lowshare > 0.75 else ("high(>=%d Hz)" % LOWBAND_HZ if lowshare < 0.25 else "mixed")
    return dict(start=a * H / sr + (WE / 2) / sr - 0.0, end=b * H / sr + (WE / 2) / sr, dur=d_s,
                rms_db=rms_all, act_lvl=act_lvl, peak_db=pk, clip=nclip, f0_med=f_med, f0_p10=float(f_p10),
                f0_p90=float(f_p90), f0_std_st=std_st, voiced_pct=100 * nv / nf, voiced_act=voic_act,
                active_pct=100 * na / nf, noise_db=noise, snr=snr, dyn_std=dyn_std, f0_band=band,
                low_share=lowshare, mix=mix, f0_prange=prange, lvl_range=lrange, pauses=npause,
                pause_rate=rate, longest_pause=lpause, score=score, warn="; ".join(warns), i=i, j=j)

# ------------------------------------------------------------ candidate windows
cands = []
for i in range(P):
    for j in range(i, P):
        span = (ph[j][1] - ph[i][0]) * H / sr
        if span > 61: break
        if span < 5: continue
        m = window_metrics(i, j)
        if m: cands.append(m)
for c_ in cands:
    c_["bin"] = next((f"{lo}-{hi}s" for lo, hi in BINS if lo <= c_["dur"] < hi + 0.001), "")

def nms(items, k, key):
    out = []
    for it in sorted(items, key=key, reverse=True):
        if all(it["end"] <= o["start"] or it["start"] >= o["end"] for o in out):
            out.append(it)
            if len(out) == k: break
    return out

fmt = lambda t: "%02d:%06.3f" % (int(t // 60), t % 60)
arr = lambda k: np.array([c_[k] for c_ in cands])
pctl = lambda k, p: float(np.percentile(arr(k), p))

pool = [c_ for c_ in cands if c_["clip"] == 0]
top_all = nms(pool, 30, lambda c_: c_["score"])
top_bin = {b: nms([c_ for c_ in pool if c_["bin"] == b], 10, lambda c_: c_["score"]) for b in [f"{lo}-{hi}s" for lo, hi in BINS]}

p40f, p40d = pctl("f0_std_st", 40), pctl("dyn_std", 40)
p75f, p75d = pctl("f0_std_st", 75), pctl("dyn_std", 75)
med_score = pctl("score", 50)
neutral = nms([c_ for c_ in pool if c_["f0_std_st"] <= p40f and c_["dyn_std"] <= p40d and c_["mix"] < 0.3
               and c_["dur"] <= 40 and c_["snr"] >= 10], 10, lambda c_: c_["score"])
zf = lambda c_: (c_["f0_std_st"] - arr("f0_std_st").mean()) / arr("f0_std_st").std() + \
                (c_["dyn_std"] - arr("dyn_std").mean()) / arr("dyn_std").std()
expressive = nms([c_ for c_ in pool if (c_["f0_std_st"] >= p75f or c_["dyn_std"] >= p75d) and c_["score"] >= med_score
                  and c_["mix"] < 0.3 and c_["f0_prange"] <= 8 and c_["dur"] <= 40], 5, zf)
p15l, p25l = pctl("act_lvl", 15), pctl("act_lvl", 25)
soft = nms([c_ for c_ in pool if c_["act_lvl"] <= p15l and c_["voiced_act"] >= 0.30 and c_["mix"] < 0.5
            and c_["dur"] <= 40], 5, lambda c_: c_["score"])
whisperlike = nms([c_ for c_ in pool if c_["act_lvl"] <= p25l and c_["voiced_act"] < 0.35], 5, lambda c_: c_["score"])
longer = nms([c_ for c_ in pool if 40 <= c_["dur"] <= 60], 5, lambda c_: c_["score"])

# ------------------------------------------------------- chunk analysis (10 s)
CH = 1000
rows = []
for s in range(0, n, CH):
    e = min(n, s + CH)
    if e - s < 100: break
    ac_ = active[s:e]; vo = voiced[s:e]
    ff = f0s[s:e][vo]
    ss = st[s:e][vo]
    sa, sb = int(s * H), int(min(len(x), (e - 1) * H + WF))
    seg_ = xf[sa:sb]
    rows.append(dict(
        start_s=round(tc(s), 3), end_s=round(tc(e - 1), 3),
        rms_dbfs=round(float(20 * np.log10(np.sqrt((seg_.astype(np.float64) ** 2).mean()) + 1e-9)), 2),
        active_median_dbfs=round(float(np.median(db[s:e][ac_])), 2) if ac_.sum() > 5 else "",
        peak_dbfs=round(float(20 * np.log10(peak[s:e].max() + 1e-9)), 2),
        clipped_samples=int(clip[s:e].sum()),
        noise_floor_est_dbfs=round(float(np.percentile(env[s:e], 10)), 2),
        local_floor_dbfs=round(float(floor[s:e].mean()), 2),
        activity_pct=round(100 * ac_.mean(), 1), voiced_pct=round(100 * vo.mean(), 1),
        f0_median_hz=round(float(np.median(ff)), 1) if len(ff) >= 20 else "",
        f0_p10_hz=round(float(np.percentile(ff, 10)), 1) if len(ff) >= 20 else "",
        f0_p90_hz=round(float(np.percentile(ff, 90)), 1) if len(ff) >= 20 else "",
        f0_std_semitones=round(float(ss.std()), 2) if len(ss) >= 20 else "",
        f0_lowband_share=round(float(low_band[s:e].sum() / max(vo.sum(), 1)), 2) if len(ff) >= 20 else "",
        lf_below300_share_db=round(float(lf[s:e][ac_].mean()), 1) if ac_.sum() > 5 else "",
        hf_above2k_share_db=round(float(hf[s:e][ac_].mean()), 1) if ac_.sum() > 5 else ""))

def write_csv(path, rows_, cols=None):
    cols = cols or list(rows_[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r_ in rows_: w.writerow({k: r_.get(k, "") for k in cols})

write_csv(os.path.join(OUT, "vivienne_audio_analysis.csv"), rows)

# candidate CSV: top 40 non-overlapping per duration bin + category tags
tag = {}
for name, lst in [("neutral", neutral), ("expressive", expressive), ("soft", soft), ("whisper_like_proxy", whisperlike), ("long", longer)]:
    for r_ in lst: tag.setdefault((round(r_["start"], 3), round(r_["end"], 3)), []).append(name)
out_rows = []
for b in [f"{lo}-{hi}s" for lo, hi in BINS]:
    for rk, r_ in enumerate(nms([c_ for c_ in pool if c_["bin"] == b], 40, lambda c_: c_["score"]), 1):
        out_rows.append(dict(
            duration_bin=b, rank_in_bin=rk, start_s=round(r_["start"], 3), end_s=round(r_["end"], 3),
            start=fmt(r_["start"]), end=fmt(r_["end"]), duration_s=round(r_["dur"], 2),
            quality_score=round(r_["score"], 1), rms_dbfs=round(r_["rms_db"], 1), active_median_dbfs=round(r_["act_lvl"], 1),
            peak_dbfs=round(r_["peak_db"], 1), clipped_samples=r_["clip"], f0_median_hz=round(r_["f0_med"], 0),
            f0_p10_hz=round(r_["f0_p10"], 0), f0_p90_hz=round(r_["f0_p90"], 0), f0_std_semitones=round(r_["f0_std_st"], 2),
            f0_band=r_["f0_band"], voiced_pct=round(r_["voiced_pct"], 1), voiced_of_active=round(r_["voiced_act"], 2),
            activity_pct=round(r_["active_pct"], 1), noise_floor_est_dbfs=round(r_["noise_db"], 1), snr_est_db=round(r_["snr"], 1),
            level_std_db=round(r_["dyn_std"], 1), pauses=r_["pauses"], longest_pause_s=round(r_["longest_pause"], 2),
            categories=",".join(tag.get((round(r_["start"], 3), round(r_["end"], 3)), [])), warnings=r_["warn"],
            speaker_verified="NO - verify by listening"))
write_csv(os.path.join(OUT, "vivienne_candidate_segments.csv"), out_rows)

# ------------------------------------------------------------ global summary
allrms = 10 * np.log10(c_pow[-1] / n)
loud = np.repeat(db > -25, H)[:len(xf)]
fw, Pw = welch(xf[:len(loud)][loud], sr, nperseg=2048)
cs = np.cumsum(Pw) / Pw.sum()
roll = {p: float(fw[np.searchsorted(cs, p / 100)]) for p in (90, 95, 99, 99.9)}
pm_ = env < floor + PAUSE_OFF
summary = dict(
    duration_s=dur, sample_rate=sr, frames=n, overall_rms_dbfs=float(allrms),
    peak_dbfs=float(20 * np.log10(np.abs(x).max() / 32768)), clipped_samples=int((ax >= 32760).sum()),
    frame_db_pctiles={str(p): float(np.percentile(db, p)) for p in (1, 5, 10, 50, 90, 99)},
    local_floor_median_dbfs=float(np.median(floor)), local_floor_p5_p95=[float(np.percentile(floor, 5)), float(np.percentile(floor, 95))],
    speech_minus_floor_db_median=float(np.median(db[active]) - np.median(floor)),
    active_pct=float(100 * active.mean()), voiced_pct_of_all=float(100 * voiced.mean()),
    voiced_pct_of_active=float(100 * voiced.sum() / active.sum()),
    f0_pctiles_5_25_50_75_95=[float(v) for v in np.percentile(f0s[voiced], [5, 25, 50, 75, 95])],
    f0_lowband_share=float(low_band.sum() / voiced.sum()),
    pauses_ge_012s=len(pauses), pauses_ge_025s=len(prim), phrases=P,
    pause_lf_share_db=float(lf[pm_].mean()), speech_lf_share_db=float(lf[active].mean()),
    pause_hf_share_db=float(hf[pm_].mean()), speech_hf_share_db=float(hf[active].mean()),
    energy_rolloff_hz=roll, n_candidates=len(cands),
    cand_per_bin={b: int(sum(1 for c_ in cands if c_["bin"] == b)) for b in [f"{lo}-{hi}s" for lo, hi in BINS]},
    score_pctiles={str(p): pctl("score", p) for p in (10, 50, 90, 99)},
    p40_f0std=p40f, p75_f0std=p75f, p40_lvlstd=p40d, p75_lvlstd=p75d, p15_level=p15l, p25_level=p25l)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

# ------------------------------------------------------------- recommendations
def block(title, lst, extra=""):
    L = ["", "=" * 78, title, "=" * 78, extra] if extra else ["", "=" * 78, title, "=" * 78]
    if not lst: L.append("(no segment met the criteria)"); return L
    for k, r_ in enumerate(lst, 1):
        L.append(f"{k:>2}. {fmt(r_['start'])} -> {fmt(r_['end'])}  dur {r_['dur']:5.1f}s  score {r_['score']:5.1f}  "
                 f"level {r_['act_lvl']:6.1f} dBFS  peak {r_['peak_db']:5.1f}  SNR~{r_['snr']:4.1f} dB  "
                 f"F0 med {r_['f0_med']:4.0f} Hz [{r_['f0_p10']:.0f}-{r_['f0_p90']:.0f}]  F0 std {r_['f0_std_st']:.1f} st  "
                 f"voiced {r_['voiced_pct']:.0f}%  band {r_['f0_band']}")
        L.append("      warnings: " + (r_["warn"] if r_["warn"] else "none from automatic checks") +
                 "   | speaker: NOT VERIFIED")
    return L

L = ["VOICE-CLONE CANDIDATE SEGMENTS - OBJECTIVE ANALYSIS (no speaker or emotion identification)",
     f"Source: {os.path.basename(IN)}  duration {fmt(dur)} ({dur:.2f}s)  {sr} Hz mono 16-bit",
     "Timestamps are MM:SS.mmm from file start; boundary precision about +/-25 ms.",
     "Extract example:  ffmpeg -ss <start_s> -to <end_s> -i Husbands_audio.wav -c copy sample.wav",
     "(start_s / end_s in seconds are in vivienne_candidate_segments.csv)",
     "", "QUALITY SCORE (0-100, objective, reproducible):",
     "  35% estimated SNR  (active-frame median level minus 10th-percentile frame level; 5 dB ->0, 25 dB ->1)",
     "  20% voiced share of active frames (0.35 ->0, 0.75 ->1)",
     "  20% F0 consistency (1 - max(phrase-median F0 range/8 semitones, F0-band mixing))",
     "  15% level (active median within -26..-14 dBFS = 1, linear falloff 12 dB)",
     "  10% pause naturalness (1-5 pauses per 10 s = 1)",
     "  -5 points if any clipped sample.  It does NOT measure speaker identity, emotion or music/SFX content."]
L += block("A. TOP 15 OVERALL (non-overlapping, any duration 5-60 s)", top_all[:15])
for b in top_bin:
    L += block(f"B. BEST IN DURATION BIN {b}", top_bin[b][:5])
L += block("C1. 10 BEST 'NEUTRAL-LIKE' CANDIDATES (low F0 variation and low level variation; NOT verified as neutral speech)", neutral)
L += block("C2. 5 BEST 'EXPRESSIVE-LIKE' CANDIDATES (high F0/loudness variation; variation is not proof of emotion)", expressive)
L += block("C3. 5 BEST SOFT / LOW-LEVEL CANDIDATES (lowest 15% median level, still voiced)", soft)
L += block("C3b. 5 LOW-LEVEL + LOW-VOICING CANDIDATES (waveform proxy for possible whisper/breathy/unvoiced material - needs listening)", whisperlike)
L += block("C4. 5 BEST LONGER CONTINUOUS CANDIDATES (40-60 s)", longer)
open(os.path.join(OUT, "vivienne_sample_recommendations.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(json.dumps(summary, indent=1))
