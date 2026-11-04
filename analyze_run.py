"""Summarise a tep3.py run log: where the overlay was lost, why, and how fast it came back.

usage: python analyze_run.py [runs/<timestamp>]     (defaults to the latest run)
"""
import csv
import glob
import os
import sys

import numpy as np

SHOWN = ("track", "detect", "coast")

run_dir = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(os.path.join("runs", "*")), key=os.path.getmtime)
rows = list(csv.DictReader(open(os.path.join(run_dir, "log.csv"))))

if not rows:
    print("empty log")
    sys.exit()

state = np.array([r["state"] for r in rows])
fps = np.array([float(r["fps"]) for r in rows])
proc = np.array([float(r["proc_ms"]) for r in rows])
det_ms = np.array([float(r["detect_ms"]) for r in rows])
motion_ok = np.array([int(r["motion_ok"]) for r in rows]).astype(bool)
tx = np.array([float(r["tx"]) if r["tx"] else np.nan for r in rows])
ty = np.array([float(r["ty"]) if r["ty"] else np.nan for r in rows])
rot = np.array([float(r["rot_deg"]) if r["rot_deg"] else np.nan for r in rows])
speed = np.hypot(tx, ty)
shown = np.isin(state, SHOWN)
duration = float(rows[-1]["t"])

print(f"run: {run_dir}")
print(f"{len(rows)} frames in {duration:.1f}s ({len(rows) / max(duration, 1e-6):.1f} fps delivered)")
print()
print("time per state:")
for s in ("track", "detect", "coast", "offscreen", "lost"):
    print(f"  {s:9s} {100 * np.mean(state == s):5.1f}%")
print(f"  overlay shown {100 * shown.mean():5.1f}%")
print()
print(f"processing ms/frame: median {np.median(proc):.0f}, p90 {np.percentile(proc, 90):.0f}, max {proc.max():.0f}")
print(f"  frames that ran SIFT: {100 * np.mean(det_ms > 0):.0f}%, SIFT ms when it ran: median "
      f"{np.median(det_ms[det_ms > 0]) if (det_ms > 0).any() else 0:.0f}")
print(f"camera motion measured on {100 * motion_ok.mean():.0f}% of frames; "
      f"camera speed px/frame median {np.nanmedian(speed):.1f}, p95 {np.nanpercentile(speed, 95):.1f}")
print()

# episodes where the overlay disappeared
episodes = []
i = 0
while i < len(rows):
    if not shown[i] and i > 0 and shown[i - 1]:
        j = i
        while j < len(rows) and not shown[j]:
            j += 1
        k = max(0, i - 3)
        episodes.append(dict(
            start=int(rows[i]["frame"]), frames=j - i, recovered=j < len(rows),
            states=sorted(set(state[i:j])),
            speed_before=np.nanmax(speed[k:i + 1]) if np.isfinite(speed[k:i + 1]).any() else np.nan,
            rot_before=np.nanmax(np.abs(rot[k:i + 1])) if np.isfinite(rot[k:i + 1]).any() else np.nan,
            motion_failed=bool((~motion_ok[k:i + 1]).any()),
        ))
        i = j
    else:
        i += 1

print(f"overlay disappeared {len(episodes)} times")
if episodes:
    lens = np.array([e["frames"] for e in episodes])
    print(f"  gap length frames: median {np.median(lens):.0f}, max {lens.max()}; "
          f"{np.mean([e['motion_failed'] for e in episodes]) * 100:.0f}% had camera-motion failure just before")
    print(f"  {'frame':>6s} {'gap':>5s} {'speed':>6s} {'rot':>5s} {'motion':>7s}  states during gap")
    for e in sorted(episodes, key=lambda e: -e["frames"])[:15]:
        print(f"  {e['start']:6d} {e['frames']:5d} {e['speed_before']:6.0f} {e['rot_before']:5.1f} "
              f"{'FAILED' if e['motion_failed'] else 'ok':>7s}  {','.join(e['states'])}"
              f"{'' if e['recovered'] else ' (never recovered)'}")
