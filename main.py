#!/usr/bin/env python3
"""
CSE-307: Operating Systems — Term Paper (Track 1)
Learning-Augmented Page Replacement:
    Classical Algorithms Meet Adaptive Prediction

Author : Faisal Ahmed
Course : CSE-307 Operating Systems, Spring 2026

This script implements and compares four page-replacement policies
on a synthetic workload that undergoes a deliberate distribution
shift partway through:

    1. FIFO   — First-In, First-Out
    2. LRU    — Least Recently Used
    3. OPT    — Optimal / Belady's algorithm
    4. Learned — Decision-Tree classifier trained on OPT's decisions

Usage:
    python main.py                  # run with defaults
    python main.py --frames 4      # change frame count
    python main.py --refs 20000    # change total reference count
"""

import os
import sys
import bisect
import argparse
from collections import deque, OrderedDict, defaultdict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt

# ─────────────────────────────── defaults ────────────────────────────
SEED         = 42
TOTAL_REFS   = 10_000
NUM_PAGES    = 50       # distinct virtual pages
NUM_FRAMES   = 8        # physical frames available
WORKING_SET  = 10       # hot pages in the locality phase
TRAIN_SIZE   = 8_000    # training-trace length
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results")


# ====================================================================
#  1.  WORKLOAD  GENERATION
# ====================================================================

def generate_locality_trace(length, num_pages=NUM_PAGES,
                            ws=WORKING_SET, seed=SEED):
    """Locality-heavy page-access trace.

    ~60 % sequential scan through a small working set,
    ~30 % random within the working set (temporal locality),
    ~10 % outlier references.
    """
    rng = np.random.RandomState(seed)
    working_set = list(range(ws))
    trace, idx = [], 0
    for _ in range(length):
        r = rng.random()
        if r < 0.60:
            trace.append(working_set[idx % ws]); idx += 1
        elif r < 0.90:
            trace.append(rng.choice(working_set))
        else:
            trace.append(rng.randint(0, num_pages))
    return trace


def generate_random_bursty_trace(length, num_pages=NUM_PAGES, seed=SEED):
    """Random / bursty trace — low spatial and temporal locality."""
    rng = np.random.RandomState(seed + 1)
    trace = []
    while len(trace) < length:
        if rng.random() < 0.25:
            page  = rng.randint(0, num_pages)
            burst = rng.randint(2, 6)
            trace.extend([page] * burst)
        else:
            trace.append(rng.randint(0, num_pages))
    return trace[:length]


def generate_shifted_workload(total=TOTAL_REFS, num_pages=NUM_PAGES):
    """First half locality-heavy, second half random / bursty."""
    half   = total // 2
    phase1 = generate_locality_trace(half, num_pages, seed=SEED)
    phase2 = generate_random_bursty_trace(total - half, num_pages,
                                          seed=SEED)
    return phase1 + phase2, half


# ====================================================================
#  2.  CLASSICAL  PAGE-REPLACEMENT  ALGORITHMS
# ====================================================================

def simulate_fifo(refs, nf):
    """First-In First-Out page replacement."""
    q, s = deque(), set()
    faults, fault_log = 0, []
    for r in refs:
        if r in s:
            fault_log.append(0)
        else:
            faults += 1
            fault_log.append(1)
            if len(q) >= nf:
                s.discard(q.popleft())
            q.append(r); s.add(r)
    return faults, len(refs) - faults, fault_log


def simulate_lru(refs, nf):
    """Least Recently Used page replacement."""
    cache = OrderedDict()
    faults, fault_log = 0, []
    for r in refs:
        if r in cache:
            cache.move_to_end(r)
            fault_log.append(0)
        else:
            faults += 1
            fault_log.append(1)
            if len(cache) >= nf:
                cache.popitem(last=False)
            cache[r] = True
    return faults, len(refs) - faults, fault_log


def simulate_optimal(refs, nf):
    """Optimal (Belady's MIN) — requires full knowledge of future."""
    n = len(refs)
    # Pre-index future positions for O(log n) lookup
    page_pos = defaultdict(list)
    for i, r in enumerate(refs):
        page_pos[r].append(i)

    def _next_use(page, cur):
        lst = page_pos[page]
        j   = bisect.bisect_right(lst, cur)
        return lst[j] if j < len(lst) else float("inf")

    frames = []
    faults, fault_log = 0, []
    for i, r in enumerate(refs):
        if r in frames:
            fault_log.append(0)
        else:
            faults += 1
            fault_log.append(1)
            if len(frames) < nf:
                frames.append(r)
            else:
                best_j, best_nu = 0, -1
                for j, f in enumerate(frames):
                    nu = _next_use(f, i)
                    if nu == float("inf"):
                        best_j = j; break
                    if nu > best_nu:
                        best_nu = nu; best_j = j
                frames[best_j] = r
    return faults, n - faults, fault_log


# ====================================================================
#  3.  LEARNED  (ADAPTIVE)  PAGE  REPLACEMENT
# ====================================================================

def _build_training_data(refs, nf):
    """Mirror Optimal's run and record per-frame feature vectors.

    Features (per cached page at each eviction event):
        recency   — time steps since the page was last accessed
        frequency — cumulative access count of the page
        age       — time steps since the page was loaded into a frame

    Label:
        1 if Optimal chose this page as the eviction victim, else 0.
    """
    n = len(refs)
    page_pos = defaultdict(list)
    for i, r in enumerate(refs):
        page_pos[r].append(i)

    def _next_use(page, cur):
        lst = page_pos[page]
        j   = bisect.bisect_right(lst, cur)
        return lst[j] if j < len(lst) else float("inf")

    frames      = []
    last_access = {}
    freq        = defaultdict(int)
    insert_time = {}

    X, y = [], []

    for i, r in enumerate(refs):
        freq[r] += 1
        if r in frames:
            last_access[r] = i
        else:
            if len(frames) < nf:
                frames.append(r)
                last_access[r]  = i
                insert_time[r]  = i
            else:
                # ---- Optimal eviction decision ----
                best_j, best_nu = 0, -1
                for j, f in enumerate(frames):
                    nu = _next_use(f, i)
                    if nu == float("inf"):
                        best_j = j; best_nu = float("inf"); break
                    if nu > best_nu:
                        best_nu = nu; best_j = j

                # Record features for every frame entry
                for j, f in enumerate(frames):
                    recency   = i - last_access.get(f, 0)
                    frequency = freq.get(f, 0)
                    age       = i - insert_time.get(f, 0)
                    X.append([recency, frequency, age])
                    y.append(1 if j == best_j else 0)

                # Perform eviction
                frames[best_j]          = r
                last_access[r]          = i
                insert_time[r]          = i

        last_access[r] = i          # always update

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def train_classifier(train_refs, nf, seed=SEED):
    """Train a DecisionTreeClassifier on Optimal's eviction behaviour."""
    X, y = _build_training_data(train_refs, nf)
    clf  = DecisionTreeClassifier(max_depth=10,
                                  min_samples_leaf=20,
                                  class_weight="balanced",
                                  random_state=seed)
    clf.fit(X, y)

    acc = clf.score(X, y)
    evict_ratio = y.mean()
    print(f"  Classifier training accuracy : {acc:.4f}")
    print(f"  Training samples             : {len(y)}")
    print(f"  Eviction-class ratio         : {evict_ratio:.4f}")
    return clf


def simulate_learned(refs, nf, clf):
    """Page replacement guided by the trained Decision-Tree."""
    frames      = []
    last_access = {}
    freq        = defaultdict(int)
    insert_time = {}
    faults, fault_log = 0, []

    # Identify the column index for the "evict" class (label 1)
    evict_col = list(clf.classes_).index(1) if 1 in clf.classes_ else -1

    for i, r in enumerate(refs):
        freq[r] += 1
        if r in frames:
            last_access[r] = i
            fault_log.append(0)
        else:
            faults += 1
            fault_log.append(1)
            if len(frames) < nf:
                frames.append(r)
                last_access[r]  = i
                insert_time[r]  = i
            else:
                # Build feature matrix for the current frame contents
                feats = []
                for f in frames:
                    recency   = i - last_access.get(f, 0)
                    frequency = freq.get(f, 0)
                    age       = i - insert_time.get(f, 0)
                    feats.append([recency, frequency, age])

                if evict_col >= 0:
                    probs   = clf.predict_proba(np.array(feats))
                    scores  = probs[:, evict_col]
                    victim  = int(np.argmax(scores))
                else:
                    # fallback: evict the page with highest recency (LRU)
                    victim = int(np.argmax([f[0] for f in feats]))

                frames[victim]          = r
                last_access[r]          = i
                insert_time[r]          = i

        last_access[r] = i
    return faults, len(refs) - faults, fault_log


# ====================================================================
#  4.  EXPERIMENT  RUNNER
# ====================================================================

def run_experiment(num_frames=NUM_FRAMES, total_refs=TOTAL_REFS):
    """Execute all experiments and write results + plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sep = "=" * 64
    print(sep)
    print("  CSE-307 Term Paper — Track 1: Learned Page Replacement")
    print(sep)
    print(f"  Frames       : {num_frames}")
    print(f"  Total refs   : {total_refs}")
    print(f"  Page universe: {NUM_PAGES}")
    print(f"  Seed         : {SEED}")
    print()

    # ── Step 1: train the classifier on a locality-only trace ──
    print("[1/4] Generating training trace (locality-heavy only) ...")
    train_trace = generate_locality_trace(TRAIN_SIZE, seed=SEED + 100)

    print("[2/4] Training Decision-Tree classifier on OPT decisions ...")
    clf = train_classifier(train_trace, num_frames)
    tree_text = export_text(clf,
                            feature_names=["recency", "frequency", "age"],
                            max_depth=4)
    with open(os.path.join(RESULTS_DIR, "decision_tree_rules.txt"), "w") as fp:
        fp.write(tree_text)
    print(f"  Top-level tree rules saved.\n")

    # ── Step 2: generate test workload with shift ──
    print("[3/4] Generating test workload with distribution shift ...")
    global TOTAL_REFS
    TOTAL_REFS = total_refs
    test_trace, shift_idx = generate_shifted_workload(total_refs)
    print(f"  Phase 1 (locality)     : refs 0 – {shift_idx - 1}")
    print(f"  Phase 2 (random/bursty): refs {shift_idx} – {len(test_trace) - 1}\n")

    # ── Step 3: simulate all algorithms ──
    print("[4/4] Running simulations ...")
    algorithms = OrderedDict([
        ("FIFO",    lambda r, n: simulate_fifo(r, n)),
        ("LRU",     lambda r, n: simulate_lru(r, n)),
        ("Optimal", lambda r, n: simulate_optimal(r, n)),
        ("Learned", lambda r, n: simulate_learned(r, n, clf)),
    ])

    results = {}
    for name, algo in algorithms.items():
        faults, hits, fl = algo(test_trace, num_frames)
        fl_arr  = np.array(fl)
        p1f     = int(fl_arr[:shift_idx].sum())
        p2f     = int(fl_arr[shift_idx:].sum())
        p1_tot  = shift_idx
        p2_tot  = len(test_trace) - shift_idx

        results[name] = dict(
            total_faults    = faults,
            total_hits      = hits,
            hit_ratio       = hits / len(test_trace),
            phase1_faults   = p1f,
            phase1_hits     = p1_tot - p1f,
            phase1_hit_ratio= (p1_tot - p1f) / p1_tot,
            phase2_faults   = p2f,
            phase2_hits     = p2_tot - p2f,
            phase2_hit_ratio= (p2_tot - p2f) / p2_tot,
            fault_log       = fl,
        )

    # ── Print summary table ──
    print()
    hdr = (f"  {'Algorithm':10s} | {'Faults':>7s} | {'Hit Ratio':>9s} | "
           f"{'Phase1 HR':>9s} | {'Phase2 HR':>9s} | {'Degradation':>11s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in algorithms:
        r = results[name]
        deg = r["phase1_hit_ratio"] - r["phase2_hit_ratio"]
        print(f"  {name:10s} | {r['total_faults']:7d} | "
              f"{r['hit_ratio']:9.4f} | {r['phase1_hit_ratio']:9.4f} | "
              f"{r['phase2_hit_ratio']:9.4f} | {deg:+11.4f}")
    print()

    # ── Save CSV ──
    rows = []
    for name in algorithms:
        r = results[name]
        rows.append({
            "Algorithm":           name,
            "Total Faults":        r["total_faults"],
            "Total Hit Ratio":     round(r["hit_ratio"], 4),
            "Phase 1 Faults":      r["phase1_faults"],
            "Phase 1 Hit Ratio":   round(r["phase1_hit_ratio"], 4),
            "Phase 2 Faults":      r["phase2_faults"],
            "Phase 2 Hit Ratio":   round(r["phase2_hit_ratio"], 4),
            "Degradation (ΔHR)":   round(r["phase1_hit_ratio"]
                                         - r["phase2_hit_ratio"], 4),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Summary table → {os.path.relpath(csv_path)}")

    # ── Generate plots ──
    _generate_plots(results, algorithms, shift_idx, len(test_trace))

    print(f"\n  All outputs saved to {os.path.relpath(RESULTS_DIR)}/")
    print(sep)
    return results, df


# ====================================================================
#  5.  PLOTTING
# ====================================================================

PALETTE = {
    "FIFO": "#e74c3c", "LRU": "#3498db",
    "Optimal": "#2ecc71", "Learned": "#9b59b6",
}


def _generate_plots(results, algos, shift_idx, total):
    names = list(algos.keys())

    # ── Plot 1: Total page faults ──
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [results[a]["total_faults"] for a in names]
    bars = ax.bar(names, vals,
                  color=[PALETTE[a] for a in names],
                  edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Total Page Faults", fontsize=12)
    ax.set_title("Total Page Faults by Algorithm", fontsize=14,
                 fontweight="bold")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "total_faults.png"), dpi=150)
    plt.close()

    # ── Plot 2: Hit ratio by phase (grouped bar) ──
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names)); w = 0.35
    p1 = [results[a]["phase1_hit_ratio"] for a in names]
    p2 = [results[a]["phase2_hit_ratio"] for a in names]
    b1 = ax.bar(x - w / 2, p1, w,
                label="Phase 1 — Locality-heavy", color="#5dade2")
    b2 = ax.bar(x + w / 2, p2, w,
                label="Phase 2 — Random / Bursty", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Hit Ratio", fontsize=12)
    ax.set_title("Hit Ratio Before vs. After Workload Shift",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hit_ratio_by_phase.png"), dpi=150)
    plt.close()

    # ── Plot 3: Rolling fault rate over time ──
    fig, ax = plt.subplots(figsize=(12, 5))
    window = 200
    for a in names:
        fl = np.array(results[a]["fault_log"], dtype=float)
        smooth = np.convolve(fl, np.ones(window) / window, mode="valid")
        ax.plot(range(len(smooth)), smooth,
                label=a, color=PALETTE[a], linewidth=1.4, alpha=0.85)
    ax.axvline(x=shift_idx, color="gray", linestyle="--",
               linewidth=1.5, label="Workload Shift")
    ax.set_xlabel("Page Reference Index", fontsize=12)
    ax.set_ylabel(f"Fault Rate (rolling window = {window})", fontsize=12)
    ax.set_title("Fault Rate Over Time — Distribution Shift Effect",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fault_rate_timeline.png"), dpi=150)
    plt.close()

    # ── Plot 4: Degradation bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    degs = [results[a]["phase1_hit_ratio"] - results[a]["phase2_hit_ratio"]
            for a in names]
    bars = ax.bar(names, degs,
                  color=[PALETTE[a] for a in names],
                  edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Hit Ratio Drop  (Phase 1 − Phase 2)", fontsize=12)
    ax.set_title("Performance Degradation After Workload Shift",
                 fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "degradation.png"), dpi=150)
    plt.close()

    print("  Plots saved.")


# ====================================================================
#  6.  CLI  ENTRY  POINT
# ====================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Learned Page Replacement — CSE-307 Term Paper")
    ap.add_argument("--frames", type=int, default=NUM_FRAMES,
                    help="Number of physical page frames (default: 8)")
    ap.add_argument("--refs",   type=int, default=TOTAL_REFS,
                    help="Total page references in test trace (default: 10000)")
    ap.add_argument("--seed",   type=int, default=SEED,
                    help="Random seed (default: 42)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    np.random.seed(args.seed)
    run_experiment(num_frames=args.frames, total_refs=args.refs)
