from __future__ import annotations
import math
from dataclasses import dataclass
from .melody_repr import MelodyRep


@dataclass
class MatchResult:
    cost: float
    end_j: int


def _interval_cost(a: int, b: int) -> float:
    return abs(a - b) / 8.0  # normalize on [-4..+4]


def _contour_cost(a: int, b: int) -> float:
    return 0.0 if a == b else 1.0


def _timing_ratio_cost(q_prev: float, q: float, s_prev: float, s: float) -> float:
    rq = q / max(1e-6, q_prev)
    rs = s / max(1e-6, s_prev)
    return abs(math.log(max(1e-6, rq)) - math.log(max(1e-6, rs)))


def dp_distance(
    query: MelodyRep,
    song: MelodyRep,
    w_int: float = 1.0,
    w_cont: float = 0.7,
    w_time: float = 0.15,
    w_abs: float = 0.2,     # NEW: absolute pitch tie-breaker
    ins_cost: float = 0.8,
    del_cost: float = 0.8,
) -> MatchResult:
    """
    DP alignment on interval+contour(+timing), with optional absolute pitch penalty.
    Lower is better.
    """
    QI, QC, QT = query.intervals, query.contour, query.ioi
    SI, SC, ST = song.intervals, song.contour, song.ioi

    m = len(QI)
    n = len(SI)
    if m == 0 or n == 0:
        return MatchResult(cost=float("inf"), end_j=-1)

    INF = 1e18
    D = [[INF] * (n + 1) for _ in range(m + 1)]

    # subsequence: start anywhere in song at zero cost for i=0
    for j in range(n + 1):
        D[0][j] = 0.0

    def local(i: int, j: int) -> float:
        ic = _interval_cost(QI[i - 1], SI[j - 1])
        cc = _contour_cost(QC[i - 1], SC[j - 1])
        tc = 0.0
        if i >= 2 and j >= 2:
            tc = _timing_ratio_cost(QT[i - 2], QT[i - 1], ST[j - 2], ST[j - 1])
        return w_int * ic + w_cont * cc + w_time * tc

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best = INF

            # match/substitute
            best = min(best, D[i - 1][j - 1] + local(i, j))

            # insertion/deletion
            best = min(best, D[i - 1][j] + ins_cost)
            best = min(best, D[i][j - 1] + del_cost)

            # note insertion: two query intervals ~ one song interval
            if i >= 2:
                merged_q = max(-4, min(4, QI[i - 2] + QI[i - 1]))
                merged_c = QC[i - 1]  # simple approximation
                ic = _interval_cost(merged_q, SI[j - 1])
                cc = _contour_cost(merged_c, SC[j - 1])
                best = min(best, D[i - 2][j - 1] + w_int * ic + w_cont * cc + 0.6)

            # note deletion: one query interval ~ two song intervals
            if j >= 2:
                merged_s = max(-4, min(4, SI[j - 2] + SI[j - 1]))
                merged_c = SC[j - 1]
                ic = _interval_cost(QI[i - 1], merged_s)
                cc = _contour_cost(QC[i - 1], merged_c)
                best = min(best, D[i - 1][j - 2] + w_int * ic + w_cont * cc + 0.6)

            D[i][j] = best

    # best subsequence: minimum over all end positions j
    best_cost = INF
    best_j = -1
    for j in range(1, n + 1):
        if D[m][j] < best_cost:
            best_cost = D[m][j]
            best_j = j - 1

    # NEW: absolute pitch tie-breaker (very light)
    # Compare average pitch level (in MIDI notes), normalize per octave.
    if query.pitches and song.pitches:
        avg_q = sum(query.pitches) / len(query.pitches)
        avg_s = sum(song.pitches) / len(song.pitches)
        abs_pen = w_abs * (abs(avg_q - avg_s) / 12.0)
        best_cost += abs_pen

    return MatchResult(cost=best_cost, end_j=best_j)
