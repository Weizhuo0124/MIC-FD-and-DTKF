#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验3：MIC-FD 参数敏感性与计算开销分析
====================================

目的
----
这不是再次调参，而是在已经由 validation set 锁定

    D_th = 1.25 m
    R_dif = 4.0 dB

之后，使用独立 test seeds 检查参数变化时 MIC-FD 的稳定性。

Full MIC-FD 选择流程
--------------------
AP1: IG
AP2: DeltaIG screening + MIC
AP3: FD
AP4: FD

其中：
- AP筛选不使用 DeepMIMO；
- AP1 固定为 (0,0)；
- AP2 的 MIC 保持当前代码的 original abs(Pearson) surrogate；
- AP3/AP4 直接在所有未选 AP 中使用 FD；
- RSS state = Raw_exact；
- D_th / R_dif 只改变 FD 阶段。

敏感性设计：单因素分析
----------------------
A. D_th sensitivity
   固定 R_dif = 4.0 dB
   D_th = 0.50, 0.75, 1.00, 1.25, 1.50 m

B. R_dif sensitivity
   固定 D_th = 1.25 m
   R_dif = 2, 3, 4, 5 dB

注意：
- 1.25 m / 4 dB 是 validation-selected baseline；
- test set 只用于报告 robustness，不再根据 test 结果重新选择参数；
- 正式 test seeds 默认 1000...1029，与 validation 100...129 不重叠；
- 默认同时报告 rss_power 和 csi_rss；
- csi_rss = CSI-derived RSS + KNN，不是 amplitude/phase + 3D CNN。

默认运行
--------
python 实验3_MIC-FD参数敏感性与计算开销.py

仅 RSS：
python 实验3_MIC-FD参数敏感性与计算开销.py --feature-mode rss_power

无 DeepMIMO 时先检查逻辑：
python 实验3_MIC-FD参数敏感性与计算开销.py --evaluation analytic
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor


# ============================================================
# 固定环境参数
# ============================================================
C0 = 299_792_458.0
FC_HZ = 3.5e9
LAMBDA = C0 / FC_HZ

ROOM_W = 8.0
ROOM_H = 4.0
GRID = 0.5
RX_Z = 1.5
TX_Z = 1.5

D0 = 1.0
PR0_DBM = -40.0
PATH_LOSS_EXPONENT = 2.2
WALL_LOSS_DB = 10.0

NOISE_SIGMA_DB = 3.0
N_MEASUREMENTS = 10
K = 5
N_TEST = 60

# validation 后锁定，敏感性分析不再重新选最优参数
BASELINE_D_TH = 1.25
BASELINE_R_DIF_DB = 4.0

VALIDATION_SEEDS = set(range(100, 130))

REFLECTION_PROFILES = {
    "none": (),
    "very_weak": (20.0, 22.0, 24.0, 26.0),
    "weak": (14.0, 16.0, 18.0, 20.0),
    "moderate": (9.0, 11.0, 13.0, 15.0),
}
OUTER_WALLS = ("x0", "xW", "y0", "yH")


# ============================================================
# 工具
# ============================================================
def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("参数列表不能为空")
    return vals


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def case_id(group: str, d_th: float, r_dif: float) -> str:
    if group == "D_th":
        return f"D敏感性_D={d_th:.2f}_R={r_dif:.2f}"
    return f"R敏感性_D={d_th:.2f}_R={r_dif:.2f}"


# ============================================================
# 墙体模型
# ============================================================
class WallModel:
    def __init__(self, wall_loss_db: float = WALL_LOSS_DB):
        self.x0 = 1.9
        self.x1 = 2.1
        self.y0 = 0.0
        self.y1 = 4.0
        self.door0 = 1.5
        self.door1 = 2.5
        self.wall_loss_db = float(wall_loss_db)

    def is_inside_wall(self, x: float, y: float) -> bool:
        return (
            self.x0 <= x <= self.x1
            and self.y0 <= y <= self.y1
            and not (self.door0 <= y <= self.door1)
        )

    def crosses_wall(self, p1: Sequence[float], p2: Sequence[float]) -> bool:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        if self.is_inside_wall(x1, y1) or self.is_inside_wall(x2, y2):
            return False

        if not (min(x1, x2) < self.x1 and max(x1, x2) > self.x0):
            return False

        if abs(x2 - x1) < 1e-12:
            return False

        for xw in (self.x0, self.x1):
            t = (xw - x1) / (x2 - x1)
            if 0.0 <= t <= 1.0:
                yi = y1 + t * (y2 - y1)
                if (
                    self.y0 <= yi <= self.y1
                    and not (self.door0 <= yi <= self.door1)
                ):
                    return True

        return False


# ============================================================
# RP
# ============================================================
def make_selection_points(wall: WallModel) -> np.ndarray:
    xs = np.arange(0.0, ROOM_W + GRID, GRID)
    ys = np.arange(0.0, ROOM_H + GRID, GRID)

    gx, gy = np.meshgrid(xs, ys)
    all_points = np.column_stack([gx.ravel(), gy.ravel()])

    return np.asarray(
        [p for p in all_points if not wall.is_inside_wall(*p)],
        dtype=float,
    )


def make_eval_points(wall: WallModel) -> np.ndarray:
    points = []

    # 与当前 Performance Evaluation 保持 x-major 顺序
    for x in np.linspace(0.0, ROOM_W, 17):
        for y in np.linspace(0.0, ROOM_H, 9):
            if not wall.is_inside_wall(float(x), float(y)):
                points.append([float(x), float(y), RX_Z])

    return np.asarray(points, dtype=float)


# ============================================================
# AP筛选 RSS：DeepMIMO 不参与
# ============================================================
def screening_rss(
    ap_xy: np.ndarray,
    ref_xy: np.ndarray,
    wall: WallModel,
) -> np.ndarray:
    out = np.zeros(len(ref_xy), dtype=float)

    for i, rp in enumerate(ref_xy):
        d = float(np.linalg.norm(rp - ap_xy))
        d = max(d, D0 / 10.0)

        rss = (
            PR0_DBM
            - 10.0
            * PATH_LOSS_EXPONENT
            * np.log10(d / D0)
        )

        if wall.crosses_wall(ap_xy, rp):
            rss -= wall.wall_loss_db

        out[i] = rss

    return out


# ============================================================
# Raw exact entropy / DeltaIG
# ============================================================
def raw_entropy(fp: np.ndarray) -> tuple[float, int]:
    if fp.ndim == 1:
        fp = fp[:, None]

    _, counts = np.unique(fp, axis=0, return_counts=True)
    p = counts.astype(float) / counts.sum()

    h = float(-np.sum(p * np.log2(p)))
    return h, int(len(counts))


def delta_ig_candidates(
    selected: list[int],
    rss_library: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    current_fp = rss_library[:, selected]
    current_h, current_states = raw_entropy(current_fp)

    delta = np.full(
        rss_library.shape[1],
        -np.inf,
        dtype=float,
    )

    for idx in range(rss_library.shape[1]):
        if idx in selected:
            continue

        fp_new = np.column_stack(
            [current_fp, rss_library[:, idx]]
        )
        h_new, _ = raw_entropy(fp_new)
        delta[idx] = h_new - current_h

    max_delta = float(np.max(delta))

    candidates = np.where(
        np.abs(delta - max_delta) < 1e-6
    )[0]

    return (
        candidates,
        delta,
        current_h,
        current_states,
        max_delta,
    )


# ============================================================
# 当前 MIC surrogate = abs(Pearson)
# ============================================================
def pseudo_mic_pearson(v1: np.ndarray, v2: np.ndarray) -> float:
    try:
        val = abs(float(pearsonr(v1, v2)[0]))
        if np.isnan(val):
            return 0.5
        return val
    except Exception:
        return 0.5


def mic_score(
    candidate_idx: int,
    selected: list[int],
    rss_library: np.ndarray,
) -> float:
    vals = [
        pseudo_mic_pearson(
            rss_library[:, s],
            rss_library[:, candidate_idx],
        )
        for s in selected
    ]
    return float(np.mean(vals))


def choose_by_mic(
    candidates: np.ndarray,
    selected: list[int],
    rss_library: np.ndarray,
) -> tuple[int, float]:
    vals = np.asarray(
        [
            mic_score(int(c), selected, rss_library)
            for c in candidates
        ],
        dtype=float,
    )

    k0 = int(np.argmin(vals))
    return int(candidates[k0]), float(vals[k0])


# ============================================================
# FD
# ============================================================
def build_neighbor_pairs(
    ref_xy: np.ndarray,
    d_th: float,
) -> np.ndarray:
    dist = np.linalg.norm(
        ref_xy[:, None, :] - ref_xy[None, :, :],
        axis=2,
    )

    return np.argwhere(
        (dist <= d_th + 1e-12)
        & (dist > 1e-12)
    )


def fd_score(
    candidate_idx: int,
    selected: list[int],
    rss_library: np.ndarray,
    neighbor_pairs: np.ndarray,
    r_dif_db: float,
) -> int:
    fp = rss_library[:, selected + [candidate_idx]]

    dif = np.abs(
        fp[neighbor_pairs[:, 0], :]
        - fp[neighbor_pairs[:, 1], :]
    )

    return int(
        np.sum(
            np.max(dif, axis=1) >= r_dif_db
        )
    )


def choose_by_fd(
    candidates: np.ndarray,
    selected: list[int],
    rss_library: np.ndarray,
    neighbor_pairs: np.ndarray,
    r_dif_db: float,
) -> tuple[int, float]:
    vals = np.asarray(
        [
            fd_score(
                int(c),
                selected,
                rss_library,
                neighbor_pairs,
                r_dif_db,
            )
            for c in candidates
        ],
        dtype=float,
    )

    k0 = int(np.argmax(vals))
    return int(candidates[k0]), float(vals[k0])


# ============================================================
# Full MIC-FD 对给定参数重新选址
# ============================================================
@dataclass
class SelectionTrace:
    experiment_group: str
    case_id: str
    sweep_value: float
    d_th: float
    r_dif_db: float
    is_validation_baseline: bool

    step: int
    selected_index: int
    selected_x: float
    selected_y: float

    current_entropy: float
    current_state_count: int

    max_delta_ig: float | None
    selected_delta_ig: float | None

    candidate_pool_size: int
    criterion: str
    criterion_value: float | None
    candidate_scope: str


def get_ap1_index(ref_xy: np.ndarray) -> int:
    dist = np.linalg.norm(
        ref_xy - np.asarray([0.0, 0.0]),
        axis=1,
    )
    idx = int(np.argmin(dist))

    if dist[idx] > 1e-9:
        raise RuntimeError("AP1=(0,0) 不在候选集合中")

    return idx


def all_unselected(
    n_candidates: int,
    selected: list[int],
) -> np.ndarray:
    return np.asarray(
        [i for i in range(n_candidates) if i not in selected],
        dtype=int,
    )


def select_full_mic_fd(
    experiment_group: str,
    sweep_value: float,
    d_th: float,
    r_dif_db: float,
    ref_xy: np.ndarray,
    rss_library: np.ndarray,
) -> tuple[list[int], list[SelectionTrace]]:
    """
    当前最终 Full MIC-FD：
        AP1: IG固定点 (0,0)
        AP2: strict max DeltaIG candidates + MIC
        AP3/AP4: FD over all unselected APs
    """
    selected = [get_ap1_index(ref_xy)]
    trace: list[SelectionTrace] = []

    neighbor_pairs = build_neighbor_pairs(
        ref_xy,
        d_th,
    )

    cid = case_id(
        experiment_group,
        d_th,
        r_dif_db,
    )

    is_baseline = (
        close(d_th, BASELINE_D_TH)
        and close(r_dif_db, BASELINE_R_DIF_DB)
    )

    # AP2: DeltaIG + MIC
    (
        candidates,
        delta,
        current_h,
        current_states,
        max_delta,
    ) = delta_ig_candidates(
        selected,
        rss_library,
    )

    best_idx, mic_val = choose_by_mic(
        candidates,
        selected,
        rss_library,
    )

    selected.append(best_idx)

    trace.append(
        SelectionTrace(
            experiment_group=experiment_group,
            case_id=cid,
            sweep_value=float(sweep_value),
            d_th=float(d_th),
            r_dif_db=float(r_dif_db),
            is_validation_baseline=bool(is_baseline),

            step=2,
            selected_index=int(best_idx),
            selected_x=float(ref_xy[best_idx, 0]),
            selected_y=float(ref_xy[best_idx, 1]),

            current_entropy=float(current_h),
            current_state_count=int(current_states),

            max_delta_ig=float(max_delta),
            selected_delta_ig=float(delta[best_idx]),

            candidate_pool_size=int(len(candidates)),
            criterion="DeltaIG_plus_MIC",
            criterion_value=float(mic_val),
            candidate_scope="max_DeltaIG_candidates",
        )
    )

    # AP3/AP4: FD only over all remaining APs
    for ap_num in (3, 4):
        current_h, current_states = raw_entropy(
            rss_library[:, selected]
        )

        candidates = all_unselected(
            len(ref_xy),
            selected,
        )

        best_idx, fd_val = choose_by_fd(
            candidates,
            selected,
            rss_library,
            neighbor_pairs,
            r_dif_db,
        )

        selected.append(best_idx)

        trace.append(
            SelectionTrace(
                experiment_group=experiment_group,
                case_id=cid,
                sweep_value=float(sweep_value),
                d_th=float(d_th),
                r_dif_db=float(r_dif_db),
                is_validation_baseline=bool(is_baseline),

                step=int(ap_num),
                selected_index=int(best_idx),
                selected_x=float(ref_xy[best_idx, 0]),
                selected_y=float(ref_xy[best_idx, 1]),

                current_entropy=float(current_h),
                current_state_count=int(current_states),

                max_delta_ig=None,
                selected_delta_ig=None,

                candidate_pool_size=int(len(candidates)),
                criterion="FD_only",
                criterion_value=float(fd_val),
                candidate_scope="all_unselected_APs",
            )
        )

    return selected, trace


# ============================================================
# Computational Cost：只测 MIC-FD AP 部署本身
# ============================================================
def timed_baseline_mic_fd_once(
    ref_xy: np.ndarray,
    rss_library: np.ndarray,
) -> dict:
    """
    对 validation-selected baseline:
        D_th = 1.25 m
        R_dif = 4.0 dB

    执行一次完整 MIC-FD AP 部署，并分阶段计时。

    重要：
    - 不包含 DeepMIMO 信道生成；
    - 不包含 KNN 定位评估；
    - 不包含 test seeds；
    - 只统计离线 AP screening / deployment algorithm 本身；
    - AP2 = DeltaIG + abs(Pearson) surrogate；
    - AP3/AP4 = FD over all unselected APs。
    """
    d_th = BASELINE_D_TH
    r_dif_db = BASELINE_R_DIF_DB

    t_selection_start = time.perf_counter()

    # AP1：当前最终算法中由 validation 前置实验固定为 (0,0)
    t0 = time.perf_counter()
    ap1_idx = get_ap1_index(ref_xy)
    selected = [ap1_idx]
    t1 = time.perf_counter()
    ap1_init_s = t1 - t0

    # FD 邻近 RP pair 构造（D_th 对应的预处理）
    t0 = time.perf_counter()
    neighbor_pairs = build_neighbor_pairs(
        ref_xy,
        d_th,
    )
    t1 = time.perf_counter()
    neighbor_pair_build_s = t1 - t0

    # AP2：strict max DeltaIG candidate pool + Pearson surrogate
    t0 = time.perf_counter()
    (
        ap2_candidates,
        _delta,
        _current_h,
        _current_states,
        _max_delta,
    ) = delta_ig_candidates(
        selected,
        rss_library,
    )
    ap2_idx, _mic_val = choose_by_mic(
        ap2_candidates,
        selected,
        rss_library,
    )
    selected.append(ap2_idx)
    t1 = time.perf_counter()
    ap2_s = t1 - t0

    # AP3：FD over all unselected APs
    t0 = time.perf_counter()
    ap3_candidates = all_unselected(
        len(ref_xy),
        selected,
    )
    ap3_idx, _fd3 = choose_by_fd(
        ap3_candidates,
        selected,
        rss_library,
        neighbor_pairs,
        r_dif_db,
    )
    selected.append(ap3_idx)
    t1 = time.perf_counter()
    ap3_s = t1 - t0

    # AP4：FD over all unselected APs
    t0 = time.perf_counter()
    ap4_candidates = all_unselected(
        len(ref_xy),
        selected,
    )
    ap4_idx, _fd4 = choose_by_fd(
        ap4_candidates,
        selected,
        rss_library,
        neighbor_pairs,
        r_dif_db,
    )
    selected.append(ap4_idx)
    t1 = time.perf_counter()
    ap4_s = t1 - t0

    selection_total_s = (
        time.perf_counter()
        - t_selection_start
    )

    return {
        "selected_indices": tuple(int(x) for x in selected),
        "selected_layout": tuple(
            (float(ref_xy[i, 0]), float(ref_xy[i, 1]))
            for i in selected
        ),
        "neighbor_pair_count": int(len(neighbor_pairs)),
        "ap2_candidate_pool_size": int(len(ap2_candidates)),
        "ap3_candidate_pool_size": int(len(ap3_candidates)),
        "ap4_candidate_pool_size": int(len(ap4_candidates)),
        "ap1_init_s": float(ap1_init_s),
        "neighbor_pair_build_s": float(neighbor_pair_build_s),
        "ap2_deltaig_mic_s": float(ap2_s),
        "ap3_fd_s": float(ap3_s),
        "ap4_fd_s": float(ap4_s),
        "selection_total_s": float(selection_total_s),
    }


def benchmark_computational_cost(
    ref_xy: np.ndarray,
    wall: WallModel,
    repeats: int,
    warmup: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    计算开销 benchmark。

    每次 repeat 都重新：
        1) 构建完整 screening RSS library；
        2) 用锁定参数执行完整 MIC-FD 4-AP deployment。

    因此可以同时得到：
        - RSS library construction time
        - AP1 initialization
        - neighbor-pair construction
        - AP2 DeltaIG+Pearson
        - AP3 FD
        - AP4 FD
        - full selection time
        - full deployment time (RSS library + selection)

    所有计时均为 wall-clock time.perf_counter()。
    """

    repeats = int(repeats)
    warmup = int(warmup)

    if repeats < 1:
        raise ValueError("cost-repeats 必须 >= 1")
    if warmup < 0:
        raise ValueError("cost-warmup 必须 >= 0")

    def build_library() -> np.ndarray:
        return np.column_stack(
            [
                screening_rss(
                    ap,
                    ref_xy,
                    wall,
                )
                for ap in ref_xy
            ]
        )

    # ----------------------------
    # warm-up：不计入正式统计
    # ----------------------------
    for _ in range(warmup):
        lib = build_library()
        timed_baseline_mic_fd_once(
            ref_xy,
            lib,
        )

    # ----------------------------
    # formal repeats
    # ----------------------------
    rows = []
    expected_indices = None
    expected_layout = None
    structural_meta = None

    for rep in range(1, repeats + 1):
        t_full_start = time.perf_counter()

        t0 = time.perf_counter()
        lib = build_library()
        t1 = time.perf_counter()
        rss_library_s = t1 - t0

        stage = timed_baseline_mic_fd_once(
            ref_xy,
            lib,
        )

        full_deployment_s = (
            time.perf_counter()
            - t_full_start
        )

        if expected_indices is None:
            expected_indices = stage["selected_indices"]
            expected_layout = stage["selected_layout"]
            structural_meta = {
                "neighbor_pair_count": stage["neighbor_pair_count"],
                "ap2_candidate_pool_size": stage["ap2_candidate_pool_size"],
                "ap3_candidate_pool_size": stage["ap3_candidate_pool_size"],
                "ap4_candidate_pool_size": stage["ap4_candidate_pool_size"],
            }
        else:
            if stage["selected_indices"] != expected_indices:
                raise RuntimeError(
                    "Computational-cost benchmark 中重复运行得到不同 AP 布局，"
                    "请检查非确定性。"
                )

        rows.append(
            {
                "repeat": int(rep),
                "rss_library_build_ms": 1000.0 * rss_library_s,
                "ap1_init_ms": 1000.0 * stage["ap1_init_s"],
                "neighbor_pair_build_ms": (
                    1000.0
                    * stage["neighbor_pair_build_s"]
                ),
                "ap2_deltaig_mic_ms": (
                    1000.0
                    * stage["ap2_deltaig_mic_s"]
                ),
                "ap3_fd_ms": (
                    1000.0
                    * stage["ap3_fd_s"]
                ),
                "ap4_fd_ms": (
                    1000.0
                    * stage["ap4_fd_s"]
                ),
                "selection_total_ms": (
                    1000.0
                    * stage["selection_total_s"]
                ),
                "full_deployment_ms": (
                    1000.0
                    * full_deployment_s
                ),
            }
        )

    runs_df = pd.DataFrame(rows)

    stage_map = [
        (
            "RSS library construction",
            "rss_library_build_ms",
        ),
        (
            "AP1 initialization",
            "ap1_init_ms",
        ),
        (
            "Neighbor-pair construction",
            "neighbor_pair_build_ms",
        ),
        (
            "AP2: DeltaIG + Pearson",
            "ap2_deltaig_mic_ms",
        ),
        (
            "AP3: FD",
            "ap3_fd_ms",
        ),
        (
            "AP4: FD",
            "ap4_fd_ms",
        ),
        (
            "Full MIC-FD selection",
            "selection_total_ms",
        ),
        (
            "Full deployment incl. RSS library",
            "full_deployment_ms",
        ),
    ]

    summary_rows = []

    for stage_name, col in stage_map:
        values = runs_df[col].to_numpy(
            dtype=float
        )

        summary_rows.append(
            {
                "stage": stage_name,
                "mean_ms": float(
                    np.mean(values)
                ),
                "std_ms": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                    if len(values) > 1
                    else 0.0
                ),
                "median_ms": float(
                    np.median(values)
                ),
                "p95_ms": float(
                    np.percentile(
                        values,
                        95,
                    )
                ),
                "min_ms": float(
                    np.min(values)
                ),
                "max_ms": float(
                    np.max(values)
                ),
                "repeats": int(
                    len(values)
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    metadata = {
        "timing_clock": "time.perf_counter",
        "unit": "ms",
        "repeats": int(repeats),
        "warmup": int(warmup),
        "D_th_m": float(BASELINE_D_TH),
        "R_dif_db": float(BASELINE_R_DIF_DB),
        "selected_indices": list(
            expected_indices
        ),
        "selected_layout": [
            [float(x), float(y)]
            for x, y
            in expected_layout
        ],
        **structural_meta,
        "includes_deepmimo": False,
        "includes_knn_evaluation": False,
        "includes_test_seed_loop": False,
        "definition_full_selection": (
            "AP1 initialization + neighbor-pair construction "
            "+ AP2 DeltaIG/Pearson + AP3 FD + AP4 FD"
        ),
        "definition_full_deployment": (
            "screening RSS library construction "
            "+ full MIC-FD selection"
        ),
    }

    return (
        runs_df,
        summary_df,
        metadata,
    )


# ============================================================
# 定位评估
# ============================================================
@dataclass
class Metrics:
    experiment_group: str
    case_id: str
    sweep_value: float
    d_th: float
    r_dif_db: float
    is_validation_baseline: bool

    feature_mode: str
    seed: int

    mean_error: float
    median_error: float
    radial_rmse: float
    p90: float
    p95: float
    max_error: float


def evaluate_one_fp(
    experiment_group: str,
    cid: str,
    sweep_value: float,
    d_th: float,
    r_dif: float,
    is_baseline: bool,
    feature_mode: str,
    fp: np.ndarray,
    points_xy: np.ndarray,
    seed: int,
    noise_sigma_db: float,
    n_measurements: int,
    k: int,
    n_test: int,
) -> Metrics:
    # 所有参数设置共享完全相同的 test RP
    rng_select = np.random.default_rng(seed)

    test_ids = rng_select.choice(
        len(points_xy),
        min(n_test, len(points_xy)),
        replace=False,
    )
    true_pos = points_xy[test_ids]

    # 所有参数设置共享相同 seed 的噪声 realization
    rng_noise = np.random.default_rng(seed)

    samples = (
        fp[test_ids, None, :]
        + rng_noise.normal(
            0.0,
            noise_sigma_db,
            size=(
                len(test_ids),
                n_measurements,
                fp.shape[1],
            ),
        )
    )

    measurements = samples.mean(axis=1)

    nx = KNeighborsRegressor(
        n_neighbors=min(k, len(fp)),
        weights="distance",
    )
    ny = KNeighborsRegressor(
        n_neighbors=min(k, len(fp)),
        weights="distance",
    )

    nx.fit(fp, points_xy[:, 0])
    ny.fit(fp, points_xy[:, 1])

    pred = np.column_stack(
        [
            nx.predict(measurements),
            ny.predict(measurements),
        ]
    )

    e = np.linalg.norm(
        pred - true_pos,
        axis=1,
    )

    return Metrics(
        experiment_group=experiment_group,
        case_id=cid,
        sweep_value=float(sweep_value),
        d_th=float(d_th),
        r_dif_db=float(r_dif),
        is_validation_baseline=bool(is_baseline),

        feature_mode=feature_mode,
        seed=int(seed),

        mean_error=float(e.mean()),
        median_error=float(np.median(e)),
        radial_rmse=float(np.sqrt(np.mean(e**2))),
        p90=float(np.percentile(e, 90)),
        p95=float(np.percentile(e, 95)),
        max_error=float(e.max()),
    )


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            [
                "feature_mode",
                "experiment_group",
                "case_id",
                "sweep_value",
                "d_th",
                "r_dif_db",
                "is_validation_baseline",
            ]
        )
        .agg(
            mean_error_avg=("mean_error", "mean"),
            mean_error_std=("mean_error", "std"),
            median_error_avg=("median_error", "mean"),
            radial_rmse_avg=("radial_rmse", "mean"),
            radial_rmse_std=("radial_rmse", "std"),
            p90_avg=("p90", "mean"),
            p95_avg=("p95", "mean"),
            max_error_avg=("max_error", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "feature_mode",
                "experiment_group",
                "sweep_value",
            ]
        )
    )

    # 只计算相对于 validation-selected baseline 的变化，
    # 不用 test 结果重新选择“最优参数”
    rows = []

    for feature in summary["feature_mode"].unique():
        for group in summary["experiment_group"].unique():
            sub = summary[
                (summary["feature_mode"] == feature)
                & (summary["experiment_group"] == group)
            ].copy()

            baseline = sub[
                sub["is_validation_baseline"]
            ]

            if len(baseline) != 1:
                raise RuntimeError(
                    f"{feature}/{group} 中 baseline 数量不是1"
                )

            base_mean = float(
                baseline.iloc[0]["mean_error_avg"]
            )
            base_rmse = float(
                baseline.iloc[0]["radial_rmse_avg"]
            )

            sub["mean_error_change_vs_baseline_m"] = (
                sub["mean_error_avg"] - base_mean
            )

            sub["mean_error_change_vs_baseline_pct"] = (
                (sub["mean_error_avg"] - base_mean)
                / base_mean
                * 100.0
            )

            sub["rmse_change_vs_baseline_m"] = (
                sub["radial_rmse_avg"] - base_rmse
            )

            sub["rmse_change_vs_baseline_pct"] = (
                (sub["radial_rmse_avg"] - base_rmse)
                / base_rmse
                * 100.0
            )

            rows.append(sub)

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# Analytic fallback
# ============================================================
def analytic_fingerprint(
    aps: np.ndarray,
    eval_xyz: np.ndarray,
    wall: WallModel,
) -> np.ndarray:
    xy = eval_xyz[:, :2]

    return np.column_stack(
        [
            screening_rss(ap, xy, wall)
            for ap in aps
        ]
    )


# ============================================================
# DeepMIMO-style 评估
# ============================================================
def _import_deepmimo():
    try:
        import deepmimo as dm
        return dm
    except Exception as exc:
        raise SystemExit(
            "选择了 --evaluation deepmimo，但未检测到 DeepMIMO。\n"
            "请在 DeepMIMO 4.0.3 环境运行，"
            "或使用 --evaluation analytic 先检查逻辑。"
        ) from exc


def az_el(vec: np.ndarray) -> tuple[float, float]:
    vec = np.asarray(vec, dtype=float)

    az = math.degrees(
        math.atan2(vec[1], vec[0])
    )

    horiz = math.hypot(
        vec[0], vec[1]
    )

    el = math.degrees(
        math.atan2(
            vec[2],
            max(horiz, 1e-12),
        )
    )

    return az, el


def mirrored_tx(
    tx: np.ndarray,
    wall_name: str,
) -> np.ndarray:
    out = np.asarray(tx, dtype=float).copy()

    if wall_name == "x0":
        out[0] = -out[0]
    elif wall_name == "xW":
        out[0] = 2.0 * ROOM_W - out[0]
    elif wall_name == "y0":
        out[1] = -out[1]
    elif wall_name == "yH":
        out[1] = 2.0 * ROOM_H - out[1]
    else:
        raise ValueError(wall_name)

    return out


def reflection_point(
    tx: np.ndarray,
    rx: np.ndarray,
    wall_name: str,
) -> np.ndarray:
    tm = mirrored_tx(tx, wall_name)

    if wall_name in ("x0", "xW"):
        plane = 0.0 if wall_name == "x0" else ROOM_W
        denom = rx[0] - tm[0]
        t = (
            0.5
            if abs(denom) < 1e-12
            else (plane - tm[0]) / denom
        )
    else:
        plane = 0.0 if wall_name == "y0" else ROOM_H
        denom = rx[1] - tm[1]
        t = (
            0.5
            if abs(denom) < 1e-12
            else (plane - tm[1]) / denom
        )

    t = float(np.clip(t, 0.0, 1.0))
    q = tm + t * (rx - tm)

    if wall_name == "x0":
        q[0] = 0.0
    elif wall_name == "xW":
        q[0] = ROOM_W
    elif wall_name == "y0":
        q[1] = 0.0
    else:
        q[1] = ROOM_H

    return q


def path_loss_dbm(
    distance: float,
    exponent: float,
) -> float:
    distance = max(float(distance), D0)

    return (
        PR0_DBM
        - 10.0
        * float(exponent)
        * math.log10(distance / D0)
    )


def build_path_matrices(
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    wall: WallModel,
    exponent: float,
    reflection_losses_db: Sequence[float],
) -> tuple[np.ndarray, ...]:
    n = len(rx_xyz)
    n_paths = 1 + len(reflection_losses_db)

    mats = [
        np.zeros((n, n_paths), dtype=float)
        for _ in range(7)
    ]

    for i, rx in enumerate(rx_xyz):
        path_specs = [(None, 0.0)]

        path_specs += [
            (
                reflection_point(
                    tx_xyz,
                    rx,
                    wn,
                ),
                float(loss),
            )
            for wn, loss
            in zip(
                OUTER_WALLS,
                reflection_losses_db,
            )
        ]

        for k0, (q, extra_loss) in enumerate(path_specs):
            if q is None:
                length = float(
                    np.linalg.norm(
                        rx - tx_xyz
                    )
                )

                wall_count = int(
                    wall.crosses_wall(
                        tx_xyz,
                        rx,
                    )
                )

                aod_vec = rx - tx_xyz
                aoa_vec = tx_xyz - rx

            else:
                length = float(
                    np.linalg.norm(
                        q - tx_xyz
                    )
                    + np.linalg.norm(
                        rx - q
                    )
                )

                wall_count = (
                    int(
                        wall.crosses_wall(
                            tx_xyz,
                            q,
                        )
                    )
                    + int(
                        wall.crosses_wall(
                            q,
                            rx,
                        )
                    )
                )

                aod_vec = q - tx_xyz
                aoa_vec = q - rx

            power_dbm = (
                path_loss_dbm(
                    length,
                    exponent,
                )
                - extra_loss
                - wall_count
                * wall.wall_loss_db
            )

            phase_deg = (
                (
                    360.0
                    * length
                    / LAMBDA
                    + 180.0
                )
                % 360.0
            ) - 180.0

            aod_az, aod_el = az_el(aod_vec)
            aoa_az, aoa_el = az_el(aoa_vec)

            mats[0][i, k0] = power_dbm - 30.0
            mats[1][i, k0] = phase_deg
            mats[2][i, k0] = length / C0
            mats[3][i, k0] = aod_az
            mats[4][i, k0] = aod_el
            mats[5][i, k0] = aoa_az
            mats[6][i, k0] = aoa_el

    return tuple(mats)


def make_deepmimo_dataset(
    dm,
    tx_xy: np.ndarray,
    rx_xyz: np.ndarray,
    wall: WallModel,
    exponent: float,
    reflection_losses_db: Sequence[float],
):
    tx_xyz = np.asarray(
        [
            float(tx_xy[0]),
            float(tx_xy[1]),
            TX_Z,
        ],
        dtype=float,
    )

    (
        power,
        phase,
        delay,
        aod_az,
        aod_el,
        aoa_az,
        aoa_el,
    ) = build_path_matrices(
        tx_xyz,
        rx_xyz,
        wall,
        exponent,
        reflection_losses_db,
    )

    return dm.Dataset(
        {
            "rx_pos": rx_xyz.astype(float),
            "tx_pos": tx_xyz.astype(float),
            "tx_vel": np.zeros(3, dtype=float),
            "rx_vel": np.zeros((len(rx_xyz), 3), dtype=float),
            "scene": dm.Scene(),
            "power": power,
            "phase": phase,
            "delay": delay,
            "aod_az": aod_az,
            "aod_el": aod_el,
            "aoa_az": aoa_az,
            "aoa_el": aoa_el,
        }
    )


def total_path_power_dbm(dataset) -> np.ndarray:
    power_dbw = np.asarray(
        dataset.power,
        dtype=float,
    )

    linear_w = np.sum(
        np.power(
            10.0,
            power_dbw / 10.0,
        ),
        axis=1,
    )

    return (
        10.0
        * np.log10(
            np.maximum(
                linear_w,
                1e-30,
            )
        )
        + 30.0
    )


def csi_derived_rss_db(
    dm,
    dataset,
    n_ant: int = 4,
    n_subcarriers: int = 32,
) -> np.ndarray:
    cp = dm.ChannelParameters()

    cp.bs_antenna.shape = [int(n_ant), 1]
    cp.ue_antenna.shape = [1, 1]

    cp.ofdm.subcarriers = int(n_subcarriers)
    cp.ofdm.selected_subcarriers = np.arange(
        int(n_subcarriers)
    )
    cp.ofdm.bandwidth = 20e6

    cp.num_paths = int(
        np.asarray(dataset.power).shape[1]
    )

    cp.freq_domain = 1

    h = np.asarray(
        dataset.compute_channels(cp)
    )

    p = np.mean(
        np.abs(h) ** 2,
        axis=tuple(
            range(1, h.ndim)
        ),
    )

    return (
        10.0
        * np.log10(
            np.maximum(
                p,
                1e-30,
            )
        )
    )


def build_deepmimo_fingerprints(
    layouts: Dict[str, np.ndarray],
    rx_xyz: np.ndarray,
    wall: WallModel,
    reflection_profile: str,
    feature_mode: str,
) -> tuple[Dict[str, np.ndarray], str]:
    dm = _import_deepmimo()

    losses = REFLECTION_PROFILES[
        reflection_profile
    ]

    # 唯一 AP 坐标只生成一次信道
    unique_ap_fp: dict[
        tuple[float, float],
        np.ndarray,
    ] = {}

    for aps in layouts.values():
        for ap in aps:
            key = (
                float(ap[0]),
                float(ap[1]),
            )

            if key in unique_ap_fp:
                continue

            ds = make_deepmimo_dataset(
                dm,
                ap,
                rx_xyz,
                wall,
                PATH_LOSS_EXPONENT,
                losses,
            )

            if feature_mode == "rss_power":
                unique_ap_fp[key] = total_path_power_dbm(ds)

            elif feature_mode == "csi_rss":
                unique_ap_fp[key] = csi_derived_rss_db(
                    dm,
                    ds,
                    n_ant=4,
                    n_subcarriers=32,
                )

            else:
                raise ValueError(feature_mode)

    fps = {
        cid: np.column_stack(
            [
                unique_ap_fp[
                    (
                        float(ap[0]),
                        float(ap[1]),
                    )
                ]
                for ap in aps
            ]
        )
        for cid, aps in layouts.items()
    }

    return fps, str(
        getattr(
            dm,
            "__version__",
            "unknown",
        )
    )


# ============================================================
# 绘图
# ============================================================
def plot_sensitivity(
    summary: pd.DataFrame,
    output_path: Path,
    group: str,
    metric: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    sub = summary[
        summary["experiment_group"] == group
    ]

    for feature in sub["feature_mode"].unique():
        cur = sub[
            sub["feature_mode"] == feature
        ].sort_values("sweep_value")

        ax.plot(
            cur["sweep_value"],
            cur[metric],
            marker="o",
            linewidth=1.6,
            label=feature,
        )

    baseline_x = (
        BASELINE_D_TH
        if group == "D_th"
        else BASELINE_R_DIF_DB
    )

    ax.axvline(
        baseline_x,
        linestyle="--",
        linewidth=1.1,
        label="validation-selected baseline",
    )

    if group == "D_th":
        ax.set_xlabel(r"$D_{th}$ (m)")
        title = r"Sensitivity to $D_{th}$"
    else:
        ax.set_xlabel(r"$R_{dif}$ (dB)")
        title = r"Sensitivity to $R_{dif}$"

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



def plot_computational_cost(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制主要 computational-cost stage。
    不把 Full total 与各子阶段叠加解释为独立成本；
    图仅用于直观展示 wall-clock runtime。
    """
    wanted = [
        "RSS library construction",
        "AP2: DeltaIG + Pearson",
        "AP3: FD",
        "AP4: FD",
        "Full MIC-FD selection",
        "Full deployment incl. RSS library",
    ]

    cur = (
        summary[
            summary["stage"].isin(wanted)
        ]
        .set_index("stage")
        .loc[wanted]
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(9.6, 5.8)
    )

    x = np.arange(
        len(cur)
    )

    ax.bar(
        x,
        cur["mean_ms"],
        yerr=cur["std_ms"],
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        cur["stage"],
        rotation=22,
        ha="right",
    )
    ax.set_ylabel(
        "Runtime (ms)"
    )
    ax.set_title(
        "MIC-FD Computational Cost "
        "(mean ± std)"
    )
    ax.grid(
        True,
        axis="y",
        alpha=0.3,
        linestyle="--",
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="实验3：MIC-FD参数敏感性分析"
    )

    parser.add_argument(
        "--output-dir",
        default="结果_实验3_MIC-FD参数敏感性分析",
    )

    parser.add_argument(
        "--d-th-values",
        default="0.5,0.75,1.0,1.25,1.5",
    )

    parser.add_argument(
        "--r-dif-values",
        default="2,3,4,5",
    )

    parser.add_argument(
        "--evaluation",
        choices=[
            "deepmimo",
            "analytic",
        ],
        default="deepmimo",
    )

    parser.add_argument(
        "--feature-mode",
        choices=[
            "rss_power",
            "csi_rss",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--reflection-profile",
        choices=list(REFLECTION_PROFILES),
        default="weak",
    )

    parser.add_argument(
        "--test-seed-start",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--test-seed-count",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--noise-sigma-db",
        type=float,
        default=NOISE_SIGMA_DB,
    )

    parser.add_argument(
        "--n-measurements",
        type=int,
        default=N_MEASUREMENTS,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=K,
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=N_TEST,
    )

    # Computational Cost：
    # 默认开启；只测 baseline MIC-FD AP部署，不计DeepMIMO/KNN/test loop。
    parser.add_argument(
        "--cost-repeats",
        type=int,
        default=50,
        help="Computational Cost 正式重复次数，默认50",
    )

    parser.add_argument(
        "--cost-warmup",
        type=int,
        default=5,
        help="Computational Cost 预热次数，不计入统计，默认5",
    )

    parser.add_argument(
        "--skip-computational-cost",
        action="store_true",
        help="跳过 Computational Cost benchmark",
    )

    args = parser.parse_args()

    d_values = parse_float_list(
        args.d_th_values
    )

    r_values = parse_float_list(
        args.r_dif_values
    )

    # baseline 必须包含在两条敏感性曲线中
    if not any(close(x, BASELINE_D_TH) for x in d_values):
        raise ValueError(
            f"D_th values 必须包含 baseline {BASELINE_D_TH}"
        )

    if not any(close(x, BASELINE_R_DIF_DB) for x in r_values):
        raise ValueError(
            f"R_dif values 必须包含 baseline {BASELINE_R_DIF_DB}"
        )

    test_seeds = list(
        range(
            args.test_seed_start,
            args.test_seed_start
            + args.test_seed_count,
        )
    )

    overlap = (
        set(test_seeds)
        & VALIDATION_SEEDS
    )

    if overlap:
        raise SystemExit(
            "ERROR：test seeds 与 validation seeds 重叠："
            f"{sorted(overlap)}"
        )

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    wall = WallModel(
        WALL_LOSS_DB
    )

    selection_xy = make_selection_points(
        wall
    )

    eval_xyz = make_eval_points(
        wall
    )

    if (
        len(selection_xy) != 147
        or len(eval_xyz) != 147
    ):
        raise RuntimeError(
            "有效RP数量异常："
            f"{len(selection_xy)} / {len(eval_xyz)}"
        )

    print("=" * 96)
    print("实验3：MIC-FD 参数敏感性分析")
    print("=" * 96)
    print("重要：这是 robustness analysis，不根据 test set 重新调参")
    print(f"validation-selected baseline: D_th={BASELINE_D_TH:.2f} m, R_dif={BASELINE_R_DIF_DB:.2f} dB")
    print(f"D_th sweep: {d_values}，固定 R_dif={BASELINE_R_DIF_DB:.2f} dB")
    print(f"R_dif sweep: {r_values}，固定 D_th={BASELINE_D_TH:.2f} m")
    print("Full MIC-FD: IG -> (DeltaIG+MIC) -> FD -> FD")
    print("AP筛选使用DeepMIMO: False")
    print("RSS state: Raw_exact")
    print("MIC实现: original abs(Pearson) surrogate")
    print(f"test seeds: {test_seeds[0]}...{test_seeds[-1]}")
    print("validation seeds: 100...129（仅用于之前的参数选择，本实验不再使用）")
    print(f"evaluation: {args.evaluation}")
    print(f"feature mode: {args.feature_mode}")
    if args.skip_computational_cost:
        print("Computational Cost: skipped")
    else:
        print(
            "Computational Cost: "
            f"baseline-only, repeats={args.cost_repeats}, "
            f"warmup={args.cost_warmup}"
        )
        print(
            "  cost scope: RSS screening library + "
            "MIC-FD AP selection only"
        )
        print(
            "  excludes: DeepMIMO / KNN / test seed loop"
        )
    print("=" * 96)

    # --------------------------------------------------------
    # AP筛选 RSS library
    # --------------------------------------------------------
    rss_library = np.column_stack(
        [
            screening_rss(
                ap,
                selection_xy,
                wall,
            )
            for ap in selection_xy
        ]
    )

    # --------------------------------------------------------
    # 单因素参数 cases
    # --------------------------------------------------------
    cases = []

    for d_th in d_values:
        cases.append(
            {
                "experiment_group": "D_th",
                "sweep_value": float(d_th),
                "d_th": float(d_th),
                "r_dif_db": float(BASELINE_R_DIF_DB),
            }
        )

    for r_dif in r_values:
        cases.append(
            {
                "experiment_group": "R_dif",
                "sweep_value": float(r_dif),
                "d_th": float(BASELINE_D_TH),
                "r_dif_db": float(r_dif),
            }
        )

    layouts: Dict[str, np.ndarray] = {}
    case_meta = {}
    layout_rows = []
    trace_rows = []

    for item in cases:
        group = item["experiment_group"]
        d_th = item["d_th"]
        r_dif = item["r_dif_db"]
        sweep_value = item["sweep_value"]

        cid = case_id(
            group,
            d_th,
            r_dif,
        )

        selected, trace = select_full_mic_fd(
            experiment_group=group,
            sweep_value=sweep_value,
            d_th=d_th,
            r_dif_db=r_dif,
            ref_xy=selection_xy,
            rss_library=rss_library,
        )

        aps = selection_xy[selected]
        layouts[cid] = aps

        case_meta[cid] = {
            **item,
            "is_validation_baseline": bool(
                close(d_th, BASELINE_D_TH)
                and close(r_dif, BASELINE_R_DIF_DB)
            ),
        }

        for order, (idx, pos) in enumerate(
            zip(selected, aps),
            start=1,
        ):
            layout_rows.append(
                {
                    "experiment_group": group,
                    "case_id": cid,
                    "sweep_value": float(sweep_value),
                    "d_th": float(d_th),
                    "r_dif_db": float(r_dif),
                    "is_validation_baseline": bool(
                        close(d_th, BASELINE_D_TH)
                        and close(r_dif, BASELINE_R_DIF_DB)
                    ),
                    "ap_order": int(order),
                    "selected_index": int(idx),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                }
            )

        trace_rows.extend(
            asdict(x)
            for x in trace
        )

    layout_df = pd.DataFrame(
        layout_rows
    )

    trace_df = pd.DataFrame(
        trace_rows
    )

    layout_df.to_csv(
        out / "参数敏感性_AP布局.csv",
        index=False,
        encoding="utf-8-sig",
    )

    trace_df.to_csv(
        out / "参数敏感性_AP筛选过程.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 各参数设置选出的AP布局 ===")
    print(
        layout_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Feature modes
    # --------------------------------------------------------
    if args.evaluation == "analytic":
        feature_modes = [
            "analytic_rssi"
        ]
    elif args.feature_mode == "both":
        feature_modes = [
            "rss_power",
            "csi_rss",
        ]
    else:
        feature_modes = [
            args.feature_mode
        ]

    all_runs = []
    all_summary = []
    deepmimo_version = None

    for feature in feature_modes:
        print(
            f"\n构建 {feature} fingerprints..."
        )

        if args.evaluation == "analytic":
            fps = {
                cid: analytic_fingerprint(
                    aps,
                    eval_xyz,
                    wall,
                )
                for cid, aps
                in layouts.items()
            }

        else:
            fps, deepmimo_version = build_deepmimo_fingerprints(
                layouts=layouts,
                rx_xyz=eval_xyz,
                wall=wall,
                reflection_profile=args.reflection_profile,
                feature_mode=feature,
            )

        repeated = []

        for cid, fp in fps.items():
            meta = case_meta[cid]

            for seed in test_seeds:
                repeated.append(
                    evaluate_one_fp(
                        experiment_group=meta[
                            "experiment_group"
                        ],
                        cid=cid,
                        sweep_value=meta[
                            "sweep_value"
                        ],
                        d_th=meta["d_th"],
                        r_dif=meta["r_dif_db"],
                        is_baseline=meta[
                            "is_validation_baseline"
                        ],
                        feature_mode=feature,
                        fp=fp,
                        points_xy=eval_xyz[:, :2],
                        seed=seed,
                        noise_sigma_db=args.noise_sigma_db,
                        n_measurements=args.n_measurements,
                        k=args.k,
                        n_test=args.n_test,
                    )
                )

        runs_df = pd.DataFrame(
            [
                asdict(x)
                for x in repeated
            ]
        )

        summary = summarize_metrics(
            runs_df
        )

        runs_df.to_csv(
            out / f"全部test_seeds_{feature}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            out / f"参数敏感性汇总_{feature}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\n=== {feature} 参数敏感性汇总 ==="
        )

        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        all_runs.append(
            runs_df
        )

        all_summary.append(
            summary
        )

    all_runs_df = pd.concat(
        all_runs,
        ignore_index=True,
    )

    summary_all = pd.concat(
        all_summary,
        ignore_index=True,
    )

    all_runs_df.to_csv(
        out / "参数敏感性_全部运行.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_all.to_csv(
        out / "参数敏感性_最终汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 单独输出两张论文表
    # --------------------------------------------------------
    d_table = summary_all[
        summary_all["experiment_group"]
        == "D_th"
    ].copy()

    r_table = summary_all[
        summary_all["experiment_group"]
        == "R_dif"
    ].copy()

    d_table.to_csv(
        out / "D_th敏感性表.csv",
        index=False,
        encoding="utf-8-sig",
    )

    r_table.to_csv(
        out / "R_dif敏感性表.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------
    plot_sensitivity(
        summary_all,
        out / "D_th_平均定位误差敏感性.png",
        group="D_th",
        metric="mean_error_avg",
        ylabel="Mean Localization Error (m)",
    )

    plot_sensitivity(
        summary_all,
        out / "R_dif_平均定位误差敏感性.png",
        group="R_dif",
        metric="mean_error_avg",
        ylabel="Mean Localization Error (m)",
    )

    plot_sensitivity(
        summary_all,
        out / "D_th_RMSE敏感性.png",
        group="D_th",
        metric="radial_rmse_avg",
        ylabel="Radial RMSE (m)",
    )

    plot_sensitivity(
        summary_all,
        out / "R_dif_RMSE敏感性.png",
        group="R_dif",
        metric="radial_rmse_avg",
        ylabel="Radial RMSE (m)",
    )

    # --------------------------------------------------------
    # 布局平台/重复情况
    # --------------------------------------------------------
    layout_signature_rows = []

    for group in ("D_th", "R_dif"):
        sub = layout_df[
            layout_df["experiment_group"] == group
        ]

        for cid in sub["case_id"].unique():
            cur = sub[
                sub["case_id"] == cid
            ].sort_values("ap_order")

            coords = [
                (
                    float(r.x),
                    float(r.y),
                )
                for r
                in cur.itertuples()
            ]

            first = cur.iloc[0]

            layout_signature_rows.append(
                {
                    "experiment_group": group,
                    "case_id": cid,
                    "sweep_value": float(
                        first["sweep_value"]
                    ),
                    "d_th": float(
                        first["d_th"]
                    ),
                    "r_dif_db": float(
                        first["r_dif_db"]
                    ),
                    "layout": str(coords),
                }
            )

    sig_df = pd.DataFrame(
        layout_signature_rows
    )

    sig_df.to_csv(
        out / "参数敏感性_布局平台分析.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Computational Cost
    # --------------------------------------------------------
    cost_runs_df = None
    cost_summary_df = None
    cost_metadata = None

    if not args.skip_computational_cost:
        print(
            "\n"
            + "=" * 96
        )
        print(
            "Computational Cost："
            "baseline MIC-FD AP deployment"
        )
        print(
            f"Locked parameters: "
            f"D_th={BASELINE_D_TH:.2f} m, "
            f"R_dif={BASELINE_R_DIF_DB:.2f} dB"
        )
        print(
            "Important: DeepMIMO/KNN/test-seed loops "
            "are NOT included."
        )
        print(
            "=" * 96
        )

        (
            cost_runs_df,
            cost_summary_df,
            cost_metadata,
        ) = benchmark_computational_cost(
            ref_xy=selection_xy,
            wall=wall,
            repeats=args.cost_repeats,
            warmup=args.cost_warmup,
        )

        cost_runs_df.to_csv(
            out / "计算开销_全部重复.csv",
            index=False,
            encoding="utf-8-sig",
        )

        cost_summary_df.to_csv(
            out / "计算开销_汇总.csv",
            index=False,
            encoding="utf-8-sig",
        )

        (
            out
            / "计算开销_元数据.json"
        ).write_text(
            json.dumps(
                cost_metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        plot_computational_cost(
            cost_summary_df,
            out
            / "计算开销_阶段运行时间.png",
        )

        print(
            "\n=== Computational Cost Summary ==="
        )
        print(
            cost_summary_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print(
            "\nBaseline selected layout:"
        )
        for i, pos in enumerate(
            cost_metadata["selected_layout"],
            start=1,
        ):
            print(
                f"  AP{i}: "
                f"({pos[0]:.1f}, {pos[1]:.1f})"
            )

        print(
            "Neighbor-pair count:",
            cost_metadata[
                "neighbor_pair_count"
            ],
        )
        print(
            "Candidate pool sizes: "
            f"AP2={cost_metadata['ap2_candidate_pool_size']}, "
            f"AP3={cost_metadata['ap3_candidate_pool_size']}, "
            f"AP4={cost_metadata['ap4_candidate_pool_size']}"
        )

    # --------------------------------------------------------
    # 配置
    # --------------------------------------------------------
    config = {
        "experiment": "实验3：MIC-FD参数敏感性分析",

        "purpose": (
            "robustness analysis after validation; "
            "not parameter re-optimization on test set"
        ),

        "full_mic_fd": (
            "IG -> (DeltaIG + MIC) -> FD -> FD"
        ),

        "baseline_selected_by_validation": {
            "D_th_m": BASELINE_D_TH,
            "R_dif_db": BASELINE_R_DIF_DB,
            "validation_seeds": list(
                range(100, 130)
            ),
        },

        "sensitivity_design": {
            "D_th_values_m": d_values,
            "D_th_fixed_R_dif_db": BASELINE_R_DIF_DB,
            "R_dif_values_db": r_values,
            "R_dif_fixed_D_th_m": BASELINE_D_TH,
            "one_factor_at_a_time": True,
            "test_set_used_for_reselection": False,
        },

        "screening": {
            "uses_deepmimo": False,
            "rss_state": "Raw_exact",
            "AP1": [0.0, 0.0],
            "AP2": (
                "strict max DeltaIG candidates "
                "+ original abs(Pearson) surrogate"
            ),
            "AP3_AP4": (
                "FD over all unselected APs"
            ),
            "mic_is_true_MIC": False,
        },

        "environment": {
            "room_m": [ROOM_W, ROOM_H],
            "grid_m": GRID,
            "valid_rps": int(
                len(selection_xy)
            ),
            "path_loss_exponent": PATH_LOSS_EXPONENT,
            "pr0_dbm": PR0_DBM,
            "wall_loss_db": WALL_LOSS_DB,
            "D_th_baseline_m": BASELINE_D_TH,
            "R_dif_baseline_db": BASELINE_R_DIF_DB,
        },

        "computational_cost": {
            "enabled": (
                not args.skip_computational_cost
            ),
            "scope": (
                "offline AP deployment only: "
                "screening RSS library + MIC-FD selection"
            ),
            "baseline_D_th_m": BASELINE_D_TH,
            "baseline_R_dif_db": BASELINE_R_DIF_DB,
            "repeats": (
                None
                if args.skip_computational_cost
                else args.cost_repeats
            ),
            "warmup": (
                None
                if args.skip_computational_cost
                else args.cost_warmup
            ),
            "clock": "time.perf_counter",
            "includes_deepmimo": False,
            "includes_knn_evaluation": False,
            "includes_test_seed_loop": False,
            "summary_file": (
                None
                if args.skip_computational_cost
                else "计算开销_汇总.csv"
            ),
        },

        "evaluation": {
            "mode": args.evaluation,
            "feature_mode": args.feature_mode,
            "deepmimo_version": deepmimo_version,
            "reflection_profile": (
                args.reflection_profile
                if args.evaluation == "deepmimo"
                else None
            ),
            "carrier_frequency_ghz": FC_HZ / 1e9,
            "bs_antennas": 4,
            "ue_antennas": 1,
            "ofdm_subcarriers": 32,
            "bandwidth_mhz": 20,
            "noise_sigma_db": args.noise_sigma_db,
            "n_measurements": args.n_measurements,
            "K": args.k,
            "n_test": args.n_test,
            "test_seeds": test_seeds,
            "same_test_rps_and_noise_across_cases": True,
            "csi_rss_note": (
                "CSI-derived RSS + KNN, "
                "not amplitude/phase + 3D CNN"
            ),
        },
    }

    (
        out / "实验配置.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("运行完成。")
    print("注意：不要根据本次 test sensitivity 的最小误差重新改 baseline 参数。")
    print(f"论文基准参数继续保持：D_th={BASELINE_D_TH:.2f} m, R_dif={BASELINE_R_DIF_DB:.2f} dB")
    if not args.skip_computational_cost:
        print(
            "Computational Cost 已输出："
            "计算开销_汇总.csv / 计算开销_阶段运行时间.png"
        )
        print(
            "注意：计算开销仅指 MIC-FD 离线 AP 部署，"
            "不包含 DeepMIMO 与定位评估。"
        )
    print("结果目录:", out.resolve())
    print("=" * 96)


if __name__ == "__main__":
    main()
