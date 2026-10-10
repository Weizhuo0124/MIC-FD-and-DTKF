#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验1：MIC-FD 五种性能方案（最终定义）
===================================

五种性能方案
------------
1. IG-only
   IG -> IG -> IG -> IG

2. IG+MIC
   IG -> MIC -> MIC -> MIC

3. IG+FD
   IG -> IG -> FD -> FD

4. Full MIC-FD
   IG -> (DeltaIG + MIC) -> FD -> FD

5. w/o IG
   Random -> (DeltaIG + MIC) -> FD -> FD
   默认使用 30 个互不重复的随机 AP1 初始化。

重要说明
--------
A. 前三个性能基准不使用 DeltaIG：
   - IG-only 的 AP2/AP3/AP4 直接比较加入候选 AP 后的联合熵 H；
   - IG+MIC 的 AP2/AP3/AP4 直接使用 MIC；
   - IG+FD 的 AP2 使用 IG，AP3/AP4 使用 FD。

B. Full 中的 DeltaIG 只出现在 AP2：
   先找最大 DeltaIG 候选集合，再用 MIC 选择冗余最低的 AP。

C. w/o IG 为了单独验证“AP1 的 IG 初始化”作用，保留 Full 中其余步骤：
   DeltaIG+MIC -> FD -> FD。

D. 本脚本不再把 w/o DeltaIG 作为性能方案。
   之前实验已显示，在当前场景中删除 DeltaIG 后最终 AP 布局不变，
   因而 DeltaIG 更适合作为候选筛选/计算步骤单独讨论，而不是性能方案。

E. 当前所谓 MIC 保持原代码实现：
   abs(Pearson correlation)，它是 MIC surrogate，不是真正的 MIC。

锁定参数
--------
D_th = 1.25 m
R_dif = 4.0 dB

Validation seeds:
100...129

正式 test seeds:
1000...1029

随机 AP1 初始化：
默认 30 个不同位置，random seed = 2000。

默认运行
--------
python 实验1_MIC-FD五种性能方案_最终版.py

仅 RSSI：
python 实验1_MIC-FD五种性能方案_最终版.py --feature-mode rss_power

无 DeepMIMO 时逻辑检查：
python 实验1_MIC-FD五种性能方案_最终版.py --evaluation analytic
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.neighbors import KNeighborsRegressor


# ============================================================
# 固定环境
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

D_TH = 1.25
R_DIF_DB = 4.0

NOISE_SIGMA_DB = 3.0
N_MEASUREMENTS = 10
K = 5
N_TEST = 60

VALIDATION_SEEDS = set(range(100, 130))

FIXED_VARIANTS = (
    "IG-only",
    "IG+MIC",
    "IG+FD",
    "Full-MIC-FD",
)

DISPLAY_ORDER = (
    "IG-only",
    "IG+MIC",
    "IG+FD",
    "Full-MIC-FD",
    "w/o-IG",
)

REFLECTION_PROFILES = {
    "none": (),
    "very_weak": (20.0, 22.0, 24.0, 26.0),
    "weak": (14.0, 16.0, 18.0, 20.0),
    "moderate": (9.0, 11.0, 13.0, 15.0),
}
OUTER_WALLS = ("x0", "xW", "y0", "yH")


# ============================================================
# 墙体
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

    def crosses_wall(
        self,
        p1: Sequence[float],
        p2: Sequence[float],
    ) -> bool:
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
# 参考点
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

    for x in np.linspace(0.0, ROOM_W, 17):
        for y in np.linspace(0.0, ROOM_H, 9):
            if not wall.is_inside_wall(float(x), float(y)):
                points.append([float(x), float(y), RX_Z])

    return np.asarray(points, dtype=float)


# ============================================================
# AP 筛选 RSS：不使用 DeepMIMO
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
# IG / entropy
# ============================================================
def raw_entropy(fp: np.ndarray) -> tuple[float, int]:
    if fp.ndim == 1:
        fp = fp[:, None]

    _, counts = np.unique(
        fp,
        axis=0,
        return_counts=True,
    )

    p = counts.astype(float) / counts.sum()
    h = float(-np.sum(p * np.log2(p)))

    return h, int(len(counts))


def ig_candidates(
    selected: list[int],
    rss_library: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    纯 IG 方法：
    对每个未选候选 j 直接计算联合指纹熵

        H(S ∪ {j})

    并选择 H 最大的候选。

    注意：
    这里方法名称和记录字段均为 IG，不把它写成 DeltaIG。
    """
    h_after = np.full(
        rss_library.shape[1],
        -np.inf,
        dtype=float,
    )

    for idx in range(rss_library.shape[1]):
        if idx in selected:
            continue

        fp = rss_library[:, selected + [idx]]
        h_after[idx], _ = raw_entropy(fp)

    max_h = float(np.max(h_after))

    candidates = np.where(
        np.abs(h_after - max_h) < 1e-6
    )[0]

    current_h, current_states = raw_entropy(
        rss_library[:, selected]
    )

    return candidates, h_after, current_h, current_states


def delta_ig_candidates(
    selected: list[int],
    rss_library: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    int,
    float,
]:
    """
    只给 Full / w/o IG 使用的 DeltaIG screening。
    """
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

        fp = rss_library[:, selected + [idx]]
        h_new, _ = raw_entropy(fp)
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
# MIC surrogate：abs(Pearson)
# ============================================================
def pseudo_mic_pearson(
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    try:
        value = abs(
            float(
                pearsonr(v1, v2)[0]
            )
        )

        if np.isnan(value):
            return 0.5

        return value

    except Exception:
        return 0.5


def mic_score(
    candidate_idx: int,
    selected: list[int],
    rss_library: np.ndarray,
) -> float:
    """
    与所有已选 AP 的 abs(Pearson) 平均值。
    越低越好。
    """
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
    scores = np.asarray(
        [
            mic_score(
                int(c),
                selected,
                rss_library,
            )
            for c in candidates
        ],
        dtype=float,
    )

    best_pos = int(np.argmin(scores))

    return (
        int(candidates[best_pos]),
        float(scores[best_pos]),
    )


# ============================================================
# FD
# ============================================================
def build_neighbor_pairs(
    ref_xy: np.ndarray,
    d_th: float,
) -> np.ndarray:
    dist = np.linalg.norm(
        ref_xy[:, None, :]
        - ref_xy[None, :, :],
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
            np.max(dif, axis=1)
            >= r_dif_db
        )
    )


def choose_by_fd(
    candidates: np.ndarray,
    selected: list[int],
    rss_library: np.ndarray,
    neighbor_pairs: np.ndarray,
    r_dif_db: float,
) -> tuple[int, float]:
    scores = np.asarray(
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

    best_pos = int(np.argmax(scores))

    return (
        int(candidates[best_pos]),
        float(scores[best_pos]),
    )


# ============================================================
# AP 选择过程
# ============================================================
@dataclass
class SelectionTrace:
    variant: str
    init_id: int | None
    step: int

    selected_index: int
    selected_x: float
    selected_y: float

    current_entropy: float | None
    current_state_count: int | None

    max_ig_after: float | None
    selected_ig_after: float | None

    max_delta_ig: float | None
    selected_delta_ig: float | None

    candidate_pool_size: int

    criterion: str
    criterion_value: float | None
    candidate_scope: str


def get_fixed_ap1_index(
    ref_xy: np.ndarray,
) -> int:
    d = np.linalg.norm(
        ref_xy
        - np.asarray([0.0, 0.0]),
        axis=1,
    )

    idx = int(np.argmin(d))

    if d[idx] > 1e-9:
        raise RuntimeError(
            "AP1=(0,0) 不在候选集中"
        )

    return idx


def all_unselected(
    n_candidates: int,
    selected: list[int],
) -> np.ndarray:
    return np.asarray(
        [
            i
            for i in range(n_candidates)
            if i not in selected
        ],
        dtype=int,
    )


def select_layout(
    variant: str,
    ref_xy: np.ndarray,
    rss_library: np.ndarray,
    random_ap1_idx: int | None = None,
    init_id: int | None = None,
) -> tuple[list[int], list[SelectionTrace]]:

    neighbor_pairs = build_neighbor_pairs(
        ref_xy,
        D_TH,
    )

    # --------------------------------------------------------
    # AP1
    # --------------------------------------------------------
    if variant == "w/o-IG":
        if random_ap1_idx is None:
            raise ValueError(
                "w/o-IG 必须提供 random_ap1_idx"
            )

        selected = [int(random_ap1_idx)]

    else:
        selected = [
            get_fixed_ap1_index(ref_xy)
        ]

    trace: list[SelectionTrace] = []

    # --------------------------------------------------------
    # AP2~AP4
    # --------------------------------------------------------
    for ap_num in range(2, 5):

        current_h, current_states = raw_entropy(
            rss_library[:, selected]
        )

        max_ig_after = None
        selected_ig_after = None
        max_delta = None
        selected_delta = None
        criterion_value = None

        # ====================================================
        # 1) IG-only
        # IG -> IG -> IG -> IG
        # ====================================================
        if variant == "IG-only":

            (
                candidates,
                h_after,
                current_h,
                current_states,
            ) = ig_candidates(
                selected,
                rss_library,
            )

            # 纯IG最大值并列时采用最低索引，保证可复现
            best_idx = int(np.min(candidates))

            max_ig_after = float(
                np.max(h_after)
            )

            selected_ig_after = float(
                h_after[best_idx]
            )

            criterion = "IG"
            candidate_scope = "max_IG_candidates"
            pool_size = int(len(candidates))

        # ====================================================
        # 2) IG+MIC
        # IG -> MIC -> MIC -> MIC
        # ====================================================
        elif variant == "IG+MIC":

            candidates = all_unselected(
                len(ref_xy),
                selected,
            )

            best_idx, criterion_value = choose_by_mic(
                candidates,
                selected,
                rss_library,
            )

            criterion = "MIC_only"
            candidate_scope = "all_unselected_APs"
            pool_size = int(len(candidates))

        # ====================================================
        # 3) IG+FD
        # IG -> IG -> FD -> FD
        # ====================================================
        elif variant == "IG+FD":

            if ap_num == 2:
                (
                    candidates,
                    h_after,
                    current_h,
                    current_states,
                ) = ig_candidates(
                    selected,
                    rss_library,
                )

                best_idx = int(
                    np.min(candidates)
                )

                max_ig_after = float(
                    np.max(h_after)
                )

                selected_ig_after = float(
                    h_after[best_idx]
                )

                criterion = "IG"
                candidate_scope = "max_IG_candidates"
                pool_size = int(len(candidates))

            else:
                candidates = all_unselected(
                    len(ref_xy),
                    selected,
                )

                best_idx, criterion_value = choose_by_fd(
                    candidates,
                    selected,
                    rss_library,
                    neighbor_pairs,
                    R_DIF_DB,
                )

                criterion = "FD_only"
                candidate_scope = "all_unselected_APs"
                pool_size = int(len(candidates))

        # ====================================================
        # 4) Full MIC-FD
        # IG -> (DeltaIG + MIC) -> FD -> FD
        # ====================================================
        elif variant == "Full-MIC-FD":

            if ap_num == 2:
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

                best_idx, criterion_value = choose_by_mic(
                    candidates,
                    selected,
                    rss_library,
                )

                selected_delta = float(
                    delta[best_idx]
                )

                criterion = "DeltaIG_plus_MIC"
                candidate_scope = "max_DeltaIG_candidates"
                pool_size = int(len(candidates))

            else:
                candidates = all_unselected(
                    len(ref_xy),
                    selected,
                )

                best_idx, criterion_value = choose_by_fd(
                    candidates,
                    selected,
                    rss_library,
                    neighbor_pairs,
                    R_DIF_DB,
                )

                criterion = "FD_only"
                candidate_scope = "all_unselected_APs"
                pool_size = int(len(candidates))

        # ====================================================
        # 5) w/o IG
        # Random -> (DeltaIG + MIC) -> FD -> FD
        # ====================================================
        elif variant == "w/o-IG":

            if ap_num == 2:
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

                best_idx, criterion_value = choose_by_mic(
                    candidates,
                    selected,
                    rss_library,
                )

                selected_delta = float(
                    delta[best_idx]
                )

                criterion = "DeltaIG_plus_MIC"
                candidate_scope = "max_DeltaIG_candidates"
                pool_size = int(len(candidates))

            else:
                candidates = all_unselected(
                    len(ref_xy),
                    selected,
                )

                best_idx, criterion_value = choose_by_fd(
                    candidates,
                    selected,
                    rss_library,
                    neighbor_pairs,
                    R_DIF_DB,
                )

                criterion = "FD_only"
                candidate_scope = "all_unselected_APs"
                pool_size = int(len(candidates))

        else:
            raise ValueError(
                f"未知 variant: {variant}"
            )

        selected.append(best_idx)

        trace.append(
            SelectionTrace(
                variant=variant,
                init_id=init_id,
                step=int(ap_num),

                selected_index=int(best_idx),
                selected_x=float(
                    ref_xy[best_idx, 0]
                ),
                selected_y=float(
                    ref_xy[best_idx, 1]
                ),

                current_entropy=float(current_h),
                current_state_count=int(current_states),

                max_ig_after=(
                    float(max_ig_after)
                    if max_ig_after is not None
                    else None
                ),

                selected_ig_after=(
                    float(selected_ig_after)
                    if selected_ig_after is not None
                    else None
                ),

                max_delta_ig=(
                    float(max_delta)
                    if max_delta is not None
                    else None
                ),

                selected_delta_ig=(
                    float(selected_delta)
                    if selected_delta is not None
                    else None
                ),

                candidate_pool_size=pool_size,

                criterion=criterion,

                criterion_value=(
                    float(criterion_value)
                    if criterion_value is not None
                    else None
                ),

                candidate_scope=candidate_scope,
            )
        )

    return selected, trace


# ============================================================
# 定位性能
# ============================================================
@dataclass
class Metrics:
    internal_name: str
    variant: str
    init_id: int | None
    feature_mode: str
    seed: int

    mean_error: float
    median_error: float
    radial_rmse: float
    p90: float
    p95: float
    max_error: float


def evaluate_fingerprints(
    fps: Dict[str, np.ndarray],
    internal_meta: Dict[str, tuple[str, int | None]],
    points_xy: np.ndarray,
    seed: int,
    noise_sigma_db: float,
    n_measurements: int,
    k: int,
    n_test: int,
    feature_mode: str,
) -> list[Metrics]:

    rng_select = np.random.default_rng(seed)

    test_ids = rng_select.choice(
        len(points_xy),
        min(n_test, len(points_xy)),
        replace=False,
    )

    true_pos = points_xy[test_ids]

    rows: list[Metrics] = []

    for internal_name, fp in fps.items():

        variant, init_id = internal_meta[
            internal_name
        ]

        # 所有布局使用同一噪声 realization
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

        nx.fit(
            fp,
            points_xy[:, 0],
        )

        ny.fit(
            fp,
            points_xy[:, 1],
        )

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

        rows.append(
            Metrics(
                internal_name=internal_name,
                variant=variant,
                init_id=init_id,
                feature_mode=feature_mode,
                seed=int(seed),

                mean_error=float(e.mean()),
                median_error=float(np.median(e)),
                radial_rmse=float(
                    np.sqrt(
                        np.mean(e**2)
                    )
                ),

                p90=float(
                    np.percentile(e, 90)
                ),

                p95=float(
                    np.percentile(e, 95)
                ),

                max_error=float(e.max()),
            )
        )

    return rows


def collapse_random_initializations(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    固定方案：每个 test seed 只有一条记录。

    w/o-IG：
    每个 test seed 有多个随机 AP1 初始化。
    先对这些初始化的指标取平均，
    得到一个与 Full 可公平配对的 seed-level 指标。
    """
    fixed = raw_df[
        raw_df["variant"] != "w/o-IG"
    ].copy()

    random_df = raw_df[
        raw_df["variant"] == "w/o-IG"
    ].copy()

    metric_cols = [
        "mean_error",
        "median_error",
        "radial_rmse",
        "p90",
        "p95",
        "max_error",
    ]

    if len(random_df):
        random_seed_mean = (
            random_df.groupby(
                [
                    "feature_mode",
                    "seed",
                ]
            )[metric_cols]
            .mean()
            .reset_index()
        )

        random_seed_mean[
            "internal_name"
        ] = "w/o-IG_随机初始化平均"

        random_seed_mean[
            "variant"
        ] = "w/o-IG"

        random_seed_mean[
            "init_id"
        ] = np.nan

        random_seed_mean = random_seed_mean[
            fixed.columns
        ]

        fixed = pd.concat(
            [
                fixed,
                random_seed_mean,
            ],
            ignore_index=True,
        )

    return fixed


def summarize_seed_level(
    seed_level_df: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        seed_level_df.groupby(
            [
                "feature_mode",
                "variant",
            ]
        )
        .agg(
            mean_error_avg=(
                "mean_error",
                "mean",
            ),

            mean_error_std=(
                "mean_error",
                "std",
            ),

            median_error_avg=(
                "median_error",
                "mean",
            ),

            radial_rmse_avg=(
                "radial_rmse",
                "mean",
            ),

            radial_rmse_std=(
                "radial_rmse",
                "std",
            ),

            p90_avg=(
                "p90",
                "mean",
            ),

            p95_avg=(
                "p95",
                "mean",
            ),

            max_error_avg=(
                "max_error",
                "mean",
            ),
        )
        .reset_index()
    )

    order_map = {
        name: i
        for i, name
        in enumerate(DISPLAY_ORDER)
    }

    summary["_order"] = summary[
        "variant"
    ].map(order_map)

    summary = summary.sort_values(
        [
            "feature_mode",
            "_order",
        ]
    ).drop(
        columns="_order"
    )

    return summary


def summarize_random_ap1(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    w/o IG 的初始化敏感性：
    每个随机 AP1 先在全部 test seeds 上取平均。
    """
    sub = raw_df[
        raw_df["variant"] == "w/o-IG"
    ].copy()

    if not len(sub):
        return pd.DataFrame()

    return (
        sub.groupby(
            [
                "feature_mode",
                "init_id",
            ]
        )
        .agg(
            mean_error_avg=(
                "mean_error",
                "mean",
            ),

            mean_error_std=(
                "mean_error",
                "std",
            ),

            radial_rmse_avg=(
                "radial_rmse",
                "mean",
            ),

            p90_avg=(
                "p90",
                "mean",
            ),

            p95_avg=(
                "p95",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "feature_mode",
                "mean_error_avg",
            ]
        )
    )


# ============================================================
# Analytic fallback
# ============================================================
def build_analytic_fingerprints(
    layouts: Dict[str, np.ndarray],
    eval_xyz: np.ndarray,
    wall: WallModel,
) -> Dict[str, np.ndarray]:

    eval_xy = eval_xyz[:, :2]

    return {
        name: np.column_stack(
            [
                screening_rss(
                    ap,
                    eval_xy,
                    wall,
                )
                for ap in aps
            ]
        )
        for name, aps in layouts.items()
    }


# ============================================================
# DeepMIMO-style 评估
# ============================================================
def _import_deepmimo():
    try:
        import deepmimo as dm
        return dm

    except Exception as exc:
        raise SystemExit(
            "选择了 --evaluation deepmimo，"
            "但当前环境未检测到 DeepMIMO。\n"
            "请在 DeepMIMO 4.0.3 环境运行，"
            "或使用 --evaluation analytic。"
        ) from exc


def az_el(
    vec: np.ndarray,
) -> tuple[float, float]:

    vec = np.asarray(
        vec,
        dtype=float,
    )

    az = math.degrees(
        math.atan2(
            vec[1],
            vec[0],
        )
    )

    horiz = math.hypot(
        vec[0],
        vec[1],
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

    out = np.asarray(
        tx,
        dtype=float,
    ).copy()

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

    tm = mirrored_tx(
        tx,
        wall_name,
    )

    if wall_name in ("x0", "xW"):

        plane = (
            0.0
            if wall_name == "x0"
            else ROOM_W
        )

        denom = rx[0] - tm[0]

        t = (
            0.5
            if abs(denom) < 1e-12
            else (plane - tm[0]) / denom
        )

    else:

        plane = (
            0.0
            if wall_name == "y0"
            else ROOM_H
        )

        denom = rx[1] - tm[1]

        t = (
            0.5
            if abs(denom) < 1e-12
            else (plane - tm[1]) / denom
        )

    t = float(
        np.clip(t, 0.0, 1.0)
    )

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

    distance = max(
        float(distance),
        D0,
    )

    return (
        PR0_DBM
        - 10.0
        * float(exponent)
        * math.log10(
            distance / D0
        )
    )


def build_path_matrices(
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    wall: WallModel,
    exponent: float,
    reflection_losses_db: Sequence[float],
) -> tuple[np.ndarray, ...]:

    n = len(rx_xyz)
    n_paths = 1 + len(
        reflection_losses_db
    )

    mats = [
        np.zeros(
            (n, n_paths),
            dtype=float,
        )
        for _ in range(7)
    ]

    for i, rx in enumerate(rx_xyz):

        path_specs = [
            (None, 0.0)
        ]

        path_specs += [
            (
                reflection_point(
                    tx_xyz,
                    rx,
                    wall_name,
                ),
                float(loss),
            )
            for wall_name, loss
            in zip(
                OUTER_WALLS,
                reflection_losses_db,
            )
        ]

        for k0, (
            q,
            extra_loss,
        ) in enumerate(path_specs):

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

            aod_az, aod_el = az_el(
                aod_vec
            )

            aoa_az, aoa_el = az_el(
                aoa_vec
            )

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
            "rx_vel": np.zeros(
                (len(rx_xyz), 3),
                dtype=float,
            ),
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


def total_path_power_dbm(
    dataset,
) -> np.ndarray:

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

    cp.bs_antenna.shape = [
        int(n_ant),
        1,
    ]

    cp.ue_antenna.shape = [
        1,
        1,
    ]

    cp.ofdm.subcarriers = int(
        n_subcarriers
    )

    cp.ofdm.selected_subcarriers = np.arange(
        int(n_subcarriers)
    )

    cp.ofdm.bandwidth = 20e6

    cp.num_paths = int(
        np.asarray(
            dataset.power
        ).shape[1]
    )

    cp.freq_domain = 1

    h = np.asarray(
        dataset.compute_channels(cp)
    )

    p = np.mean(
        np.abs(h) ** 2,
        axis=tuple(
            range(
                1,
                h.ndim,
            )
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
    exponent: float,
    reflection_profile: str,
    feature_mode: str,
) -> tuple[
    Dict[str, np.ndarray],
    str,
]:

    dm = _import_deepmimo()

    losses = REFLECTION_PROFILES[
        reflection_profile
    ]

    # 缓存每个唯一AP，避免 w/o IG 30套布局重复生成信道
    unique_ap_features: dict[
        tuple[float, float],
        np.ndarray,
    ] = {}

    for aps in layouts.values():
        for ap in aps:

            key = (
                float(ap[0]),
                float(ap[1]),
            )

            if key in unique_ap_features:
                continue

            ds = make_deepmimo_dataset(
                dm,
                ap,
                rx_xyz,
                wall,
                exponent,
                losses,
            )

            if feature_mode == "rss_power":

                unique_ap_features[
                    key
                ] = total_path_power_dbm(
                    ds
                )

            elif feature_mode == "csi_rss":

                unique_ap_features[
                    key
                ] = csi_derived_rss_db(
                    dm,
                    ds,
                    n_ant=4,
                    n_subcarriers=32,
                )

            else:
                raise ValueError(
                    feature_mode
                )

    fingerprints = {
        name: np.column_stack(
            [
                unique_ap_features[
                    (
                        float(ap[0]),
                        float(ap[1]),
                    )
                ]
                for ap in aps
            ]
        )
        for name, aps
        in layouts.items()
    }

    version = str(
        getattr(
            dm,
            "__version__",
            "unknown",
        )
    )

    return fingerprints, version


# ============================================================
# 配对统计
# ============================================================
def paired_vs_full(
    seed_level_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in seed_level_df[
        "feature_mode"
    ].drop_duplicates():

        sub = seed_level_df[
            seed_level_df[
                "feature_mode"
            ]
            == feature
        ]

        full = sub[
            sub["variant"]
            == "Full-MIC-FD"
        ].sort_values("seed")

        full_map = dict(
            zip(
                full["seed"],
                full["mean_error"],
            )
        )

        for variant in DISPLAY_ORDER:

            if variant == "Full-MIC-FD":
                continue

            alt = sub[
                sub["variant"]
                == variant
            ].sort_values("seed")

            alt_map = dict(
                zip(
                    alt["seed"],
                    alt["mean_error"],
                )
            )

            common = sorted(
                set(full_map)
                & set(alt_map)
            )

            if not common:
                continue

            a = np.asarray(
                [
                    float(alt_map[s])
                    for s in common
                ],
                dtype=float,
            )

            f = np.asarray(
                [
                    float(full_map[s])
                    for s in common
                ],
                dtype=float,
            )

            diff = a - f

            try:
                if np.allclose(diff, 0.0):
                    stat = 0.0
                    p = 1.0
                else:
                    stat, p = wilcoxon(diff)
                    stat = float(stat)
                    p = float(p)

            except Exception:
                stat = np.nan
                p = np.nan

            rows.append(
                {
                    "feature_mode": feature,
                    "ablation_variant": variant,
                    "full_variant": "Full-MIC-FD",
                    "n_paired_seeds": len(common),

                    "ablation_mean_error_avg": float(
                        a.mean()
                    ),

                    "full_mean_error_avg": float(
                        f.mean()
                    ),

                    "mean_ablation_minus_full_m": float(
                        diff.mean()
                    ),

                    "full_win_rate": float(
                        np.mean(f < a)
                    ),

                    "ablation_win_rate": float(
                        np.mean(a < f)
                    ),

                    "relative_full_improvement_pct": float(
                        (
                            a.mean()
                            - f.mean()
                        )
                        / a.mean()
                        * 100.0
                    )
                    if a.mean() > 0
                    else np.nan,

                    "wilcoxon_stat": stat,
                    "wilcoxon_p": p,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 主程序
# ============================================================
def main() -> None:

    parser = argparse.ArgumentParser(
        description="实验1：MIC-FD五种性能方案最终版"
    )

    parser.add_argument(
        "--output-dir",
        default="结果_实验1_MIC-FD五种性能方案_最终版",
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
        choices=list(
            REFLECTION_PROFILES
        ),
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
        "--random-ap1-count",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--random-ap1-seed",
        type=int,
        default=2000,
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

    args = parser.parse_args()

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

    out = Path(
        args.output_dir
    )

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
            "有效参考点应为147，当前为："
            f"{len(selection_xy)} / "
            f"{len(eval_xyz)}"
        )

    print("=" * 96)
    print("实验1：MIC-FD 五种性能方案（最终定义）")
    print("=" * 96)
    print("1) IG-only       : IG -> IG -> IG -> IG")
    print("2) IG+MIC        : IG -> MIC -> MIC -> MIC")
    print("3) IG+FD         : IG -> IG -> FD -> FD")
    print("4) Full-MIC-FD   : IG -> (DeltaIG+MIC) -> FD -> FD")
    print("5) w/o-IG        : Random -> (DeltaIG+MIC) -> FD -> FD")
    print("-" * 96)
    print("AP筛选使用DeepMIMO: False")
    print("RSS状态: Raw_exact")
    print("固定IG初始化AP1: (0.0,0.0)")
    print(f"D_th = {D_TH:.2f} m")
    print(f"R_dif = {R_DIF_DB:.2f} dB")
    print("MIC实现: original abs(Pearson) surrogate")
    print(
        f"随机AP1初始化: "
        f"{args.random_ap1_count}个不同位置, "
        f"seed={args.random_ap1_seed}"
    )
    print(
        f"正式test seeds: "
        f"{test_seeds[0]}...{test_seeds[-1]}"
    )
    print("Validation seeds: 100...129（不再使用）")
    print(f"Evaluation: {args.evaluation}")
    print(f"Feature mode: {args.feature_mode}")
    print("=" * 96)

    # --------------------------------------------------------
    # RSS library
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
    # 固定五组
    # --------------------------------------------------------
    layouts: Dict[
        str,
        np.ndarray,
    ] = {}

    internal_meta: Dict[
        str,
        tuple[str, int | None],
    ] = {}

    layout_rows = []
    trace_rows = []

    for variant in FIXED_VARIANTS:

        selected, trace = select_layout(
            variant,
            selection_xy,
            rss_library,
        )

        internal_name = variant

        layouts[
            internal_name
        ] = selection_xy[
            selected
        ]

        internal_meta[
            internal_name
        ] = (
            variant,
            None,
        )

        for order, (
            idx,
            pos,
        ) in enumerate(
            zip(
                selected,
                layouts[internal_name],
            ),
            start=1,
        ):

            layout_rows.append(
                {
                    "variant": variant,
                    "init_id": np.nan,
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

    # --------------------------------------------------------
    # w/o IG：30个不同随机AP1
    # --------------------------------------------------------
    if not (
        1
        <= args.random_ap1_count
        <= len(selection_xy)
    ):
        raise ValueError(
            "--random-ap1-count 必须在 1 到 147 之间"
        )

    rng_ap1 = np.random.default_rng(
        args.random_ap1_seed
    )

    random_ap1_indices = rng_ap1.choice(
        len(selection_xy),
        size=args.random_ap1_count,
        replace=False,
    )

    for init_id, ap1_idx in enumerate(
        random_ap1_indices,
        start=1,
    ):

        selected, trace = select_layout(
            "w/o-IG",
            selection_xy,
            rss_library,
            random_ap1_idx=int(ap1_idx),
            init_id=int(init_id),
        )

        internal_name = (
            f"w/o-IG_init{init_id:02d}"
        )

        layouts[
            internal_name
        ] = selection_xy[
            selected
        ]

        internal_meta[
            internal_name
        ] = (
            "w/o-IG",
            int(init_id),
        )

        for order, (
            idx,
            pos,
        ) in enumerate(
            zip(
                selected,
                layouts[internal_name],
            ),
            start=1,
        ):

            layout_rows.append(
                {
                    "variant": "w/o-IG",
                    "init_id": int(init_id),
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
        out / "五种性能方案_AP布局.csv",
        index=False,
        encoding="utf-8-sig",
    )

    trace_df.to_csv(
        out / "五种性能方案_AP筛选过程.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 四组确定性布局 ===")

    print(
        layout_df[
            layout_df["variant"]
            != "w/o-IG"
        ].to_string(
            index=False
        )
    )

    print("\n=== w/o IG 随机AP1布局（前5组初始化） ===")

    print(
        layout_df[
            (
                layout_df["variant"]
                == "w/o-IG"
            )
            & (
                layout_df["init_id"]
                <= 5
            )
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # feature modes
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

    all_raw_metrics = []
    all_seed_level = []
    all_summary = []

    deepmimo_version = None

    for feature in feature_modes:

        print(
            f"\n构建 {feature} 指纹..."
        )

        if args.evaluation == "analytic":

            fps = build_analytic_fingerprints(
                layouts,
                eval_xyz,
                wall,
            )

        else:

            (
                fps,
                deepmimo_version,
            ) = build_deepmimo_fingerprints(
                layouts=layouts,
                rx_xyz=eval_xyz,
                wall=wall,
                exponent=PATH_LOSS_EXPONENT,
                reflection_profile=args.reflection_profile,
                feature_mode=feature,
            )

        repeated: list[Metrics] = []

        for seed in test_seeds:

            repeated.extend(
                evaluate_fingerprints(
                    fps=fps,
                    internal_meta=internal_meta,
                    points_xy=eval_xyz[:, :2],
                    seed=seed,
                    noise_sigma_db=args.noise_sigma_db,
                    n_measurements=args.n_measurements,
                    k=args.k,
                    n_test=args.n_test,
                    feature_mode=feature,
                )
            )

        raw_df = pd.DataFrame(
            [
                asdict(x)
                for x in repeated
            ]
        )

        raw_df.to_csv(
            out
            / f"原始结果_{feature}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        seed_level_df = collapse_random_initializations(
            raw_df
        )

        seed_level_df.to_csv(
            out
            / f"按test_seed汇总_{feature}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary = summarize_seed_level(
            seed_level_df
        )

        summary.to_csv(
            out
            / f"五种性能方案30seed汇总_{feature}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        random_summary = summarize_random_ap1(
            raw_df
        )

        random_summary.to_csv(
            out
            / f"w-o-IG随机初始化敏感性_{feature}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\n=== {feature} 五种性能方案结果 ==="
        )

        print(
            summary.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
            )
        )

        if len(random_summary):

            print(
                "\n=== w/o IG 随机初始化统计（前10个） ==="
            )

            print(
                random_summary.head(10).to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:.4f}"
                    ),
                )
            )

        all_raw_metrics.append(
            raw_df
        )

        all_seed_level.append(
            seed_level_df
        )

        all_summary.append(
            summary
        )

    # --------------------------------------------------------
    # 总表
    # --------------------------------------------------------
    raw_all = pd.concat(
        all_raw_metrics,
        ignore_index=True,
    )

    seed_level_all = pd.concat(
        all_seed_level,
        ignore_index=True,
    )

    summary_all = pd.concat(
        all_summary,
        ignore_index=True,
    )

    summary_all.to_csv(
        out / "五种性能方案_最终汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Full vs 其他组
    # --------------------------------------------------------
    paired = paired_vs_full(
        seed_level_all
    )

    paired.to_csv(
        out / "Full对其他四组_配对统计.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== Full 对其他四组的配对统计 ==="
    )

    print(
        paired.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # --------------------------------------------------------
    # 核心机制对照表
    # --------------------------------------------------------
    mechanism_rows = []

    for feature in summary_all[
        "feature_mode"
    ].drop_duplicates():

        sub = summary_all[
            summary_all[
                "feature_mode"
            ]
            == feature
        ]

        m = dict(
            zip(
                sub["variant"],
                sub["mean_error_avg"],
            )
        )

        comparisons = [
            (
                "IG作用",
                "w/o-IG",
                "Full-MIC-FD",
            ),
            (
                "MIC持续选择 vs IG-only",
                "IG-only",
                "IG+MIC",
            ),
            (
                "FD后期选择 vs IG-only",
                "IG-only",
                "IG+FD",
            ),
        ]

        for effect, baseline, enhanced in comparisons:

            if (
                baseline not in m
                or enhanced not in m
            ):
                continue

            b = float(m[baseline])
            e = float(m[enhanced])

            mechanism_rows.append(
                {
                    "feature_mode": feature,
                    "effect": effect,
                    "baseline": baseline,
                    "enhanced": enhanced,
                    "baseline_mean_error": b,
                    "enhanced_mean_error": e,
                    "improvement_pct": (
                        (b - e)
                        / b
                        * 100.0
                        if b > 0
                        else np.nan
                    ),
                }
            )

    mechanism_df = pd.DataFrame(
        mechanism_rows
    )

    mechanism_df.to_csv(
        out / "机制消融作用分析.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== 机制消融作用分析 ==="
    )

    print(
        mechanism_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # 配置
    # --------------------------------------------------------
    config = {
        "实验": "MIC-FD五种性能方案最终版",

        "五种性能方案定义": {
            "IG-only": (
                "IG -> IG -> IG -> IG"
            ),

            "IG+MIC": (
                "IG -> MIC -> MIC -> MIC"
            ),

            "IG+FD": (
                "IG -> IG -> FD -> FD"
            ),

            "Full-MIC-FD": (
                "IG -> (DeltaIG + MIC) -> FD -> FD"
            ),

            "w/o-IG": (
                "Random -> (DeltaIG + MIC) -> FD -> FD"
            ),
        },

        "重要定义": {
            "IG": (
                "直接最大化加入候选后的联合熵 H；"
                "前三个基准不标记为DeltaIG"
            ),

            "DeltaIG": (
                "只作为Full AP2的信息增益候选筛选机制；"
                "w/o-IG保留它是为了只消融IG；"
                "本脚本不再把w/o-DeltaIG作为性能方案"
            ),

            "MIC": (
                "当前实现为 original abs(Pearson) surrogate，"
                "不是真MIC"
            ),
        },

        "锁定参数": {
            "D_th_m": D_TH,
            "R_dif_db": R_DIF_DB,
            "validation_seeds": list(
                range(100, 130)
            ),
            "重新调参": False,
        },

        "w/o_IG": {
            "random_ap1_count": int(
                args.random_ap1_count
            ),

            "random_ap1_seed": int(
                args.random_ap1_seed
            ),

            "random_ap1_indices": [
                int(x)
                for x
                in random_ap1_indices
            ],

            "统计方式": (
                "每个test seed先对全部随机AP1初始化取平均，"
                "再与Full做seed-level配对比较"
            ),
        },

        "评估": {
            "mode": args.evaluation,
            "feature_mode": args.feature_mode,
            "deepmimo_version": deepmimo_version,
            "reflection_profile": (
                args.reflection_profile
                if args.evaluation
                == "deepmimo"
                else None
            ),

            "test_seeds": test_seeds,
            "noise_sigma_db": (
                args.noise_sigma_db
            ),
            "n_measurements": (
                args.n_measurements
            ),
            "n_test": (
                args.n_test
            ),
            "K": args.k,

            "same_test_rps_and_noise": True,

            "csi_rss_note": (
                "CSI-derived RSS + KNN，"
                "不是 amplitude/phase + 3D CNN"
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

    print(
        "\n结果目录:",
        out.resolve(),
    )


if __name__ == "__main__":
    main()
