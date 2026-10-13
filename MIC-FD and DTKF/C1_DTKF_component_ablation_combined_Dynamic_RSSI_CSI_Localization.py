#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor

# ============================== 固定参数 ==============================
C0 = 299_792_458.0
FC_HZ = 3.5e9
LAMBDA = C0 / FC_HZ
ROOM_W, ROOM_H = 12.0, 8.0
RX_Z = TX_Z = 1.5
D0, PR0_DBM = 1.0, -40.0
PATH_LOSS_EXPONENT = 2.2
WALL_LOSS_DB = 10.0
GRID_STEP = 0.5
RSS_NOISE_SIGMA_DB = 3.0
RSS_N_MEASUREMENTS = 10
RSS_K = 5
D_TH = 1.25
R_DIF_DB = 4.0
REFLECTION_PROFILES = {
    "none": (),
    "very_weak": (20.0, 22.0, 24.0, 26.0),
    "weak": (14.0, 16.0, 18.0, 20.0),
    "moderate": (9.0, 11.0, 13.0, 15.0),
}
OUTER_WALLS = ("x0", "xW", "y0", "yH")


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ============================== 场景 ==============================
@dataclass(frozen=True)
class SolidRect:
    x0: float
    x1: float
    y0: float
    y1: float
    group: str


class LargeRoomWallModel:
    def __init__(self, wall_loss_db: float = WALL_LOSS_DB):
        self.wall_loss_db = float(wall_loss_db)
        self.vertical_x0, self.vertical_x1 = 3.9, 4.1
        self.vertical_door_y0, self.vertical_door_y1 = 3.0, 4.5
        self.horizontal_y0, self.horizontal_y1 = 3.9, 4.1
        self.horizontal_x0, self.horizontal_x1 = 4.0, 12.0
        self.horizontal_door_x0, self.horizontal_door_x1 = 7.0, 8.5
        self.solids = [
            SolidRect(3.9, 4.1, 0.0, 3.0, "vertical"),
            SolidRect(3.9, 4.1, 4.5, 8.0, "vertical"),
            SolidRect(4.0, 7.0, 3.9, 4.1, "horizontal"),
            SolidRect(8.5, 12.0, 3.9, 4.1, "horizontal"),
        ]

    @staticmethod
    def _inside(x: float, y: float, r: SolidRect) -> bool:
        return r.x0 - 1e-12 <= x <= r.x1 + 1e-12 and r.y0 - 1e-12 <= y <= r.y1 + 1e-12

    def is_inside_wall(self, x: float, y: float) -> bool:
        return any(self._inside(float(x), float(y), r) for r in self.solids)

    @staticmethod
    def _segment_intersects_rect(p1: Sequence[float], p2: Sequence[float], r: SolidRect) -> bool:
        x0, y0 = float(p1[0]), float(p1[1])
        x1, y1 = float(p2[0]), float(p2[1])
        dx, dy = x1 - x0, y1 - y0
        p = (-dx, dx, -dy, dy)
        q = (x0 - r.x0, r.x1 - x0, y0 - r.y0, r.y1 - y0)
        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if abs(pi) < 1e-15:
                if qi < 0:
                    return False
                continue
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return False
        return True

    def count_crossings(self, p1: Sequence[float], p2: Sequence[float]) -> int:
        groups = {r.group for r in self.solids if self._segment_intersects_rect(p1, p2, r)}
        return len(groups)


def make_reference_points(wall: LargeRoomWallModel, step: float) -> np.ndarray:
    pts = []
    for x in np.arange(0.0, ROOM_W + step / 2, step):
        for y in np.arange(0.0, ROOM_H + step / 2, step):
            if not wall.is_inside_wall(float(x), float(y)):
                pts.append([float(x), float(y), RX_Z])
    return np.asarray(pts, dtype=float)


def _resample_polyline(waypoints: np.ndarray, n_steps: int) -> np.ndarray:
    """按弧长等间距重采样折线，保持速度基本均匀。"""
    waypoints = np.asarray(waypoints, dtype=float)
    lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("轨迹waypoint存在重复点")
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    s_values = np.linspace(0.0, cum[-1], int(n_steps))
    out = np.zeros((len(s_values), 2), float)
    seg = 0
    for i, s in enumerate(s_values):
        while seg < len(lengths) - 1 and s > cum[seg + 1]:
            seg += 1
        t = (s - cum[seg]) / max(lengths[seg], 1e-12)
        out[i] = (1 - t) * waypoints[seg] + t * waypoints[seg + 1]
    return out


def make_dynamic_trajectory(n_steps: int) -> np.ndarray:
    """
    正式测试轨迹：按用户标注的红线重新设计。

    关键改动：
    1) 左下区域：从 (0.5,0.5) 斜向纵墙门口，而不是沿左侧/上侧绕行；
    2) 进入右下区域后：先沿 x≈4.5 向下，再沿 y≈0.65 向右，
       形成用户标注的“竖直 + 底部水平”L形路线；
    3) 其余仍通过两个门洞完成三分区闭环，不穿实体墙。
    """
    waypoints = np.asarray([
        [0.50, 0.50],
        [1.30, 1.30],
        [2.40, 2.30],
        [3.50, 3.60],
        [4.50, 3.60],   # 纵墙门洞：左 -> 右下

        # 红线：门口下行 + 右下方水平走廊
        [4.50, 0.65],
        [11.40, 0.65],
        [11.40, 3.50],
        [8.40, 3.50],
        [7.75, 3.50],
        [7.75, 4.50],   # 横墙门洞：右下 -> 右上

        [8.50, 4.80],
        [11.40, 7.40],
        [7.80, 7.40],
        [5.00, 7.40],
        [4.50, 4.30],
        [3.50, 4.30],   # 纵墙门洞：右上 -> 左区
        [3.40, 7.40],
        [0.50, 7.40],
        [0.50, 0.50],
    ], dtype=float)
    return _resample_polyline(waypoints, n_steps)


def make_validation_dynamic_trajectory(n_steps: int) -> np.ndarray:
    """
    独立动态验证轨迹。

    与正式测试轨迹具有相同的运动类型（斜线、直角转弯、两个门洞），
    但坐标和转弯位置不同。它只用于选择 DTKF 数值参数，不使用正式测试真值。
    """
    waypoints = np.asarray([
        [0.90, 0.90],
        [1.80, 1.55],
        [3.45, 3.45],
        [4.45, 3.45],
        [4.70, 1.05],
        [10.70, 1.05],
        [10.70, 3.35],
        [8.05, 3.35],
        [8.05, 4.45],
        [9.00, 5.10],
        [10.60, 6.80],
        [7.20, 6.85],
        [5.10, 6.45],
        [4.45, 4.25],
        [3.55, 4.25],
        [2.90, 6.70],
        [0.95, 6.70],
        [0.90, 0.90],
    ], dtype=float)
    return _resample_polyline(waypoints, n_steps)

def verify_trajectory(xy: np.ndarray, wall: LargeRoomWallModel) -> None:
    bad = [i for i, p in enumerate(xy) if wall.is_inside_wall(p[0], p[1])]
    if bad:
        raise RuntimeError(f"轨迹进入墙体，前几个索引：{bad[:10]}")


# ============================== MIC-FD AP筛选 ==============================
def screening_rss(ap_xy: np.ndarray, rp_xy: np.ndarray, wall: LargeRoomWallModel) -> np.ndarray:
    out = np.zeros(len(rp_xy), float)
    for i, rp in enumerate(rp_xy):
        d = max(float(np.linalg.norm(rp - ap_xy)), D0 / 10.0)
        rss = PR0_DBM - 10.0 * PATH_LOSS_EXPONENT * np.log10(d / D0)
        rss -= wall.count_crossings(ap_xy, rp) * wall.wall_loss_db
        out[i] = rss
    return out


def raw_entropy(fp: np.ndarray) -> tuple[float, int]:
    if fp.ndim == 1:
        fp = fp[:, None]
    _, counts = np.unique(fp, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p))), int(len(counts))


def pseudo_mic(a: np.ndarray, b: np.ndarray) -> float:
    try:
        value = abs(float(pearsonr(a, b)[0]))
        return 0.5 if np.isnan(value) else value
    except Exception:
        return 0.5


def build_neighbor_pairs(xy: np.ndarray, d_th: float) -> np.ndarray:
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    return np.argwhere((d > 1e-12) & (d <= d_th + 1e-12))


def fd_score(candidate: int, selected: list[int], rss_lib: np.ndarray, pairs: np.ndarray, r_dif: float) -> int:
    fp = rss_lib[:, selected + [candidate]]
    dif = np.abs(fp[pairs[:, 0]] - fp[pairs[:, 1]])
    return int(np.sum(np.max(dif, axis=1) >= r_dif))


def select_mic_fd_aps(candidate_xy: np.ndarray, wall: LargeRoomWallModel) -> tuple[np.ndarray, pd.DataFrame]:
    rss_lib = np.column_stack([screening_rss(ap, candidate_xy, wall) for ap in candidate_xy])
    trace = []

    # AP1：最大IG
    h1 = np.asarray([raw_entropy(rss_lib[:, j])[0] for j in range(len(candidate_xy))])
    max_h = float(h1.max())
    pool1 = np.where(np.abs(h1 - max_h) < 1e-6)[0]
    ap1 = int(pool1.min())
    selected = [ap1]
    trace.append(dict(step=1, criterion="IG", candidate_pool=len(pool1), selected_index=ap1,
                      x=candidate_xy[ap1, 0], y=candidate_xy[ap1, 1], score=max_h))

    # AP2：最大DeltaIG候选 + pseudo-MIC
    current_h = raw_entropy(rss_lib[:, selected])[0]
    delta = np.full(len(candidate_xy), -np.inf)
    for j in range(len(candidate_xy)):
        if j not in selected:
            delta[j] = raw_entropy(rss_lib[:, selected + [j]])[0] - current_h
    max_delta = float(delta.max())
    pool2 = np.where(np.abs(delta - max_delta) < 1e-6)[0]
    mic = np.asarray([pseudo_mic(rss_lib[:, ap1], rss_lib[:, j]) for j in pool2])
    ap2 = int(pool2[int(np.argmin(mic))])
    selected.append(ap2)
    trace.append(dict(step=2, criterion="DeltaIG+MIC", candidate_pool=len(pool2), selected_index=ap2,
                      x=candidate_xy[ap2, 0], y=candidate_xy[ap2, 1], score=float(mic.min())))

    pairs = build_neighbor_pairs(candidate_xy, D_TH)
    for step in (3, 4):
        candidates = np.asarray([j for j in range(len(candidate_xy)) if j not in selected], int)
        scores = np.asarray([fd_score(int(j), selected, rss_lib, pairs, R_DIF_DB) for j in candidates], float)
        chosen = int(candidates[int(np.argmax(scores))])
        selected.append(chosen)
        trace.append(dict(step=step, criterion="FD", candidate_pool=len(candidates), selected_index=chosen,
                          x=candidate_xy[chosen, 0], y=candidate_xy[chosen, 1], score=float(scores.max())))

    return candidate_xy[selected].copy(), pd.DataFrame(trace)


# ============================== DeepMIMO-style CSI ==============================
def import_stack():
    try:
        import deepmimo as dm
    except Exception as exc:
        raise SystemExit("完整实验需要 DeepMIMO：pip install deepmimo") from exc
    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit("CSI 3D CNN需要 TensorFlow：pip install tensorflow") from exc
    return dm, tf


def az_el(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, float)
    return math.degrees(math.atan2(v[1], v[0])), math.degrees(math.atan2(v[2], max(math.hypot(v[0], v[1]), 1e-12)))


def mirrored_tx(tx: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(tx, float).copy()
    if name == "x0": out[0] = -out[0]
    elif name == "xW": out[0] = 2 * ROOM_W - out[0]
    elif name == "y0": out[1] = -out[1]
    elif name == "yH": out[1] = 2 * ROOM_H - out[1]
    else: raise ValueError(name)
    return out


def reflection_point(tx: np.ndarray, rx: np.ndarray, name: str) -> np.ndarray:
    m = mirrored_tx(tx, name)
    if name in ("x0", "xW"):
        plane = 0.0 if name == "x0" else ROOM_W
        den = rx[0] - m[0]
        t = 0.5 if abs(den) < 1e-12 else (plane - m[0]) / den
    else:
        plane = 0.0 if name == "y0" else ROOM_H
        den = rx[1] - m[1]
        t = 0.5 if abs(den) < 1e-12 else (plane - m[1]) / den
    t = float(np.clip(t, 0.0, 1.0))
    q = m + t * (rx - m)
    if name == "x0": q[0] = 0.0
    elif name == "xW": q[0] = ROOM_W
    elif name == "y0": q[1] = 0.0
    else: q[1] = ROOM_H
    return q


def path_loss_dbm(distance: float, exponent: float) -> float:
    return PR0_DBM - 10.0 * exponent * math.log10(max(float(distance), D0) / D0)


def build_path_matrices(tx_xyz: np.ndarray, rx_xyz: np.ndarray, wall: LargeRoomWallModel,
                        exponent: float, reflection_losses: Sequence[float]) -> tuple[np.ndarray, ...]:
    mats = [np.zeros((len(rx_xyz), 1 + len(reflection_losses)), float) for _ in range(7)]
    for i, rx in enumerate(rx_xyz):
        paths = [(None, 0.0)] + [(reflection_point(tx_xyz, rx, n), float(loss))
                                 for n, loss in zip(OUTER_WALLS, reflection_losses)]
        for k, (bounce, extra_loss) in enumerate(paths):
            if bounce is None:
                length = float(np.linalg.norm(rx - tx_xyz))
                crossings = wall.count_crossings(tx_xyz, rx)
                aod, aoa = rx - tx_xyz, tx_xyz - rx
            else:
                length = float(np.linalg.norm(bounce - tx_xyz) + np.linalg.norm(rx - bounce))
                crossings = wall.count_crossings(tx_xyz, bounce) + wall.count_crossings(bounce, rx)
                aod, aoa = bounce - tx_xyz, bounce - rx
            p_dbm = path_loss_dbm(length, exponent) - extra_loss - crossings * wall.wall_loss_db
            phase = ((360.0 * length / LAMBDA + 180.0) % 360.0) - 180.0
            aod_az, aod_el = az_el(aod); aoa_az, aoa_el = az_el(aoa)
            mats[0][i, k] = p_dbm - 30.0
            mats[1][i, k] = phase
            mats[2][i, k] = length / C0
            mats[3][i, k], mats[4][i, k] = aod_az, aod_el
            mats[5][i, k], mats[6][i, k] = aoa_az, aoa_el
    return tuple(mats)


def make_dataset(dm, tx_xy: np.ndarray, rx_xyz: np.ndarray, wall: LargeRoomWallModel,
                 exponent: float, losses: Sequence[float]):
    tx_xyz = np.asarray([tx_xy[0], tx_xy[1], TX_Z], float)
    power, phase, delay, aod_az, aod_el, aoa_az, aoa_el = build_path_matrices(tx_xyz, rx_xyz, wall, exponent, losses)
    return dm.Dataset({
        "rx_pos": rx_xyz.astype(float), "tx_pos": tx_xyz, "tx_vel": np.zeros(3),
        "rx_vel": np.zeros((len(rx_xyz), 3)), "scene": dm.Scene(), "power": power,
        "phase": phase, "delay": delay, "aod_az": aod_az, "aod_el": aod_el,
        "aoa_az": aoa_az, "aoa_el": aoa_el,
    })


def compute_complex_csi(dm, dataset, n_ant: int, n_sub: int, bandwidth_hz: float) -> np.ndarray:
    cp = dm.ChannelParameters()
    cp.bs_antenna.shape = [int(n_ant), 1]
    cp.ue_antenna.shape = [1, 1]
    cp.ofdm.subcarriers = int(n_sub)
    cp.ofdm.selected_subcarriers = np.arange(int(n_sub))
    cp.ofdm.bandwidth = float(bandwidth_hz)
    cp.num_paths = int(np.asarray(dataset.power).shape[1])
    cp.freq_domain = 1
    h = dataset.compute_channels(cp)
    if h is None: h = getattr(dataset, "channel", None)
    if h is None: h = getattr(dataset, "channels", None)
    if h is None: raise RuntimeError("DeepMIMO未返回信道")
    h = np.asarray(h)
    if h.ndim != 4: raise RuntimeError(f"DeepMIMO信道形状异常：{h.shape}")
    h = h[:, 0] if h.shape[1] == 1 else h.mean(axis=1)
    return np.asarray(h, np.complex64)


def build_csi(dm, aps_xy: np.ndarray, rx_xyz: np.ndarray, wall: LargeRoomWallModel,
              exponent: float, profile: str, n_ant: int, n_sub: int, bandwidth_hz: float) -> np.ndarray:
    losses = REFLECTION_PROFILES[profile]
    per_ap = []
    for ap in aps_xy:
        ds = make_dataset(dm, ap, rx_xyz, wall, exponent, losses)
        per_ap.append(compute_complex_csi(dm, ds, n_ant, n_sub, bandwidth_hz))
    return np.stack(per_ap, axis=1)


# ============================== RSS ==============================
def power_db_from_csi(csi: np.ndarray) -> np.ndarray:
    p = np.mean(np.abs(csi) ** 2, axis=(-2, -1))
    return 10.0 * np.log10(np.maximum(p, 1e-30))


def rss_predict(fp_db: np.ndarray, rp_xy: np.ndarray, true_power_db: np.ndarray,
                seed: int, sigma_db: float, n_measurements: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = true_power_db[:, None, :] + rng.normal(0.0, sigma_db,
        size=(len(true_power_db), int(n_measurements), true_power_db.shape[1]))
    meas = samples.mean(axis=1)
    k_use = min(int(k), len(fp_db))
    mx = KNeighborsRegressor(n_neighbors=k_use, weights="distance")
    my = KNeighborsRegressor(n_neighbors=k_use, weights="distance")
    mx.fit(fp_db, rp_xy[:, 0]); my.fit(fp_db, rp_xy[:, 1])
    pred = np.column_stack([mx.predict(meas), my.predict(meas)])
    pred[:, 0] = np.clip(pred[:, 0], 0, ROOM_W); pred[:, 1] = np.clip(pred[:, 1], 0, ROOM_H)
    return pred


# ============================== CSI CNN ==============================
def augment_csi(base: np.ndarray, n_samples: int, seed: int,
                amp_noise_db: float, phase_noise_deg: float, awgn_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_pos, n_ap, n_ant, n_sub = base.shape
    shape = (n_pos, int(n_samples), n_ap, n_ant, n_sub)
    amp = 10.0 ** (rng.normal(0.0, amp_noise_db, size=shape) / 20.0)
    phase = rng.normal(0.0, math.radians(phase_noise_deg), size=shape)
    expanded = base[:, None] * amp * np.exp(1j * phase)
    rms = np.sqrt(np.mean(np.abs(base) ** 2, axis=(2, 3), keepdims=True))[:, None]
    noise = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / math.sqrt(2.0)
    expanded = expanded + awgn_ratio * rms * noise
    return np.asarray(expanded.reshape(-1, n_ap, n_ant, n_sub), np.complex64), np.repeat(np.arange(n_pos), int(n_samples))


def csi_to_tensor(csi: np.ndarray) -> np.ndarray:
    mag = np.maximum(np.abs(csi), 1e-12)
    amp_db = 20.0 * np.log10(mag)
    rel_amp = amp_db - amp_db.mean(axis=-1, keepdims=True)
    ratio = csi[..., 1:] * np.conj(csi[..., :-1])
    phase_diff = np.angle(ratio)
    phase_diff = np.concatenate([np.zeros_like(phase_diff[..., :1]), phase_diff], axis=-1)
    ap_power = power_db_from_csi(csi)
    ap_map = np.broadcast_to(ap_power[:, :, None, None], amp_db.shape)
    return np.asarray(np.stack([amp_db, rel_amp, np.sin(phase_diff), np.cos(phase_diff), ap_map], axis=-1), np.float32)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(np.clip((x - self.mean) / self.std, -8, 8), np.float32)


def fit_normalizer(x: np.ndarray) -> Normalizer:
    mean = x.mean(axis=(0, 1, 2, 3), keepdims=True)
    std = np.maximum(x.std(axis=(0, 1, 2, 3), keepdims=True), 1e-6)
    return Normalizer(mean.astype(np.float32), std.astype(np.float32))


def normalize_xy(xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy, np.float32) / np.asarray([ROOM_W, ROOM_H], np.float32)


def denormalize_xy(xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy, np.float32) * np.asarray([ROOM_W, ROOM_H], np.float32)


def create_cnn(tf, input_shape: tuple[int, ...], seed: int, lr: float):
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=input_shape)
    def block(x, filters, kernel, s):
        x = tf.keras.layers.Conv3D(filters, kernel, padding="same", use_bias=False,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=s))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation("swish")(x)
    x = block(inp, 24, (1, 1, 5), seed)
    res = x
    x = block(x, 24, (1, 1, 5), seed + 1)
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.MaxPool3D((1, 1, 2))(x)
    x = block(x, 40, (1, 3, 3), seed + 2)
    x = tf.keras.layers.SpatialDropout3D(0.08, seed=seed + 3)(x)
    x = block(x, 56, (3, 1, 3), seed + 4)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)
    x = block(x, 64, (1, 1, 3), seed + 5)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(192, activation="swish",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 6),
        kernel_regularizer=tf.keras.regularizers.l2(3e-4))(x)
    x = tf.keras.layers.Dropout(0.15, seed=seed + 7)(x)
    x = tf.keras.layers.Dense(96, activation="swish",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 8),
        kernel_regularizer=tf.keras.regularizers.l2(2e-4))(x)
    out = tf.keras.layers.Dense(2, activation="linear",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 9),
        bias_initializer=tf.keras.initializers.Constant([0.5, 0.5]))(x)
    model = tf.keras.Model(inp, out)
    scale = tf.constant([ROOM_W, ROOM_H], tf.float32)
    def loss(y_true, y_pred):
        d = (y_pred - y_true) * scale
        sq = tf.reduce_sum(tf.square(d), axis=-1)
        return tf.reduce_mean(tf.sqrt(sq + 1e-8) + 0.05 * sq)
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0)
    except AttributeError:
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss)
    return model


def split_positions(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n)
    n_val = max(10, int(round(n * val_fraction)))
    return np.sort(ids[n_val:]), np.sort(ids[:n_val])


def aggregate_predictions(pred: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                          wanted: np.ndarray, use_n: int) -> tuple[np.ndarray, np.ndarray]:
    yt, yp = [], []
    for pid in wanted:
        idx = np.flatnonzero(ids == pid)
        if len(idx) == 0: continue
        use = idx[:max(1, min(int(use_n), len(idx)))]
        yt.append(labels[use[0]])
        yp.append(pred[use].mean(axis=0))
    return np.asarray(yt, float), np.asarray(yp, float)


def train_csi(tf, base_csi: np.ndarray, rp_xy: np.ndarray, train_ids: np.ndarray, val_ids: np.ndarray, args):
    samples, ids = augment_csi(base_csi, args.samples_per_position, args.seed + 10000,
        args.amplitude_noise_db, args.phase_noise_deg, args.awgn_ratio)
    labels = rp_xy[ids]
    tensor = csi_to_tensor(samples)
    train_mask, val_mask = np.isin(ids, train_ids), np.isin(ids, val_ids)
    norm = fit_normalizer(tensor[train_mask])
    tensor = norm.transform(tensor)
    best_error, best_weights, best_seed = float("inf"), None, args.seed
    for restart in range(args.restarts):
        rs = args.seed + restart * 1000
        tf.keras.backend.clear_session()
        model = create_cnn(tf, tensor.shape[1:], rs, args.learning_rate)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=2e-6, verbose=0),
        ]
        model.fit(tensor[train_mask], normalize_xy(labels[train_mask]),
            validation_data=(tensor[val_mask], normalize_xy(labels[val_mask])),
            epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=0, shuffle=True)
        raw = denormalize_xy(model.predict(tensor[val_mask], verbose=0))
        val_true, val_pred = aggregate_predictions(raw, labels[val_mask], ids[val_mask], val_ids, args.eval_measurements)
        err = float(np.mean(np.linalg.norm(val_pred - val_true, axis=1)))
        print(f"  CSI restart {restart+1}/{args.restarts}: validation mean={err:.4f} m")
        if err < best_error:
            best_error, best_seed = err, rs
            best_weights = [np.array(w, copy=True) for w in model.get_weights()]
    if best_weights is None: raise RuntimeError("CSI CNN训练失败")
    tf.keras.backend.clear_session()
    model = create_cnn(tf, tensor.shape[1:], best_seed, args.learning_rate)
    model.set_weights(best_weights)
    raw = denormalize_xy(model.predict(tensor[val_mask], verbose=0))
    val_true, val_pred = aggregate_predictions(raw, labels[val_mask], ids[val_mask], val_ids, args.eval_measurements)
    val_pred[:, 0] = np.clip(val_pred[:, 0], 0, ROOM_W); val_pred[:, 1] = np.clip(val_pred[:, 1], 0, ROOM_H)
    return model, norm, val_true, val_pred, best_error


def predict_dynamic_csi(model, norm: Normalizer, base_csi: np.ndarray, args, seed: int | None = None) -> np.ndarray:
    use_seed = args.seed + 41000 if seed is None else int(seed)
    samples, ids = augment_csi(base_csi, args.dynamic_csi_measurements, use_seed,
        args.amplitude_noise_db, args.phase_noise_deg, args.awgn_ratio)
    pred = denormalize_xy(model.predict(norm.transform(csi_to_tensor(samples)), verbose=0))
    out = np.zeros((len(base_csi), 2), float)
    for pid in range(len(base_csi)):
        out[pid] = pred[ids == pid].mean(axis=0)
    out[:, 0] = np.clip(out[:, 0], 0, ROOM_W); out[:, 1] = np.clip(out[:, 1], 0, ROOM_H)
    return out


# ============================== DTKF ==============================
def make_spd(m: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    m = 0.5 * (np.asarray(m, float) + np.asarray(m, float).T)
    values, vectors = np.linalg.eigh(m)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def estimate_bias_cov(true_xy: np.ndarray, pred_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = pred_xy - true_xy
    bias = error.mean(axis=0)
    centered = error - bias
    cov = np.cov(centered, rowvar=False, ddof=1) if len(centered) > 1 else np.eye(2) * 0.1
    return bias, make_spd(cov, 2e-3)


def kf_update(xp: np.ndarray, pp: np.ndarray, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    s = h @ pp @ h.T + r
    k = pp @ h.T @ np.linalg.pinv(s)
    x = xp + k @ (z - h @ xp)
    i = np.eye(4)
    p = (i - k @ h) @ pp @ (i - k @ h).T + k @ r @ k.T
    return x, make_spd(p, 1e-6)


def deduplicate_common_prior(x0, p0, xr, pr, xc, pc):
    l0, lr, lc = np.linalg.pinv(p0), np.linalg.pinv(pr), np.linalg.pinv(pc)
    info = lr + lc - l0
    eta = lr @ xr + lc @ xc - l0 @ x0
    p = make_spd(np.linalg.pinv(info), 1e-6)
    return p @ eta, p


def run_dtkf(rss_xy: np.ndarray, csi_xy: np.ndarray, rss_cov: np.ndarray, csi_cov: np.ndarray,
             threshold: float, dt: float, accel_std: float, inflation: float) -> np.ndarray:
    f = np.asarray([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]], float)
    q = accel_std ** 2 * np.asarray([
        [dt**4/4,0,dt**3/2,0], [0,dt**4/4,0,dt**3/2],
        [dt**3/2,0,dt**2,0], [0,dt**3/2,0,dt**2]], float)
    x = np.asarray([(rss_xy[0,0]+csi_xy[0,0])/2, (rss_xy[0,1]+csi_xy[0,1])/2, 0, 0], float)
    p = np.diag([1.,1.,0.5,0.5]); out = np.zeros((len(rss_xy), 2), float)
    h = np.asarray([[1.,0.,0.,0.],[0.,1.,0.,0.]])
    for i in range(len(rss_xy)):
        if i == 0: xp, pp = x.copy(), p.copy()
        else: xp, pp = f @ x, make_spd(f @ p @ f.T + q, 1e-6)
        rr, rc = rss_cov.copy(), csi_cov.copy()
        if np.linalg.norm(rss_xy[i] - csi_xy[i]) > threshold:
            prior_xy = h @ xp
            if np.linalg.norm(rss_xy[i] - prior_xy) >= np.linalg.norm(csi_xy[i] - prior_xy): rr *= inflation
            else: rc *= inflation
        xr, pr = kf_update(xp, pp, rss_xy[i], rr)
        xc, pc = kf_update(xp, pp, csi_xy[i], rc)
        x, p = deduplicate_common_prior(xp, pp, xr, pr, xc, pc)
        x[0] = np.clip(x[0], 0, ROOM_W); x[1] = np.clip(x[1], 0, ROOM_H)
        out[i] = x[:2]
    return out


def parse_float_candidates(text: str) -> list[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("候选参数列表不能为空")
    return values


def calibrate_dtkf_on_validation_trajectory(
    dm, model, norm: Normalizer, aps: np.ndarray, wall: LargeRoomWallModel,
    fp_db: np.ndarray, rp_xy: np.ndarray, rss_bias: np.ndarray, csi_bias: np.ndarray,
    rss_cov: np.ndarray, csi_cov: np.ndarray, args, out: Path,
) -> tuple[float, float, float, float, float, pd.DataFrame]:
    """
    不改变 DTKF 公式，只在独立动态验证轨迹上选择数值参数：
      1) process acceleration std
      2) innovation gate covariance inflation
      3) innovation threshold percentile
      4) RSS measurement covariance scale
      5) CSI measurement covariance scale

    关键：
    - 正式测试轨迹完全不参与参数选择；
    - RSS 局部分支恢复源代码协议：DTKF 内部使用 K=1 最近指纹候选；
    - 主结果中的 RSS-KNN 仍然使用 K=5，不改变单源基线。
    """
    val_xy = make_validation_dynamic_trajectory(args.validation_dynamic_steps)
    verify_trajectory(val_xy, wall)
    val_xyz = np.column_stack([val_xy, np.full(len(val_xy), RX_Z)])
    val_csi = build_csi(
        dm, aps, val_xyz, wall, args.exponent, args.reflection_profile,
        args.n_antennas, args.n_subcarriers, args.bandwidth_mhz * 1e6
    )

    # DTKF RSS局部分支：K=1，恢复源代码设计
    rss_val_dyn = rss_predict(
        fp_db, rp_xy, power_db_from_csi(val_csi), args.seed + 51000,
        args.rss_noise_sigma_db, args.rss_n_measurements, 1
    ) - rss_bias

    csi_val_dyn = predict_dynamic_csi(
        model, norm, val_csi, args, seed=args.seed + 61000
    ) - csi_bias

    for pred in (rss_val_dyn, csi_val_dyn):
        pred[:, 0] = np.clip(pred[:, 0], 0, ROOM_W)
        pred[:, 1] = np.clip(pred[:, 1], 0, ROOM_H)

    accel_candidates = parse_float_candidates(args.process_accel_candidates)
    inflation_candidates = parse_float_candidates(args.gate_inflation_candidates)
    percentile_candidates = parse_float_candidates(args.innovation_percentile_candidates)
    rss_scale_candidates = parse_float_candidates(args.rss_cov_scale_candidates)
    csi_scale_candidates = parse_float_candidates(args.csi_cov_scale_candidates)

    innovation = np.linalg.norm(rss_val_dyn - csi_val_dyn, axis=1)
    csi_error = np.linalg.norm(csi_val_dyn - val_xy, axis=1)
    csi_mean = float(np.mean(csi_error))

    rows = []
    for percentile in percentile_candidates:
        if not 50 <= percentile <= 99:
            raise ValueError("innovation-percentile候选应在50~99")
        threshold = float(np.percentile(innovation, percentile))

        for rss_scale in rss_scale_candidates:
            if rss_scale <= 0:
                raise ValueError("rss-cov-scale候选必须>0")
            rr = rss_cov * float(rss_scale)

            for csi_scale in csi_scale_candidates:
                if csi_scale <= 0:
                    raise ValueError("csi-cov-scale候选必须>0")
                rc = csi_cov * float(csi_scale)

                for accel in accel_candidates:
                    if accel <= 0:
                        raise ValueError("process-accel候选必须>0")

                    for inflation in inflation_candidates:
                        if inflation < 1:
                            raise ValueError("gate-inflation候选必须>=1")

                        fused = run_dtkf(
                            rss_val_dyn, csi_val_dyn, rr, rc, threshold,
                            args.dt, accel, inflation
                        )
                        e_f = np.linalg.norm(fused - val_xy, axis=1)
                        rows.append({
                            "innovation_percentile": float(percentile),
                            "innovation_threshold_m": threshold,
                            "rss_cov_scale": float(rss_scale),
                            "csi_cov_scale": float(csi_scale),
                            "process_accel_std": float(accel),
                            "gate_inflation": float(inflation),
                            "fusion_mean_error_m": float(e_f.mean()),
                            "fusion_rmse_m": float(np.sqrt(np.mean(e_f ** 2))),
                            "csi_mean_error_m": csi_mean,
                            "improvement_vs_csi_pct": float(
                                (csi_mean - e_f.mean()) / max(csi_mean, 1e-12) * 100.0
                            ),
                        })

    table = pd.DataFrame(rows).sort_values(
        [
            "fusion_mean_error_m", "fusion_rmse_m",
            "process_accel_std", "gate_inflation",
            "rss_cov_scale", "csi_cov_scale",
        ]
    ).reset_index(drop=True)

    best = table.iloc[0]
    table.to_csv(out / "DTKF_独立动态验证参数搜索.csv", index=False, encoding="utf-8-sig")

    validation_frame = pd.DataFrame({
        "true_x": val_xy[:, 0], "true_y": val_xy[:, 1],
        "rss_local_k1_x": rss_val_dyn[:, 0], "rss_local_k1_y": rss_val_dyn[:, 1],
        "csi_x": csi_val_dyn[:, 0], "csi_y": csi_val_dyn[:, 1],
    })
    validation_frame.to_csv(out / "DTKF_独立动态验证轨迹预测.csv", index=False, encoding="utf-8-sig")

    return (
        float(best["process_accel_std"]),
        float(best["gate_inflation"]),
        float(best["innovation_threshold_m"]),
        float(best["rss_cov_scale"]),
        float(best["csi_cov_scale"]),
        table,
    )


# ============================== 绘图 ==============================
def draw_room(ax: plt.Axes, wall: LargeRoomWallModel, labels: bool = True) -> None:
    ax.plot([0,ROOM_W,ROOM_W,0,0], [0,0,ROOM_H,ROOM_H,0], "k-", lw=1.4)
    used = False
    for r in wall.solids:
        ax.add_patch(Rectangle((r.x0,r.y0), r.x1-r.x0, r.y1-r.y0,
            facecolor="gray", edgecolor="black", lw=2, alpha=0.75,
            label="Wall" if labels and not used else None))
        used = True
    door_kwargs = dict(facecolor="#dff3df", edgecolor="#7fbf7f", linestyle="--", lw=2, alpha=0.85)
    ax.add_patch(Rectangle((wall.vertical_x0, wall.vertical_door_y0),
        wall.vertical_x1-wall.vertical_x0, wall.vertical_door_y1-wall.vertical_door_y0,
        label="Door" if labels else None, **door_kwargs))
    ax.add_patch(Rectangle((wall.horizontal_door_x0, wall.horizontal_y0),
        wall.horizontal_door_x1-wall.horizontal_door_x0, wall.horizontal_y1-wall.horizontal_y0,
        **door_kwargs))


def plot_preview(wall, rp_xyz, aps, traj, path: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_room(ax, wall)
    ax.scatter(rp_xyz[:,0], rp_xyz[:,1], s=8, alpha=0.20, label="Reference Points")
    ax.plot(traj[:,0], traj[:,1], lw=2.1, label="Dynamic Trajectory")
    ax.scatter(aps[:,0], aps[:,1], marker="s", s=95, edgecolor="black", label="MIC-FD AP")
    for i, p in enumerate(aps, 1): ax.text(p[0]+0.12, p[1]+0.12, f"AP{i}", fontsize=10)
    ax.set(xlim=(-0.3,ROOM_W+0.3), ylim=(-0.3,ROOM_H+0.3), xlabel="X Position (m)", ylabel="Y Position (m)")
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5,1.12))
    fig.tight_layout(); fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


def plot_main_figure(wall, true_xy, rss_xy, csi_xy, fused_xy, dt: float, path: Path):
    er = np.linalg.norm(rss_xy - true_xy, axis=1)
    ec = np.linalg.norm(csi_xy - true_xy, axis=1)
    ef = np.linalg.norm(fused_xy - true_xy, axis=1)

    t = np.arange(len(true_xy)) * dt

    # 左右两个子图等宽
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(18, 7),
        gridspec_kw={
            "width_ratios": [1, 1],
            "wspace": 0.18
        }
    )

    # ============================================================
    # 左图：轨迹
    # ============================================================
    draw_room(ax1, wall, labels=True)

    ax1.plot(
        true_xy[:, 0], true_xy[:, 1],
        color="#4b4bff",
        lw=3.0,
        label="Ground Truth",
        zorder=4
    )

    ax1.plot(
        fused_xy[:, 0], fused_xy[:, 1],
        "r--",
        lw=2.5,
        label="DTKF Fusion",
        zorder=5
    )

    stride = max(1, len(true_xy) // 90)

    ax1.scatter(
        rss_xy[::stride, 0],
        rss_xy[::stride, 1],
        s=38,
        color="orange",
        alpha=0.78,
        label="RSS Measurements",
        zorder=3
    )

    ax1.scatter(
        csi_xy[::stride, 0],
        csi_xy[::stride, 1],
        s=34,
        color="green",
        alpha=0.70,
        label="CSI Measurements",
        zorder=3
    )

    worst = np.argsort(ef)[-min(12, len(ef)):]

    ax1.scatter(
        fused_xy[worst, 0],
        fused_xy[worst, 1],
        s=270,
        facecolor="#ff6b6b",
        edgecolor="red",
        alpha=0.22,
        zorder=2
    )

    # 固定左图范围
    xlim = (-0.4, ROOM_W + 0.4)
    ylim = (-0.4, ROOM_H + 0.4)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")

    # 左图保持真实空间比例：1 m = 1 m
    ax1.set_aspect("equal", adjustable="box")

    ax1.grid(alpha=0.25)

    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=11
    )

    # ============================================================
    # 右图：定位误差
    # ============================================================
    ax2.plot(
        t, er,
        color="#ffb340",
        lw=2.0,
        label="RSS Error"
    )

    ax2.plot(
        t, ec,
        color="#2c9b43",
        lw=2.0,
        label="CSI Error"
    )

    ax2.plot(
        t, ef,
        color="#ff3030",
        lw=2.1,
        label="DTKF Fusion Error"
    )

    mr = er.mean()
    mc = ec.mean()
    mf = ef.mean()

    ax2.axhline(
        mr,
        color="#ffb340",
        ls=":",
        lw=2,
        label=f"RSS Mean: {mr:.3f}m"
    )

    ax2.axhline(
        mc,
        color="#2c9b43",
        ls=":",
        lw=2,
        label=f"CSI Mean: {mc:.3f}m"
    )

    ax2.axhline(
        mf,
        color="#ff3030",
        ls=":",
        lw=2,
        label=f"Fusion Mean: {mf:.3f}m"
    )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Localization Error (m)")
    ax2.grid(alpha=0.25)

    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=11
    )

    # ============================================================
    # 关键修改：
    # 让右图画框的“高度/宽度”与左图完全一致
    # ============================================================

    left_box_ratio = (
        (ylim[1] - ylim[0]) /
        (xlim[1] - xlim[0])
    )

    ax2.set_box_aspect(left_box_ratio)

    # 左图也显式指定相同 box aspect，避免布局时出现轻微尺寸差异
    ax1.set_box_aspect(left_box_ratio)

    # ============================================================
    # 整体布局
    # ============================================================
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.12,
        top=0.82,
        wspace=0.18
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def metrics(name: str, true_xy: np.ndarray, pred_xy: np.ndarray) -> dict:
    e = np.linalg.norm(pred_xy - true_xy, axis=1)
    return dict(method=name, mean_error_m=e.mean(), median_error_m=np.median(e), rmse_m=np.sqrt(np.mean(e**2)),
                p90_m=np.percentile(e,90), p95_m=np.percentile(e,95), max_error_m=e.max(), std_error_m=e.std())


# ============================== 主流程 ==============================
def run(args):
    set_determinism(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    wall = LargeRoomWallModel(args.wall_loss_db)
    rp_xyz = make_reference_points(wall, args.grid_step); rp_xy = rp_xyz[:,:2]
    traj_xy = make_dynamic_trajectory(args.dynamic_steps); verify_trajectory(traj_xy, wall)

    print("="*92)
    print("复杂大场景 MIC-FD 动态融合实验（自然轨迹 + 独立动态验证）")
    print(f"房间: {ROOM_W}×{ROOM_H} m | 有效RP: {len(rp_xyz)} | 动态时刻: {len(traj_xy)}")
    print("纵墙: x=3.9~4.1, door y=3.0~4.5")
    print("横墙: y=3.9~4.1, x=4~12, door x=7.0~8.5")
    print("="*92)

    print("\n[1/7] 新场景重新执行 MIC-FD AP部署...")
    aps, trace = select_mic_fd_aps(rp_xy, wall)
    print(trace.to_string(index=False, float_format=lambda x:f"{x:.4f}"))
    print("\n新场景AP布局:")
    for i,p in enumerate(aps,1): print(f"  AP{i}: ({p[0]:.2f}, {p[1]:.2f})")
    trace.to_csv(out/"新场景_MIC-FD_AP筛选过程.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"AP":[f"AP{i}" for i in range(1,5)], "x":aps[:,0], "y":aps[:,1]}).to_csv(out/"新场景_MIC-FD_AP布局.csv", index=False, encoding="utf-8-sig")
    plot_preview(wall, rp_xyz, aps, traj_xy, out/"复杂场景与MIC-FD_AP布局.png")
    if args.preview_only:
        print(f"\npreview-only完成，结果目录: {out.resolve()}")
        return

    dm, tf = import_stack()
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
    tf.keras.utils.set_random_seed(args.seed)
    train_ids, val_ids = split_positions(len(rp_xyz), args.validation_fraction, args.seed)
    print(f"train={len(train_ids)}, validation={len(val_ids)}")

    print("\n[2/7] 构建参考点 DeepMIMO-style CSI...")
    ref_csi = build_csi(dm, aps, rp_xyz, wall, args.exponent, args.reflection_profile,
                        args.n_antennas, args.n_subcarriers, args.bandwidth_mhz*1e6)
    print("CSI shape:", ref_csi.shape)

    print("\n[3/7] 训练 CSI 3D CNN...")
    model, norm, csi_val_true, csi_val_pred, best_val = train_csi(tf, ref_csi, rp_xy, train_ids, val_ids, args)
    print(f"best CSI validation mean={best_val:.4f} m")

    print("\n[4/7] 验证集标定 RSS/CSI bias、covariance...")
    fp_db = power_db_from_csi(ref_csi)
    rss_val_true = rp_xy[val_ids]

    # K=5：只用于报告单源 RSS 基线
    rss_val_k5 = rss_predict(
        fp_db, rp_xy, fp_db[val_ids], args.seed+21000,
        args.rss_noise_sigma_db, args.rss_n_measurements, args.rss_k
    )
    rss_k5_bias, _ = estimate_bias_cov(rss_val_true, rss_val_k5)

    # K=1：恢复源代码 DTKF 最近指纹局部分支
    rss_val_local = rss_predict(
        fp_db, rp_xy, fp_db[val_ids], args.seed+21000,
        args.rss_noise_sigma_db, args.rss_n_measurements, 1
    )
    rss_local_bias, rss_cov = estimate_bias_cov(rss_val_true, rss_val_local)
    csi_bias, csi_cov = estimate_bias_cov(csi_val_true, csi_val_pred)
    print(f"RSS K=5 bias={rss_k5_bias}")
    print(f"DTKF RSS K=1 bias={rss_local_bias}, CSI bias={csi_bias}")

    print("\n[5/8] 独立动态验证轨迹选择 DTKF 数值参数（不使用正式测试轨迹）...")
    (
        selected_accel, selected_inflation, selected_threshold,
        selected_rss_scale, selected_csi_scale, dtkf_search
    ) = calibrate_dtkf_on_validation_trajectory(
        dm, model, norm, aps, wall, fp_db, rp_xy, rss_local_bias, csi_bias,
        rss_cov, csi_cov, args, out
    )
    print(
        f"DTKF validation-selected: process_accel_std={selected_accel:.4f}, "
        f"gate_inflation={selected_inflation:.2f}, "
        f"threshold={selected_threshold:.4f} m, "
        f"R_rss_scale={selected_rss_scale:.2f}, R_csi_scale={selected_csi_scale:.2f}"
    )
    print(
        dtkf_search.head(8).to_string(index=False, float_format=lambda x:f"{x:.4f}")
    )

    print("\n[6/8] 构建正式动态测试轨迹 CSI...")
    traj_xyz = np.column_stack([traj_xy, np.full(len(traj_xy), RX_Z)])
    traj_csi = build_csi(dm, aps, traj_xyz, wall, args.exponent, args.reflection_profile,
                         args.n_antennas, args.n_subcarriers, args.bandwidth_mhz*1e6)

    print("\n[7/8] 动态 RSS-KNN / CSI-CNN 定位...")
    trajectory_power_db = power_db_from_csi(traj_csi)

    # 单源 RSS 结果：继续保持 K=5
    rss_dyn = rss_predict(
        fp_db, rp_xy, trajectory_power_db, args.seed+31000,
        args.rss_noise_sigma_db, args.rss_n_measurements, args.rss_k
    ) - rss_k5_bias

    # DTKF 局部 RSS 分支：恢复源代码 K=1 最近指纹候选
    rss_dyn_local = rss_predict(
        fp_db, rp_xy, trajectory_power_db, args.seed+31000,
        args.rss_noise_sigma_db, args.rss_n_measurements, 1
    ) - rss_local_bias

    csi_dyn = predict_dynamic_csi(model, norm, traj_csi, args) - csi_bias
    for pred in (rss_dyn, rss_dyn_local, csi_dyn):
        pred[:,0] = np.clip(pred[:,0],0,ROOM_W)
        pred[:,1] = np.clip(pred[:,1],0,ROOM_H)

    print("\n[8/8] 动态公共先验去重 DTKF...")
    fused = run_dtkf(
        rss_dyn_local, csi_dyn,
        rss_cov * selected_rss_scale, csi_cov * selected_csi_scale,
        selected_threshold, args.dt, selected_accel, selected_inflation
    )

    summary = pd.DataFrame([
        metrics("RSS-KNN", traj_xy, rss_dyn),
        metrics("CSI-3D-CNN", traj_xy, csi_dyn),
        metrics("Dynamic-DTKF", traj_xy, fused),
    ])
    print("\n=== 动态定位结果 ===")
    print(summary.to_string(index=False, float_format=lambda x:f"{x:.4f}"))
    csi_mean_test = float(summary.loc[summary["method"] == "CSI-3D-CNN", "mean_error_m"].iloc[0])
    fusion_mean_test = float(summary.loc[summary["method"] == "Dynamic-DTKF", "mean_error_m"].iloc[0])
    improvement_test = (csi_mean_test - fusion_mean_test) / max(csi_mean_test, 1e-12) * 100.0
    print(f"DTKF vs CSI mean-error improvement = {improvement_test:.2f}%")
    summary.to_csv(out/"复杂场景_动态定位指标.csv", index=False, encoding="utf-8-sig")

    dyn = pd.DataFrame({
        "time_s":np.arange(len(traj_xy))*args.dt, "true_x":traj_xy[:,0], "true_y":traj_xy[:,1],
        "rss_x":rss_dyn[:,0], "rss_y":rss_dyn[:,1],
        "rss_dtkf_local_k1_x":rss_dyn_local[:,0], "rss_dtkf_local_k1_y":rss_dyn_local[:,1],
        "csi_x":csi_dyn[:,0], "csi_y":csi_dyn[:,1],
        "fusion_x":fused[:,0], "fusion_y":fused[:,1],
        "rss_error_m":np.linalg.norm(rss_dyn-traj_xy,axis=1),
        "csi_error_m":np.linalg.norm(csi_dyn-traj_xy,axis=1),
        "fusion_error_m":np.linalg.norm(fused-traj_xy,axis=1),
    })
    dyn.to_csv(out/"复杂场景_动态轨迹预测.csv", index=False, encoding="utf-8-sig")
    plot_main_figure(wall, traj_xy, rss_dyn, csi_dyn, fused, args.dt, out/"复杂场景_动态融合_轨迹与误差.png")
    np.savez_compressed(
        out/"DTKF标定参数.npz",
        rss_k5_bias=rss_k5_bias, rss_local_k1_bias=rss_local_bias, csi_bias=csi_bias,
        rss_covariance=rss_cov, csi_covariance=csi_cov,
        innovation_threshold=np.asarray([selected_threshold]),
        selected_process_accel_std=np.asarray([selected_accel]),
        selected_gate_inflation=np.asarray([selected_inflation]),
        selected_rss_cov_scale=np.asarray([selected_rss_scale]),
        selected_csi_cov_scale=np.asarray([selected_csi_scale]),
    )
    model.save_weights(out/"CSI_3D_CNN.weights.h5")
    np.savez_compressed(out/"CSI归一化参数.npz", mean=norm.mean, std=norm.std)
    config = {
        "room_m":[ROOM_W,ROOM_H], "grid_step_m":args.grid_step,
        "selected_APs":aps.tolist(), "D_th_m":D_TH, "R_dif_db":R_DIF_DB,
        "MIC_implementation":"abs(Pearson) surrogate, not true MIC",
        "DeepMIMO_used_for_AP_screening":False, "dynamic_steps":args.dynamic_steps,
        "dt_s":args.dt, "innovation_threshold_m":selected_threshold,
        "dtkf_rss_local_branch":"K=1 nearest fingerprint candidate",
        "dtkf_validation_selected_process_accel_std":selected_accel,
        "dtkf_validation_selected_gate_inflation":selected_inflation,
        "dtkf_validation_selected_rss_cov_scale":selected_rss_scale,
        "dtkf_validation_selected_csi_cov_scale":selected_csi_scale,
        "dtkf_test_improvement_vs_csi_pct":improvement_test,
        "rss":{"noise_sigma_db":args.rss_noise_sigma_db,"n_measurements":args.rss_n_measurements,"K":args.rss_k},
        "csi":{"n_antennas":args.n_antennas,"n_subcarriers":args.n_subcarriers,"bandwidth_mhz":args.bandwidth_mhz,
               "samples_per_position":args.samples_per_position,"amplitude_noise_db":args.amplitude_noise_db,
               "phase_noise_deg":args.phase_noise_deg,"awgn_ratio":args.awgn_ratio},
    }
    (out/"实验配置.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"\n主图: {(out/'复杂场景_动态融合_轨迹与误差.png').resolve()}")
    print(f"结果目录: {out.resolve()}")


def parser():
    p = argparse.ArgumentParser(description="12×8m复杂大场景 MIC-FD RSS/CSI 动态DTKF")
    p.add_argument("--output-dir", default="结果_复杂大场景_MIC-FD动态融合")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--grid-step", type=float, default=0.5)
    p.add_argument("--wall-loss-db", type=float, default=WALL_LOSS_DB)
    p.add_argument("--exponent", type=float, default=PATH_LOSS_EXPONENT)
    p.add_argument("--reflection-profile", choices=tuple(REFLECTION_PROFILES), default="weak")
    p.add_argument("--rss-noise-sigma-db", type=float, default=RSS_NOISE_SIGMA_DB)
    p.add_argument("--rss-n-measurements", type=int, default=RSS_N_MEASUREMENTS)
    p.add_argument("--rss-k", type=int, default=RSS_K)
    p.add_argument("--n-antennas", type=int, default=4)
    p.add_argument("--n-subcarriers", type=int, default=32)
    p.add_argument("--bandwidth-mhz", type=float, default=20.0)
    p.add_argument("--samples-per-position", type=int, default=20)
    p.add_argument("--amplitude-noise-db", type=float, default=0.8)
    p.add_argument("--phase-noise-deg", type=float, default=3.0)
    p.add_argument("--awgn-ratio", type=float, default=0.01)
    p.add_argument("--validation-fraction", type=float, default=0.18)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--eval-measurements", type=int, default=1)
    p.add_argument("--dynamic-csi-measurements", type=int, default=1)
    p.add_argument("--dynamic-steps", type=int, default=600)
    p.add_argument("--validation-dynamic-steps", type=int, default=220)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--process-accel-std", type=float, default=0.16, help="兼容旧版；正式运行由独立动态验证轨迹选择")
    p.add_argument("--gate-inflation", type=float, default=4.0, help="兼容旧版；正式运行由独立动态验证轨迹选择")
    p.add_argument("--process-accel-candidates", default="0.01,0.02,0.035,0.05,0.08,0.12,0.16")
    p.add_argument("--gate-inflation-candidates", default="2,3,4,6,8")
    p.add_argument("--innovation-percentile", type=float, default=75.0, help="兼容旧版，不用于正式选择")
    p.add_argument("--innovation-percentile-candidates", default="60,70,80,90")
    p.add_argument("--rss-cov-scale-candidates", default="0.7,1.0,1.4")
    p.add_argument("--csi-cov-scale-candidates", default="0.7,1.0,1.4")
    return p


def main():
    args = parser().parse_args()
    if args.grid_step <= 0: raise ValueError("grid-step必须>0")
    if args.dynamic_steps < 100: raise ValueError("dynamic-steps至少100")
    if args.validation_dynamic_steps < 80: raise ValueError("validation-dynamic-steps至少80")
    if not 0.05 <= args.validation_fraction <= 0.4: raise ValueError("validation-fraction应在0.05~0.4")
    if args.samples_per_position < 2 or args.restarts < 1 or args.epochs < 1: raise ValueError("CNN训练参数无效")
    if not 50 <= args.innovation_percentile <= 99: raise ValueError("innovation-percentile应在50~99")
    run(args)


# =====================================================================
# DTKF Component Ablation（基于上面的复杂场景完整代码）
# =====================================================================
# 注意：当前这份动态DTKF真实实现包含三个可以严格开/关的核心机制：
#   1) constant-velocity motion prediction
#   2) innovation-gated covariance inflation
#   3) common-prior information deduplication
# 因此本消融只拆这三个真实存在的组件。
# 当前代码没有 covariance intersection (CI)，所以这里不虚构 CI 消融。
#
# 消融策略：one-component-at-a-time
#   Full DTKF              : motion ✓, gating ✓, dedup ✓
#   w/o Motion Prediction  : motion ×, gating ✓, dedup ✓
#   w/o Innovation Gating  : motion ✓, gating ×, dedup ✓
#   w/o Prior Dedup        : motion ✓, gating ✓, dedup ×
#
# 公平性：
#   - 所有方案使用完全相同的 MIC-FD AP、RSS-K1局部分支、CSI-CNN输出、测试轨迹；
#   - 每个消融方案只在“独立动态验证轨迹”上选择自己仍然有效的数值参数；
#   - 正式600点测试轨迹不参与任何参数选择；
#   - CSI-CNN 仅作为参考基线，不算作 DTKF component variant。


def _naive_information_fusion(xr, pr, xc, pc):
    """不扣除公共先验：故意保留 common-prior double counting，作为 w/o dedup 消融。"""
    lr = np.linalg.pinv(pr)
    lc = np.linalg.pinv(pc)
    info = lr + lc
    eta = lr @ xr + lc @ xc
    p = make_spd(np.linalg.pinv(info), 1e-6)
    return p @ eta, p


def run_dtkf_variant(
    rss_xy: np.ndarray,
    csi_xy: np.ndarray,
    rss_cov: np.ndarray,
    csi_cov: np.ndarray,
    threshold: float,
    dt: float,
    process_std: float,
    inflation: float,
    *,
    use_motion_prediction: bool,
    use_innovation_gating: bool,
    use_prior_deduplication: bool,
) -> np.ndarray:
    """统一DTKF执行器；只通过三个布尔开关做组件消融。"""
    rss_xy = np.asarray(rss_xy, float)
    csi_xy = np.asarray(csi_xy, float)

    f_cv = np.asarray([
        [1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]
    ], float)
    q_cv = process_std ** 2 * np.asarray([
        [dt**4/4,0,dt**3/2,0], [0,dt**4/4,0,dt**3/2],
        [dt**3/2,0,dt**2,0], [0,dt**3/2,0,dt**2]
    ], float)

    # w/o motion prediction：保留时间滤波，但改为 random-walk prior，
    # 不利用恒速度运动模型预测下一位置。
    f_rw = np.eye(4, dtype=float)
    q_rw = process_std ** 2 * np.diag([
        max(dt**2, 1e-9), max(dt**2, 1e-9), 0.25, 0.25
    ])

    x = np.asarray([
        (rss_xy[0,0] + csi_xy[0,0]) / 2.0,
        (rss_xy[0,1] + csi_xy[0,1]) / 2.0,
        0.0, 0.0,
    ], float)
    p = np.diag([1.0, 1.0, 0.5, 0.5])
    h = np.asarray([[1.,0.,0.,0.], [0.,1.,0.,0.]])
    out = np.zeros((len(rss_xy), 2), float)

    for i in range(len(rss_xy)):
        if i == 0:
            xp, pp = x.copy(), p.copy()
        else:
            if use_motion_prediction:
                xp = f_cv @ x
                pp = make_spd(f_cv @ p @ f_cv.T + q_cv, 1e-6)
            else:
                xp = f_rw @ x
                pp = make_spd(f_rw @ p @ f_rw.T + q_rw, 1e-6)

        rr = np.asarray(rss_cov, float).copy()
        rc = np.asarray(csi_cov, float).copy()

        if use_innovation_gating:
            if np.linalg.norm(rss_xy[i] - csi_xy[i]) > threshold:
                prior_xy = h @ xp
                rss_innovation = np.linalg.norm(rss_xy[i] - prior_xy)
                csi_innovation = np.linalg.norm(csi_xy[i] - prior_xy)
                if rss_innovation >= csi_innovation:
                    rr *= inflation
                else:
                    rc *= inflation

        xr, pr = kf_update(xp, pp, rss_xy[i], rr)
        xc, pc = kf_update(xp, pp, csi_xy[i], rc)

        if use_prior_deduplication:
            x, p = deduplicate_common_prior(xp, pp, xr, pr, xc, pc)
        else:
            x, p = _naive_information_fusion(xr, pr, xc, pc)

        x[0] = np.clip(x[0], 0.0, ROOM_W)
        x[1] = np.clip(x[1], 0.0, ROOM_H)
        out[i] = x[:2]

    return out


ABLATION_VARIANTS = {
    'Full DTKF': dict(
        use_motion_prediction=True,
        use_innovation_gating=True,
        use_prior_deduplication=True,
    ),
    'w/o Motion Prediction': dict(
        use_motion_prediction=False,
        use_innovation_gating=True,
        use_prior_deduplication=True,
    ),
    'w/o Innovation Gating': dict(
        use_motion_prediction=True,
        use_innovation_gating=False,
        use_prior_deduplication=True,
    ),
    'w/o Prior Deduplication': dict(
        use_motion_prediction=True,
        use_innovation_gating=True,
        use_prior_deduplication=False,
    ),
}


def _required_base_outputs(output_dir: Path) -> list[Path]:
    return [
        output_dir / 'DTKF_独立动态验证轨迹预测.csv',
        output_dir / '复杂场景_动态轨迹预测.csv',
        output_dir / 'DTKF标定参数.npz',
    ]


def _load_ablation_inputs(output_dir: Path):
    required = _required_base_outputs(output_dir)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            '缺少基础实验输出，请先不带 --reuse-existing 完整运行一次。缺少：\n' + '\n'.join(missing)
        )

    val = pd.read_csv(output_dir / 'DTKF_独立动态验证轨迹预测.csv')
    test = pd.read_csv(output_dir / '复杂场景_动态轨迹预测.csv')
    calib = np.load(output_dir / 'DTKF标定参数.npz')

    val_true = val[['true_x','true_y']].to_numpy(float)
    val_rss = val[['rss_local_k1_x','rss_local_k1_y']].to_numpy(float)
    val_csi = val[['csi_x','csi_y']].to_numpy(float)

    test_true = test[['true_x','true_y']].to_numpy(float)
    test_rss = test[['rss_dtkf_local_k1_x','rss_dtkf_local_k1_y']].to_numpy(float)
    test_csi = test[['csi_x','csi_y']].to_numpy(float)

    rss_cov = np.asarray(calib['rss_covariance'], float)
    csi_cov = np.asarray(calib['csi_covariance'], float)

    return val_true, val_rss, val_csi, test_true, test_rss, test_csi, rss_cov, csi_cov


def _variant_search(
    variant_name: str,
    switches: dict,
    val_true: np.ndarray,
    val_rss: np.ndarray,
    val_csi: np.ndarray,
    rss_cov: np.ndarray,
    csi_cov: np.ndarray,
    args,
) -> tuple[dict, pd.DataFrame]:
    accel_candidates = parse_float_candidates(args.process_accel_candidates)
    inflation_candidates = parse_float_candidates(args.gate_inflation_candidates)
    percentile_candidates = parse_float_candidates(args.innovation_percentile_candidates)
    rss_scale_candidates = parse_float_candidates(args.rss_cov_scale_candidates)
    csi_scale_candidates = parse_float_candidates(args.csi_cov_scale_candidates)

    # 关闭 gating 时 threshold / inflation 不再是有效自由度，避免重复搜索。
    if not switches['use_innovation_gating']:
        percentile_candidates = [float('nan')]
        inflation_candidates = [1.0]

    disagreement = np.linalg.norm(val_rss - val_csi, axis=1)
    csi_mean = float(np.mean(np.linalg.norm(val_csi - val_true, axis=1)))
    rows = []

    for percentile in percentile_candidates:
        threshold = (
            float(np.percentile(disagreement, percentile))
            if switches['use_innovation_gating'] else float('inf')
        )
        for rss_scale in rss_scale_candidates:
            rr = rss_cov * float(rss_scale)
            for csi_scale in csi_scale_candidates:
                rc = csi_cov * float(csi_scale)
                for process_std in accel_candidates:
                    for inflation in inflation_candidates:
                        pred = run_dtkf_variant(
                            val_rss, val_csi, rr, rc, threshold,
                            args.dt, float(process_std), float(inflation),
                            **switches,
                        )
                        e = np.linalg.norm(pred - val_true, axis=1)
                        rows.append({
                            'variant': variant_name,
                            'innovation_percentile': percentile,
                            'innovation_threshold_m': threshold,
                            'rss_cov_scale': float(rss_scale),
                            'csi_cov_scale': float(csi_scale),
                            'process_std': float(process_std),
                            'gate_inflation': float(inflation),
                            'validation_mean_error_m': float(e.mean()),
                            'validation_rmse_m': float(np.sqrt(np.mean(e**2))),
                            'validation_p95_m': float(np.percentile(e, 95)),
                            'csi_validation_mean_error_m': csi_mean,
                            'improvement_vs_csi_pct': float((csi_mean - e.mean()) / max(csi_mean, 1e-12) * 100.0),
                        })

    table = pd.DataFrame(rows).sort_values(
        ['validation_mean_error_m','validation_rmse_m','validation_p95_m','process_std','gate_inflation'],
        na_position='last'
    ).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    return best, table


def _plot_ablation_bars(summary: pd.DataFrame, path: Path) -> None:
    plot_df = summary[summary['method'] != 'CSI-3D-CNN'].copy()
    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.bar(x - width/2, plot_df['mean_error_m'], width, label='Mean Error')
    ax.bar(x + width/2, plot_df['rmse_m'], width, label='RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['method'], rotation=12, ha='right')
    ax.set_ylabel('Localization Error (m)')
    ax.set_title('DTKF Component Ablation in the Complex Scenario')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches='tight')
    plt.close(fig)


def _plot_ablation_cdf(error_map: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for name, error in error_map.items():
        e = np.sort(np.asarray(error, float))
        cdf = np.arange(1, len(e)+1) / len(e)
        ax.plot(e, cdf, lw=1.8, label=name)
    ax.set_xlabel('Localization Error (m)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF of DTKF Component Ablation')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches='tight')
    plt.close(fig)


def _plot_ablation_timeseries(error_map: dict[str, np.ndarray], dt: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    n = len(next(iter(error_map.values())))
    t = np.arange(n) * float(dt)
    for name, error in error_map.items():
        ax.plot(t, error, lw=1.35, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Localization Error (m)')
    ax.set_title('DTKF Component Ablation: Error Time Series')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches='tight')
    plt.close(fig)


def run_component_ablation_from_saved_outputs(args) -> pd.DataFrame:
    output_dir = Path(args.output_dir)
    ablation_dir = output_dir / 'DTKF组件消融'
    ablation_dir.mkdir(parents=True, exist_ok=True)

    (
        val_true, val_rss, val_csi,
        test_true, test_rss, test_csi,
        rss_cov, csi_cov,
    ) = _load_ablation_inputs(output_dir)

    print('\n' + '='*92)
    print('DTKF COMPONENT ABLATION')
    print('Same MIC-FD deployment / same RSS-K1 local branch / same CSI-CNN / same test trajectory')
    print('Each variant is tuned only on the independent dynamic validation trajectory.')
    print('='*92)

    component_table = pd.DataFrame([
        {
            'variant': name,
            'motion_prediction': int(cfg['use_motion_prediction']),
            'innovation_gating': int(cfg['use_innovation_gating']),
            'prior_deduplication': int(cfg['use_prior_deduplication']),
        }
        for name, cfg in ABLATION_VARIANTS.items()
    ])
    component_table.to_csv(ablation_dir / 'DTKF组件定义.csv', index=False, encoding='utf-8-sig')

    all_search = []
    selected_rows = []
    predictions = {}
    error_map = {
        'CSI-3D-CNN': np.linalg.norm(test_csi - test_true, axis=1)
    }

    for name, switches in ABLATION_VARIANTS.items():
        print(f'\n[Validation search] {name}')
        best, search = _variant_search(
            name, switches,
            val_true, val_rss, val_csi,
            rss_cov, csi_cov, args,
        )
        all_search.append(search)
        selected_rows.append(best)

        print(
            f"  selected: mean={best['validation_mean_error_m']:.4f} m, "
            f"process={best['process_std']:.4f}, "
            f"threshold={best['innovation_threshold_m']:.4f}, "
            f"inflation={best['gate_inflation']:.2f}, "
            f"Rrss={best['rss_cov_scale']:.2f}, Rcsi={best['csi_cov_scale']:.2f}"
        )

        pred = run_dtkf_variant(
            test_rss,
            test_csi,
            rss_cov * float(best['rss_cov_scale']),
            csi_cov * float(best['csi_cov_scale']),
            float(best['innovation_threshold_m']),
            args.dt,
            float(best['process_std']),
            float(best['gate_inflation']),
            **switches,
        )
        predictions[name] = pred
        error_map[name] = np.linalg.norm(pred - test_true, axis=1)

    search_table = pd.concat(all_search, ignore_index=True)
    search_table.to_csv(
        ablation_dir / 'DTKF组件消融_独立验证参数搜索.csv',
        index=False,
        encoding='utf-8-sig',
    )
    selected_table = pd.DataFrame(selected_rows)
    selected_table.to_csv(
        ablation_dir / 'DTKF组件消融_各方案验证最优参数.csv',
        index=False,
        encoding='utf-8-sig',
    )

    rows = [metrics('CSI-3D-CNN', test_true, test_csi)]
    for name in ABLATION_VARIANTS:
        rows.append(metrics(name, test_true, predictions[name]))
    summary = pd.DataFrame(rows)

    full_mean = float(summary.loc[summary['method'] == 'Full DTKF', 'mean_error_m'].iloc[0])
    csi_mean = float(summary.loc[summary['method'] == 'CSI-3D-CNN', 'mean_error_m'].iloc[0])
    full_error = error_map['Full DTKF']

    summary['improvement_vs_csi_pct'] = (csi_mean - summary['mean_error_m']) / max(csi_mean, 1e-12) * 100.0
    summary['degradation_vs_full_pct'] = (summary['mean_error_m'] - full_mean) / max(full_mean, 1e-12) * 100.0
    summary['full_win_rate_pct'] = np.nan
    for idx, row in summary.iterrows():
        name = row['method']
        if name in error_map and name != 'Full DTKF':
            summary.loc[idx, 'full_win_rate_pct'] = float(np.mean(full_error < error_map[name]) * 100.0)

    summary.to_csv(
        ablation_dir / 'DTKF组件消融_正式测试结果.csv',
        index=False,
        encoding='utf-8-sig',
    )

    pred_frame = pd.DataFrame({
        'time_s': np.arange(len(test_true)) * args.dt,
        'true_x': test_true[:,0], 'true_y': test_true[:,1],
        'csi_x': test_csi[:,0], 'csi_y': test_csi[:,1],
        'csi_error_m': error_map['CSI-3D-CNN'],
    })
    for name, pred in predictions.items():
        safe = (
            name.lower().replace(' ', '_').replace('/', 'without_')
            .replace('-', '_').replace('(', '').replace(')', '')
        )
        pred_frame[f'{safe}_x'] = pred[:,0]
        pred_frame[f'{safe}_y'] = pred[:,1]
        pred_frame[f'{safe}_error_m'] = error_map[name]
    pred_frame.to_csv(
        ablation_dir / 'DTKF组件消融_正式测试逐时刻预测.csv',
        index=False,
        encoding='utf-8-sig',
    )

    _plot_ablation_bars(summary, ablation_dir / 'DTKF组件消融_平均误差与RMSE.png')
    _plot_ablation_cdf(error_map, ablation_dir / 'DTKF组件消融_CDF.png')
    _plot_ablation_timeseries(error_map, args.dt, ablation_dir / 'DTKF组件消融_误差时间序列.png')

    print('\n=== DTKF Component Ablation: Formal Test ===')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n解释规则：')
    print('  degradation_vs_full_pct > 0 ：去掉该组件后比 Full DTKF 更差。')
    print('  Full DTKF 最好时，说明 motion prediction / gating / prior dedup 的组合在该复杂场景有效。')
    print(f'\n消融结果目录: {ablation_dir.resolve()}')
    return summary


def ablation_parser():
    p = parser()
    p.description = '12×8m复杂场景 DTKF Component Ablation'
    p.add_argument(
        '--reuse-existing',
        action='store_true',
        help=(
            '复用 output-dir 中已经完成的红线轨迹基础实验结果，'
            '跳过DeepMIMO和CNN训练；推荐你已经跑过基础实验时使用。'
        ),
    )
    return p


def main():
    args = ablation_parser().parse_args()
    if args.grid_step <= 0: raise ValueError('grid-step必须>0')
    if args.dynamic_steps < 100: raise ValueError('dynamic-steps至少100')
    if args.validation_dynamic_steps < 80: raise ValueError('validation-dynamic-steps至少80')
    if not 0.05 <= args.validation_fraction <= 0.4: raise ValueError('validation-fraction应在0.05~0.4')
    if args.samples_per_position < 2 or args.restarts < 1 or args.epochs < 1: raise ValueError('CNN训练参数无效')

    if not args.reuse_existing:
        # 先完整运行与当前主实验完全一致的基础流程；
        # 训练一次CNN并生成 validation/test RSS/CSI observations。
        run(args)
        if args.preview_only:
            return
    else:
        missing = [p for p in _required_base_outputs(Path(args.output_dir)) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                '--reuse-existing 已开启，但缺少基础结果：\n' + '\n'.join(str(p) for p in missing)
            )

    run_component_ablation_from_saved_outputs(args)


# =====================================================================
# 合并版：DTKF Component Ablation + Dynamic RSSI/CSI Localization 主图
# 作用：
#   1) 保留原来的复杂场景 DTKF 组件消融；
#   2) 在同一次运行里，额外保证输出与 Dynamic RSSI/CSI Localization
#      完全同风格的主图（轨迹 + 误差）；
#   3) 若使用 --reuse-existing，则直接从已保存CSV重建主图，不重新训练CNN。
# =====================================================================

import shutil


def _load_dynamic_main_outputs(output_dir: Path):
    csv_path = output_dir / "复杂场景_动态轨迹预测.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"缺少动态定位输出文件：{csv_path}\n"
            "请先完成一次复杂场景主实验，或不要使用 --reuse-existing。"
        )
    df = pd.read_csv(csv_path)
    required = [
        "true_x", "true_y",
        "rss_x", "rss_y",
        "csi_x", "csi_y",
        "fusion_x", "fusion_y",
    ]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"动态预测CSV缺少列: {c}")

    dt = 1.0
    if "time_s" in df.columns and len(df) >= 2:
        t = df["time_s"].to_numpy(float)
        d = np.diff(t)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d):
            dt = float(np.median(d))

    return {
        "true_xy": df[["true_x", "true_y"]].to_numpy(float),
        "rss_xy": df[["rss_x", "rss_y"]].to_numpy(float),
        "csi_xy": df[["csi_x", "csi_y"]].to_numpy(float),
        "fusion_xy": df[["fusion_x", "fusion_y"]].to_numpy(float),
        "dt": dt,
    }


def ensure_dynamic_main_figure(args) -> Path:
    """
    保证“Dynamic RSSI/CSI Localization”同款主图一定存在，
    并复制一份到 DTKF组件消融 子目录，方便论文直接使用。
    """
    output_dir = Path(args.output_dir)
    ablation_dir = output_dir / "DTKF组件消融"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    main_fig = output_dir / "复杂场景_动态融合_轨迹与误差.png"
    ablation_fig = ablation_dir / "复杂场景_动态融合_轨迹与误差.png"
    ablation_fig_alt = ablation_dir / "DTKF组件消融_动态RSSI_CSI定位主图.png"

    # 如果主图不存在，则从已保存CSV重建
    if not main_fig.exists():
        data = _load_dynamic_main_outputs(output_dir)
        wall = LargeRoomWallModel(float(getattr(args, "wall_loss_db", WALL_LOSS_DB)))
        plot_main_figure(
            wall,
            data["true_xy"],
            data["rss_xy"],
            data["csi_xy"],
            data["fusion_xy"],
            data["dt"],
            main_fig,
        )

    # 复制到 DTKF 组件消融目录
    shutil.copy2(main_fig, ablation_fig)
    shutil.copy2(main_fig, ablation_fig_alt)
    return main_fig


def merged_parser():
    p = parser()
    p.description = "复杂场景：DTKF Component Ablation + Dynamic RSSI/CSI Localization 合并版"
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "复用 output-dir 中已有的复杂场景主实验结果。"
            "此模式下不会重新训练CNN，会直接重建/复制Dynamic主图并运行DTKF组件消融。"
        ),
    )
    return p


def main():
    args = merged_parser().parse_args()
    if args.grid_step <= 0:
        raise ValueError("grid-step必须>0")
    if args.dynamic_steps < 100:
        raise ValueError("dynamic-steps至少100")
    if args.validation_dynamic_steps < 80:
        raise ValueError("validation-dynamic-steps至少80")
    if not 0.05 <= args.validation_fraction <= 0.4:
        raise ValueError("validation-fraction应在0.05~0.4")
    if args.samples_per_position < 2 or args.restarts < 1 or args.epochs < 1:
        raise ValueError("CNN训练参数无效")

    output_dir = Path(args.output_dir)

    if not args.reuse_existing:
        # 先完整运行 Dynamic RSSI/CSI Localization 主实验
        run(args)
        if args.preview_only:
            return
    else:
        missing = [p for p in _required_base_outputs(output_dir) if not p.exists()]
        dynamic_csv = output_dir / "复杂场景_动态轨迹预测.csv"
        if missing or (not dynamic_csv.exists()):
            msg = []
            if missing:
                msg.append("缺少基础DTKF结果：")
                msg.extend(str(p) for p in missing)
            if not dynamic_csv.exists():
                msg.append(f"缺少动态主图重建所需CSV：{dynamic_csv}")
            raise FileNotFoundError(
                "--reuse-existing 已开启，但结果不完整：\n" + "\n".join(msg)
            )

    # 无论是否复用，都保证生成/复制 Dynamic RSSI/CSI Localization 同款主图
    fig_path = ensure_dynamic_main_figure(args)
    print(f"\nDynamic RSSI/CSI Localization 主图: {fig_path.resolve()}")

    # 再执行 DTKF 组件消融
    run_component_ablation_from_saved_outputs(args)

    print("\n说明：")
    print("  1) 本脚本已将 Dynamic RSSI/CSI Localization 与 DTKF Component Ablation 合并；")
    print("  2) 同款主图已额外复制到：结果目录/DTKF组件消融/")
    print("  3) 如果你只想基于已锁定canonical run做分析，请使用 --reuse-existing。")


if __name__ == "__main__":
    main()
