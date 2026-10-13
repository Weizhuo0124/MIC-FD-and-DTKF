# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOM_W = 12.0
ROOM_H = 8.0


# ======================================================================
# 通用
# ======================================================================

def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def clip_positions(xy: np.ndarray) -> np.ndarray:
    out = np.asarray(xy, float).copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, ROOM_W)
    out[:, 1] = np.clip(out[:, 1], 0.0, ROOM_H)
    return out


def make_spd(m: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    m = np.asarray(m, float)
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, floor)
    return v @ np.diag(w) @ v.T


def covariance_from_residuals(
    residuals: np.ndarray,
    shrinkage: float = 0.50,
    floor: float = 1e-4,
) -> np.ndarray:
    residuals = np.asarray(residuals, float)
    if len(residuals) <= 1:
        cov = np.eye(residuals.shape[1], dtype=float)
    else:
        cov = np.cov(residuals, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    diag = np.diag(np.diag(cov))
    cov = (1.0 - float(shrinkage)) * cov + float(shrinkage) * diag
    return make_spd(cov, floor)


def errors(true_xy: np.ndarray, pred_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(pred_xy) - np.asarray(true_xy), axis=1)


def metric_row(method: str, true_xy: np.ndarray, pred_xy: np.ndarray) -> dict:
    e = errors(true_xy, pred_xy)
    return {
        "method": method,
        "mean_error_m": float(np.mean(e)),
        "median_error_m": float(np.median(e)),
        "rmse_m": float(np.sqrt(np.mean(e ** 2))),
        "p90_m": float(np.percentile(e, 90)),
        "p95_m": float(np.percentile(e, 95)),
        "max_error_m": float(np.max(e)),
        "std_error_m": float(np.std(e)),
    }


def import_module_from_path(path: Path, name: str):
    """
    Windows/Python 3.12-safe dynamic import.

    dataclasses 在装饰类时会通过 sys.modules[cls.__module__]
    查找当前模块命名空间，因此必须在 exec_module() 之前
    先把 module 注册到 sys.modules。
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {path}")

    module = importlib.util.module_from_spec(spec)

    # 关键修复：先注册模块，再执行模块代码。
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return module


def find_complex_script(requested: str) -> Path:
    p = Path(requested)
    if p.exists():
        return p.resolve()

    candidates = [
        Path.cwd() / "复杂大场景_MIC-FD动态融合实验_红线轨迹_验证优化版.py",
        Path.cwd() / "复杂大场景_MIC-FD动态融合实验_自然轨迹验证版.py",
        Path.cwd() / "复杂大场景_MIC-FD动态融合实验.py",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "找不到复杂场景主脚本。请用 --complex-script 指定它的完整路径。"
    )


# ======================================================================
# canonical run
# ======================================================================

def load_canonical(output_dir: Path) -> dict:
    req = [
        output_dir / "DTKF_独立动态验证轨迹预测.csv",
        output_dir / "复杂场景_动态轨迹预测.csv",
        output_dir / "DTKF标定参数.npz",
        output_dir / "实验配置.json",
        output_dir / "新场景_MIC-FD_AP布局.csv",
        output_dir / "CSI_3D_CNN.weights.h5",
        output_dir / "CSI归一化参数.npz",
    ]
    missing = [p for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 canonical run 文件：\n" + "\n".join(str(p) for p in missing)
        )

    val = pd.read_csv(output_dir / "DTKF_独立动态验证轨迹预测.csv")
    test = pd.read_csv(output_dir / "复杂场景_动态轨迹预测.csv")
    calib = np.load(output_dir / "DTKF标定参数.npz")
    config = json.loads((output_dir / "实验配置.json").read_text(encoding="utf-8"))
    ap_table = pd.read_csv(output_dir / "新场景_MIC-FD_AP布局.csv")
    aps = ap_table[["x", "y"]].to_numpy(float)

    return {
        "val": val,
        "test": test,
        "calib": calib,
        "config": config,
        "aps": aps,
    }


# ======================================================================
# source-protocol RSS
# ======================================================================

def mean_noisy_rss(
    true_power_db: np.ndarray,
    seed: int,
    sigma_db: float,
    n_measurements: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    samples = true_power_db[:, None, :] + rng.normal(
        0.0,
        float(sigma_db),
        size=(len(true_power_db), int(n_measurements), true_power_db.shape[1]),
    )
    return np.mean(samples, axis=1)


def rss_position_and_confidence(
    fp_db: np.ndarray,
    rp_xy: np.ndarray,
    mean_measurements: np.ndarray,
    k: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    k_use = min(int(k), len(fp_db))

    model_x = KNeighborsRegressor(n_neighbors=k_use, weights="distance")
    model_y = KNeighborsRegressor(n_neighbors=k_use, weights="distance")
    model_x.fit(fp_db, rp_xy[:, 0])
    model_y.fit(fp_db, rp_xy[:, 1])

    pred = np.column_stack([
        model_x.predict(mean_measurements),
        model_y.predict(mean_measurements),
    ])
    pred = clip_positions(pred)

    # confidence always uses at least 5 neighbours when possible, matching source intent
    n_conf = min(max(5, k_use), len(fp_db))
    nn = NearestNeighbors(n_neighbors=n_conf)
    nn.fit(fp_db)
    dist, idx = nn.kneighbors(mean_measurements)

    rows = []
    for i, (drow, irow) in enumerate(zip(dist, idx)):
        w = 1.0 / np.maximum(drow, 1e-9)
        w /= np.sum(w)
        xy = rp_xy[irow]
        center = np.sum(w[:, None] * xy, axis=0)
        spread = math.sqrt(float(np.sum(w * np.sum((xy - center) ** 2, axis=1))))
        rows.append({
            "position_id": i,
            "rss_neighbor_spread": spread,
            "rss_nearest_distance": float(drow[0]),
            "rss_distance_ratio": float(drow[0] / max(drow[-1], 1e-9)),
        })

    return pred, pd.DataFrame(rows)


# ======================================================================
# continuous RSS field — copied in structure from source protocol
# ======================================================================

class RSSFingerprintField:
    def __init__(
        self,
        positions_xy: np.ndarray,
        fingerprints_db: np.ndarray,
        n_neighbors: int = 12,
        length_scale_m: float = 0.65,
    ):
        self.positions = np.asarray(positions_xy, float)
        self.fingerprints = np.asarray(fingerprints_db, float)
        self.n_neighbors = min(max(3, int(n_neighbors)), len(self.positions))
        self.length_scale_m = max(float(length_scale_m), 1e-3)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn.fit(self.positions)

    def predict(self, position_xy: np.ndarray) -> np.ndarray:
        x = np.asarray(position_xy, float).reshape(1, 2)
        d, idx = self.nn.kneighbors(x)
        d = d[0]
        idx = idx[0]
        if d[0] < 1e-10:
            return self.fingerprints[idx[0]].copy()
        lw = -0.5 * (d / self.length_scale_m) ** 2
        lw -= np.max(lw)
        w = np.exp(lw)
        w /= np.maximum(np.sum(w), 1e-12)
        return np.sum(w[:, None] * self.fingerprints[idx], axis=0)

    def predict_batch(self, positions_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(positions_xy, float).reshape(-1, 2)
        d, idx = self.nn.kneighbors(pts)
        lw = -0.5 * (d / self.length_scale_m) ** 2
        lw -= np.max(lw, axis=1, keepdims=True)
        w = np.exp(lw)
        w /= np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
        out = np.sum(w[:, :, None] * self.fingerprints[idx], axis=1)
        exact = d[:, 0] < 1e-10
        if np.any(exact):
            out[exact] = self.fingerprints[idx[exact, 0]]
        return out

    def jacobian(self, position_xy: np.ndarray, step_m: float = 1e-3) -> np.ndarray:
        x = np.asarray(position_xy, float)
        jac = np.zeros((self.fingerprints.shape[1], 2), float)
        for axis in range(2):
            plus = x.copy()
            minus = x.copy()
            plus[axis] += step_m
            minus[axis] -= step_m
            plus[0] = np.clip(plus[0], 0, ROOM_W)
            plus[1] = np.clip(plus[1], 0, ROOM_H)
            minus[0] = np.clip(minus[0], 0, ROOM_W)
            minus[1] = np.clip(minus[1], 0, ROOM_H)
            denom = max(float(plus[axis] - minus[axis]), 1e-9)
            jac[:, axis] = (self.predict(plus) - self.predict(minus)) / denom
        return jac


# ======================================================================
# source-style nonlinear baselines
# ======================================================================

def ekf_fuse_one(csi_position, rss_measurement, field, csi_cov, rss_cov):
    x_prior = np.asarray(csi_position, float)
    p_prior = make_spd(csi_cov)
    h = field.predict(x_prior)
    H = field.jacobian(x_prior)
    innovation = np.asarray(rss_measurement, float) - h
    S = make_spd(H @ p_prior @ H.T + rss_cov)
    K = np.linalg.solve(S, H @ p_prior).T
    x = x_prior + K @ innovation
    I = np.eye(2)
    ikh = I - K @ H
    p = ikh @ p_prior @ ikh.T + K @ rss_cov @ K.T
    return clip_positions(x[None])[0], make_spd(p)


def ukf_sigma_points(mean, cov, alpha=0.35, beta=2.0, kappa=0.0):
    mean = np.asarray(mean, float)
    cov = make_spd(cov)
    n = len(mean)
    lam = alpha ** 2 * (n + kappa) - n
    scale = n + lam
    root = np.linalg.cholesky(scale * cov)

    pts = [mean]
    for j in range(n):
        pts.append(mean + root[:, j])
        pts.append(mean - root[:, j])

    wm = np.full(2 * n + 1, 1.0 / (2 * scale))
    wc = wm.copy()
    wm[0] = lam / scale
    wc[0] = lam / scale + (1 - alpha ** 2 + beta)
    return np.asarray(pts), wm, wc


def ukf_fuse_one(csi_position, rss_measurement, field, csi_cov, rss_cov):
    mean = np.asarray(csi_position, float)
    cov = make_spd(csi_cov)
    pts, wm, wc = ukf_sigma_points(mean, cov)
    pts[:, 0] = np.clip(pts[:, 0], 0, ROOM_W)
    pts[:, 1] = np.clip(pts[:, 1], 0, ROOM_H)

    z_pts = np.asarray([field.predict(p) for p in pts])
    z_mean = np.sum(wm[:, None] * z_pts, axis=0)

    S = np.asarray(rss_cov, float).copy()
    C = np.zeros((2, z_pts.shape[1]))
    for p, z, w in zip(pts, z_pts, wc):
        dx = p - mean
        dz = z - z_mean
        S += w * np.outer(dz, dz)
        C += w * np.outer(dx, dz)

    S = make_spd(S)
    K = np.linalg.solve(S, C.T).T
    x = mean + K @ (np.asarray(rss_measurement) - z_mean)
    p = cov - K @ S @ K.T
    return clip_positions(x[None])[0], make_spd(p)


def pf_fuse_one(
    csi_position,
    rss_measurement,
    field,
    csi_cov,
    rss_cov,
    n_particles,
    seed,
):
    rng = np.random.default_rng(int(seed))
    particles = rng.multivariate_normal(
        np.asarray(csi_position, float),
        make_spd(csi_cov),
        size=int(n_particles),
    )
    particles[:, 0] = np.clip(particles[:, 0], 0, ROOM_W)
    particles[:, 1] = np.clip(particles[:, 1], 0, ROOM_H)

    inv_r = np.linalg.inv(make_spd(rss_cov))
    predicted_rss = field.predict_batch(particles)
    res = np.asarray(rss_measurement, float)[None] - predicted_rss
    logw = -0.5 * np.einsum("ni,ij,nj->n", res, inv_r, res)
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= np.maximum(np.sum(w), 1e-30)

    est = np.sum(w[:, None] * particles, axis=0)
    centered = particles - est
    cov = np.einsum("n,ni,nj->ij", w, centered, centered)
    return clip_positions(est[None])[0], make_spd(cov)


def fgo_fuse_one(csi_position, rss_measurement, field, csi_cov, rss_cov):
    csi_cov = make_spd(csi_cov)
    rss_cov = make_spd(rss_cov)
    csi_root = np.linalg.cholesky(csi_cov)
    rss_root = np.linalg.cholesky(rss_cov)

    def residual(position):
        csi_factor = np.linalg.solve(csi_root, position - csi_position)
        rss_factor = np.linalg.solve(
            rss_root,
            rss_measurement - field.predict(position),
        )
        return np.concatenate([csi_factor, rss_factor])

    result = least_squares(
        residual,
        x0=clip_positions(np.asarray(csi_position)[None])[0],
        bounds=([0.0, 0.0], [ROOM_W, ROOM_H]),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=80,
    )

    est = clip_positions(result.x[None])[0]
    J = np.asarray(result.jac, float)
    information = J.T @ J + np.eye(2) * 1e-6
    cov = make_spd(np.linalg.inv(information))
    return est, cov


def run_nonlinear_method(
    method,
    csi_positions,
    rss_measurements,
    field,
    csi_covariance,
    rss_covariance,
    rss_scale,
    particle_count,
    seed,
):
    predictions = []
    covariances = []
    scaled_rss_cov = make_spd(rss_covariance * float(rss_scale))

    for i, (csi_pos, rss_meas) in enumerate(zip(csi_positions, rss_measurements)):
        if method == "EKF":
            pred, cov = ekf_fuse_one(
                csi_pos, rss_meas, field,
                csi_covariance, scaled_rss_cov,
            )
        elif method == "UKF":
            pred, cov = ukf_fuse_one(
                csi_pos, rss_meas, field,
                csi_covariance, scaled_rss_cov,
            )
        elif method == "PF":
            pred, cov = pf_fuse_one(
                csi_pos, rss_meas, field,
                csi_covariance, scaled_rss_cov,
                particle_count, seed + i * 1009,
            )
        elif method == "FGO":
            pred, cov = fgo_fuse_one(
                csi_pos, rss_meas, field,
                csi_covariance, scaled_rss_cov,
            )
        else:
            raise ValueError(method)

        predictions.append(pred)
        covariances.append(cov)

    return np.asarray(predictions), np.asarray(covariances)


@dataclass
class BaselineCalibration:
    csi_bias: np.ndarray
    csi_covariance: np.ndarray
    rss_bias: np.ndarray
    rss_covariance: np.ndarray
    selected_scales: dict
    search: pd.DataFrame


def calibrate_baselines(
    val_true,
    val_csi_raw,
    val_rss_measurements,
    field,
    args,
) -> BaselineCalibration:
    csi_bias = np.mean(val_csi_raw - val_true, axis=0)
    val_csi = val_csi_raw - csi_bias
    csi_cov = covariance_from_residuals(
        val_csi - val_true, shrinkage=0.50
    )

    clean_rss = np.asarray([field.predict(p) for p in val_true])
    rss_bias = np.mean(val_rss_measurements - clean_rss, axis=0)
    val_rss_measurements = val_rss_measurements - rss_bias
    rss_cov = covariance_from_residuals(
        val_rss_measurements - clean_rss,
        shrinkage=0.65,
        floor=1e-3,
    )

    scale_grid = [float(x) for x in args.rss_scale_grid.split(",")]
    rows = []
    selected = {}

    for method in ["EKF", "UKF", "PF", "FGO"]:
        best_mean = float("inf")
        best_scale = 1.0

        for scale in scale_grid:
            pred, _ = run_nonlinear_method(
                method,
                val_csi,
                val_rss_measurements,
                field,
                csi_cov,
                rss_cov,
                scale,
                min(args.pf_particles, 320) if method == "PF" else args.pf_particles,
                args.seed + 61000,
            )
            e = errors(val_true, pred)
            mean_e = float(np.mean(e))
            rows.append({
                "method": method,
                "rss_covariance_scale": scale,
                "validation_mean_error_m": mean_e,
                "validation_rmse_m": float(np.sqrt(np.mean(e ** 2))),
            })
            if mean_e < best_mean:
                best_mean = mean_e
                best_scale = scale

        selected[method] = best_scale

    return BaselineCalibration(
        csi_bias=np.asarray(csi_bias),
        csi_covariance=csi_cov,
        rss_bias=np.asarray(rss_bias),
        rss_covariance=rss_cov,
        selected_scales=selected,
        search=pd.DataFrame(rows),
    )


# ======================================================================
# CSI confidence for source-style DeepFusion
# ======================================================================

def load_csi_model_and_normalizer(base, tf, output_dir: Path, config: dict):
    n_ant = int(config["csi"]["n_antennas"])
    n_sub = int(config["csi"]["n_subcarriers"])
    input_shape = (4, n_ant, n_sub, 5)

    model = base.create_cnn(
        tf,
        input_shape=input_shape,
        seed=42,
        lr=5e-4,
    )
    model.load_weights(output_dir / "CSI_3D_CNN.weights.h5")

    z = np.load(output_dir / "CSI归一化参数.npz")
    norm = base.Normalizer(
        mean=np.asarray(z["mean"], np.float32),
        std=np.asarray(z["std"], np.float32),
    )
    return model, norm


def csi_prediction_dispersion(
    base,
    model,
    norm,
    base_csi,
    seed,
    config,
    n_samples,
) -> np.ndarray:
    samples, ids = base.augment_csi(
        base_csi,
        int(n_samples),
        int(seed),
        float(config["csi"]["amplitude_noise_db"]),
        float(config["csi"]["phase_noise_deg"]),
        float(config["csi"]["awgn_ratio"]),
    )
    pred = base.denormalize_xy(
        model.predict(
            norm.transform(base.csi_to_tensor(samples)),
            verbose=0,
        )
    )
    pred = clip_positions(pred)

    dispersion = np.zeros(len(base_csi), float)
    for pid in range(len(base_csi)):
        arr = pred[ids == pid]
        if len(arr) <= 1:
            dispersion[pid] = 0.0
        else:
            cov = np.cov(arr, rowvar=False, ddof=1)
            dispersion[pid] = math.sqrt(
                max(float(np.trace(cov)), 1e-12)
            )
    return dispersion


# ======================================================================
# source-style DeepFusion
# ======================================================================

def deepfusion_arrays(
    rss5,
    rss1,
    csi,
    true_xy,
    rss_conf,
    csi_dispersion,
    rss_measurements,
):
    diff = rss1 - csi
    position_pair = np.column_stack([
        rss1[:, 0] / ROOM_W,
        rss1[:, 1] / ROOM_H,
        csi[:, 0] / ROOM_W,
        csi[:, 1] / ROOM_H,
    ])

    context = np.column_stack([
        rss5[:, 0] / ROOM_W,
        rss5[:, 1] / ROOM_H,
        diff[:, 0] / ROOM_W,
        diff[:, 1] / ROOM_H,
        np.linalg.norm(diff, axis=1),
        rss_conf["rss_neighbor_spread"].to_numpy(float),
        rss_conf["rss_nearest_distance"].to_numpy(float),
        rss_conf["rss_distance_ratio"].to_numpy(float),
        np.asarray(csi_dispersion, float),
        np.asarray(rss_measurements, float),
    ])

    return (
        position_pair.astype(np.float32),
        context.astype(np.float32),
        np.asarray(true_xy, np.float32),
    )


def train_deepfusion(
    val_pair,
    val_context,
    val_true,
    test_pair,
    test_context,
    args,
    result_dir,
):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit("DeepFusion需要TensorFlow") from exc

    rng = np.random.default_rng(args.seed + 73000)
    n = len(val_true)
    perm = rng.permutation(n)
    monitor_count = max(8, int(round(0.25 * n)))
    monitor_idx = perm[:monitor_count]
    train_idx = perm[monitor_count:]

    scaler = StandardScaler()
    scaler.fit(val_context[train_idx])
    ctx_train_base = scaler.transform(val_context[train_idx]).astype(np.float32)
    ctx_monitor = scaler.transform(val_context[monitor_idx]).astype(np.float32)
    ctx_test = scaler.transform(test_context).astype(np.float32)

    repeats = max(1, int(args.deep_fusion_augmentations))
    p_train = np.repeat(val_pair[train_idx], repeats, axis=0)
    c_train = np.repeat(ctx_train_base, repeats, axis=0)
    y_train = np.repeat(val_true[train_idx], repeats, axis=0)

    p_train += rng.normal(0, 0.006, size=p_train.shape).astype(np.float32)
    c_train += rng.normal(0, 0.025, size=c_train.shape).astype(np.float32)
    p_train = np.clip(p_train, 0, 1)

    scale_xy = tf.constant([ROOM_W, ROOM_H], tf.float32)

    def radial_loss(y_true, y_pred):
        d = (y_pred - y_true) * scale_xy
        sq = tf.reduce_sum(tf.square(d), axis=-1)
        return tf.reduce_mean(tf.sqrt(sq + 1e-8) + 0.05 * sq)

    def radial_metric(y_true, y_pred):
        d = (y_pred - y_true) * scale_xy
        return tf.reduce_mean(
            tf.sqrt(tf.reduce_sum(tf.square(d), axis=-1) + 1e-8)
        )

    def create_model(seed):
        tf.keras.utils.set_random_seed(seed)

        pos_in = tf.keras.Input(shape=(4,), name="rss1_csi_positions")
        ctx_in = tf.keras.Input(
            shape=(val_context.shape[1],),
            name="fusion_context",
        )
        merged = tf.keras.layers.Concatenate()([pos_in, ctx_in])
        h = tf.keras.layers.Dense(
            48,
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
        )(merged)
        h = tf.keras.layers.Dropout(0.12, seed=seed + 1)(h)
        h = tf.keras.layers.Dense(
            24,
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(3e-4),
        )(h)

        gate = tf.keras.layers.Dense(
            2, activation="sigmoid", name="rss_gate"
        )(h)
        correction = tf.keras.layers.Dense(
            2, activation="tanh", name="bounded_correction"
        )(h)

        rss_pos = tf.keras.layers.Lambda(lambda x: x[:, :2])(pos_in)
        csi_pos = tf.keras.layers.Lambda(lambda x: x[:, 2:])(pos_in)
        one_minus = tf.keras.layers.Lambda(lambda x: 1.0 - x)(gate)

        fused = tf.keras.layers.Add()([
            tf.keras.layers.Multiply()([gate, rss_pos]),
            tf.keras.layers.Multiply()([one_minus, csi_pos]),
        ])

        corr_scale = tf.constant(
            [0.35 / ROOM_W, 0.35 / ROOM_H],
            tf.float32,
        )
        corr = tf.keras.layers.Lambda(
            lambda x: x * corr_scale
        )(correction)

        out = tf.keras.layers.Add(name="normalized_xy")([fused, corr])

        model = tf.keras.Model(
            [pos_in, ctx_in], out,
            name="source_style_deepfusion",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.deep_fusion_learning_rate,
                clipnorm=1.0,
            ),
            loss=radial_loss,
            metrics=[radial_metric],
        )
        return model

    best_weights = None
    best_error = float("inf")
    best_restart = -1
    best_epochs = 0

    for restart in range(args.deep_fusion_restarts):
        seed = args.seed + 80000 + restart * 1000
        tf.keras.backend.clear_session()
        model = create_model(seed)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_radial_metric",
                mode="min",
                patience=35,
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_radial_metric",
                mode="min",
                factor=0.5,
                patience=12,
                min_lr=2e-6,
                verbose=0,
            ),
        ]

        history = model.fit(
            [p_train, c_train],
            val_true[train_idx].repeat(repeats, axis=0)
            / np.asarray([ROOM_W, ROOM_H], np.float32),
            validation_data=(
                [val_pair[monitor_idx], ctx_monitor],
                val_true[monitor_idx]
                / np.asarray([ROOM_W, ROOM_H], np.float32),
            ),
            epochs=args.deep_fusion_epochs,
            batch_size=args.deep_fusion_batch_size,
            callbacks=callbacks,
            verbose=0,
            shuffle=True,
        )

        monitor_pred = (
            model.predict(
                [val_pair[monitor_idx], ctx_monitor],
                verbose=0,
            )
            * np.asarray([ROOM_W, ROOM_H], np.float32)
        )
        mean_e = float(np.mean(errors(
            val_true[monitor_idx],
            monitor_pred,
        )))

        if mean_e < best_error:
            best_error = mean_e
            best_weights = [
                np.array(w, copy=True)
                for w in model.get_weights()
            ]
            best_restart = restart + 1
            best_epochs = len(history.history.get("loss", []))

    if best_weights is None:
        raise RuntimeError("DeepFusion训练失败")

    tf.keras.backend.clear_session()
    model = create_model(
        args.seed + 80000 + (best_restart - 1) * 1000
    )
    model.set_weights(best_weights)

    test_pred = (
        model.predict([test_pair, ctx_test], verbose=0)
        * np.asarray([ROOM_W, ROOM_H], np.float32)
    )
    test_pred = clip_positions(test_pred)

    gate_model = tf.keras.Model(
        model.inputs,
        model.get_layer("rss_gate").output,
    )
    gate = gate_model.predict(
        [test_pair, ctx_test],
        verbose=0,
    )

    pd.DataFrame({
        "rss_gate_x": gate[:, 0],
        "rss_gate_y": gate[:, 1],
        "csi_gate_x": 1 - gate[:, 0],
        "csi_gate_y": 1 - gate[:, 1],
    }).to_csv(
        result_dir / "DeepFusion_gates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model.save_weights(result_dir / "DeepFusion.weights.h5")
    np.savez_compressed(
        result_dir / "DeepFusion_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    info = {
        "selected_restart": best_restart,
        "validation_monitor_mean_error_m": best_error,
        "epochs_run": best_epochs,
        "validation_train_positions": len(train_idx),
        "validation_monitor_positions": monitor_count,
    }
    return test_pred, info


# ======================================================================
# plots
# ======================================================================

def plot_cdf(true_xy, predictions, path):
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    for name, pred in predictions.items():
        e = np.sort(errors(true_xy, pred))
        cdf = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, cdf, lw=1.6, label=name)

    ax.set_xlabel("Localization Error (m)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Fusion Strategy Comparison — Source-Protocol Baselines")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bar(summary, path):
    use = summary[
        summary["method"].isin([
            "RSS+CSI-EKF",
            "RSS+CSI-UKF",
            "RSS+CSI-PF",
            "RSS+CSI-FGO",
            "RSS+CSI-DeepFusion",
            "Proposed Dynamic-DTKF",
        ])
    ].copy()

    x = np.arange(len(use))
    w = 0.26

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.bar(x - w, use["mean_error_m"], w, label="Mean")
    ax.bar(x, use["rmse_m"], w, label="RMSE")
    ax.bar(x + w, use["p95_m"], w, label="P95")
    ax.set_xticks(x)
    ax.set_xticklabels(use["method"], rotation=12)
    ax.set_ylabel("Localization Error (m)")
    ax.set_title("Fusion Strategy Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# main
# ======================================================================

def run(args):
    output_dir = Path(args.output_dir)
    result_dir = output_dir / "Fusion_Strategy_Comparison_SourceProtocol"
    result_dir.mkdir(parents=True, exist_ok=True)

    canonical = load_canonical(output_dir)

    complex_script = find_complex_script(args.complex_script)
    print(f"复杂场景主脚本: {complex_script}")
    base = import_module_from_path(complex_script, "complex_scene_base")

    stack = base.import_stack()
    if not isinstance(stack, (tuple, list)) or len(stack) < 2:
        raise RuntimeError(
            f"complex-scene import_stack() 返回值异常: {type(stack).__name__}, {stack!r}"
        )
    dm, tf = stack[0], stack[1]

    config = canonical["config"]
    aps = canonical["aps"]
    val_df = canonical["val"]
    test_df = canonical["test"]
    calib_npz = canonical["calib"]

    seed = args.seed

    # Base args — only used to reconstruct deterministic physical data.
    base_args = base.parser().parse_args([])
    base_args.seed = seed
    base_args.output_dir = str(output_dir)

    base_args.grid_step = float(config.get("grid_step_m", 0.5))
    base_args.rss_noise_sigma_db = float(config["rss"]["noise_sigma_db"])
    base_args.rss_n_measurements = int(config["rss"]["n_measurements"])
    base_args.rss_k = int(config["rss"]["K"])

    base_args.n_antennas = int(config["csi"]["n_antennas"])
    base_args.n_subcarriers = int(config["csi"]["n_subcarriers"])
    base_args.bandwidth_mhz = float(config["csi"]["bandwidth_mhz"])
    base_args.samples_per_position = int(config["csi"]["samples_per_position"])
    base_args.amplitude_noise_db = float(config["csi"]["amplitude_noise_db"])
    base_args.phase_noise_deg = float(config["csi"]["phase_noise_deg"])
    base_args.awgn_ratio = float(config["csi"]["awgn_ratio"])

    print("=" * 104)
    print("5. FUSION STRATEGY COMPARISON — SOURCE-PROTOCOL REFERENCE")
    print("Mainstream baselines follow the uploaded source-code observation protocol.")
    print("Formal test labels are NEVER used for tuning.")
    print("=" * 104)

    # ------------------------------------------------------------------
    # 1. reconstruct RSS fingerprints / raw measurements
    # ------------------------------------------------------------------
    print("\n[1/5] Reconstructing source-protocol RSS fingerprints and dynamic measurements...")

    wall = base.LargeRoomWallModel()
    rp_xyz = base.make_reference_points(wall, base_args.grid_step)
    rp_xy = rp_xyz[:, :2]

    ref_csi = base.build_csi(
        dm,
        aps,
        rp_xyz,
        wall,
        base_args.exponent,
        base_args.reflection_profile,
        base_args.n_antennas,
        base_args.n_subcarriers,
        base_args.bandwidth_mhz * 1e6,
    )
    fp_db = base.power_db_from_csi(ref_csi)

    val_true = val_df[["true_x", "true_y"]].to_numpy(float)
    test_true = test_df[["true_x", "true_y"]].to_numpy(float)

    val_xyz = np.column_stack([
        val_true,
        np.full(len(val_true), base.RX_Z),
    ])
    test_xyz = np.column_stack([
        test_true,
        np.full(len(test_true), base.RX_Z),
    ])

    val_base_csi = base.build_csi(
        dm, aps, val_xyz, wall,
        base_args.exponent,
        base_args.reflection_profile,
        base_args.n_antennas,
        base_args.n_subcarriers,
        base_args.bandwidth_mhz * 1e6,
    )
    test_base_csi = base.build_csi(
        dm, aps, test_xyz, wall,
        base_args.exponent,
        base_args.reflection_profile,
        base_args.n_antennas,
        base_args.n_subcarriers,
        base_args.bandwidth_mhz * 1e6,
    )

    val_power = base.power_db_from_csi(val_base_csi)
    test_power = base.power_db_from_csi(test_base_csi)

    # same seeds as current complex-scene main experiment
    val_rss_raw = mean_noisy_rss(
        val_power,
        seed + 51000,
        base_args.rss_noise_sigma_db,
        base_args.rss_n_measurements,
    )
    test_rss_raw = mean_noisy_rss(
        test_power,
        seed + 31000,
        base_args.rss_noise_sigma_db,
        base_args.rss_n_measurements,
    )

    # source-style KNN outputs/confidences for DeepFusion
    val_rss5_raw, val_rss_conf = rss_position_and_confidence(
        fp_db, rp_xy, val_rss_raw, 5
    )
    val_rss1_raw, _ = rss_position_and_confidence(
        fp_db, rp_xy, val_rss_raw, 1
    )
    test_rss5_raw, test_rss_conf = rss_position_and_confidence(
        fp_db, rp_xy, test_rss_raw, 5
    )
    test_rss1_raw, _ = rss_position_and_confidence(
        fp_db, rp_xy, test_rss_raw, 1
    )

    rss_k5_bias = np.asarray(calib_npz["rss_k5_bias"], float)
    rss_k1_bias = np.asarray(calib_npz["rss_local_k1_bias"], float)

    val_rss5 = clip_positions(val_rss5_raw - rss_k5_bias)
    val_rss1 = clip_positions(val_rss1_raw - rss_k1_bias)
    test_rss5 = clip_positions(test_rss5_raw - rss_k5_bias)
    test_rss1 = clip_positions(test_rss1_raw - rss_k1_bias)

    # canonical position predictions
    val_csi = val_df[["csi_x", "csi_y"]].to_numpy(float)
    test_csi = test_df[["csi_x", "csi_y"]].to_numpy(float)
    test_dtkf = test_df[["fusion_x", "fusion_y"]].to_numpy(float)

    # Consistency check with saved K1
    saved_test_rss1 = test_df[
        ["rss_dtkf_local_k1_x", "rss_dtkf_local_k1_y"]
    ].to_numpy(float)
    rss1_reconstruction_diff = float(
        np.mean(np.linalg.norm(test_rss1 - saved_test_rss1, axis=1))
    )
    print(
        f"  reconstructed RSS-K1 vs saved canonical mean difference = "
        f"{rss1_reconstruction_diff:.8f} m"
    )

    # ------------------------------------------------------------------
    # 2. source-style nonlinear baseline calibration
    # ------------------------------------------------------------------
    print("\n[2/5] Validation-only calibration for EKF / UKF / PF / FGO...")

    field = RSSFingerprintField(
        rp_xy,
        fp_db,
        n_neighbors=args.fingerprint_field_neighbors,
        length_scale_m=args.fingerprint_field_length_scale,
    )

    calibration = calibrate_baselines(
        val_true,
        val_csi,
        val_rss_raw,
        field,
        args,
    )

    calibration.search.to_csv(
        result_dir / "mainstream_baseline_validation_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("  selected RSS covariance scales:")
    for k, v in calibration.selected_scales.items():
        print(f"    {k}: {v:.2f}")

    corrected_test_csi = test_csi - calibration.csi_bias
    corrected_test_rss = test_rss_raw - calibration.rss_bias

    # ------------------------------------------------------------------
    # 3. mainstream source-style baselines
    # ------------------------------------------------------------------
    print("\n[3/5] Running source-style mainstream baselines...")

    predictions = {
        "RSS-KNN": test_df[["rss_x", "rss_y"]].to_numpy(float),
        "CSI-3D-CNN": test_csi,
    }
    runtimes = []

    name_map = {
        "EKF": "RSS+CSI-EKF",
        "UKF": "RSS+CSI-UKF",
        "PF": "RSS+CSI-PF",
        "FGO": "RSS+CSI-FGO",
    }

    for method in ["EKF", "UKF", "PF", "FGO"]:
        started = time.perf_counter()
        pred, cov = run_nonlinear_method(
            method,
            corrected_test_csi,
            corrected_test_rss,
            field,
            calibration.csi_covariance,
            calibration.rss_covariance,
            calibration.selected_scales[method],
            args.pf_particles,
            seed + 62000,
        )
        display = name_map[method]
        predictions[display] = pred
        runtimes.append({
            "method": display,
            "runtime_s": time.perf_counter() - started,
        })
        np.save(
            result_dir / f"{method.lower()}_covariances.npy",
            cov,
        )

    # ------------------------------------------------------------------
    # 4. source-style DeepFusion
    # ------------------------------------------------------------------
    print("\n[4/5] Building CSI confidence and training source-style DeepFusion...")

    model, norm = load_csi_model_and_normalizer(
        base, tf, output_dir, config
    )

    val_csi_disp = csi_prediction_dispersion(
        base,
        model,
        norm,
        val_base_csi,
        seed + 71000,
        config,
        args.csi_confidence_samples,
    )
    test_csi_disp = csi_prediction_dispersion(
        base,
        model,
        norm,
        test_base_csi,
        seed + 72000,
        config,
        args.csi_confidence_samples,
    )

    val_pair, val_context, val_targets = deepfusion_arrays(
        val_rss5,
        val_rss1,
        val_csi,
        val_true,
        val_rss_conf,
        val_csi_disp,
        val_rss_raw,
    )
    test_pair, test_context, _ = deepfusion_arrays(
        test_rss5,
        test_rss1,
        test_csi,
        test_true,
        test_rss_conf,
        test_csi_disp,
        test_rss_raw,
    )

    started = time.perf_counter()
    deep_pred, deep_info = train_deepfusion(
        val_pair,
        val_context,
        val_targets,
        test_pair,
        test_context,
        args,
        result_dir,
    )
    predictions["RSS+CSI-DeepFusion"] = deep_pred
    runtimes.append({
        "method": "RSS+CSI-DeepFusion",
        "runtime_s": time.perf_counter() - started,
    })

    # Proposed DTKF always comes from canonical run
    predictions["Proposed Dynamic-DTKF"] = test_dtkf

    # ------------------------------------------------------------------
    # 5. metrics
    # ------------------------------------------------------------------
    print("\n[5/5] Formal-test metrics and figures...")

    order = [
        "RSS-KNN",
        "CSI-3D-CNN",
        "RSS+CSI-EKF",
        "RSS+CSI-UKF",
        "RSS+CSI-PF",
        "RSS+CSI-FGO",
        "RSS+CSI-DeepFusion",
        "Proposed Dynamic-DTKF",
    ]

    summary = pd.DataFrame([
        metric_row(name, test_true, predictions[name])
        for name in order
    ])

    dtkf_mean = float(
        summary.loc[
            summary["method"] == "Proposed Dynamic-DTKF",
            "mean_error_m"
        ].iloc[0]
    )
    summary["dtkf_improvement_vs_method_pct"] = (
        summary["mean_error_m"] - dtkf_mean
    ) / np.maximum(summary["mean_error_m"], 1e-12) * 100.0

    summary.to_csv(
        result_dir / "Fusion_Strategy_Comparison_SourceProtocol_汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(runtimes).to_csv(
        result_dir / "Fusion_Strategy_Comparison_SourceProtocol_运行时间.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        result_dir / "mainstream_baseline_covariances.npz",
        csi_bias=calibration.csi_bias,
        csi_covariance=calibration.csi_covariance,
        rss_bias=calibration.rss_bias,
        rss_covariance=calibration.rss_covariance,
    )

    pred_frame = pd.DataFrame({
        "time_s": (
            test_df["time_s"].to_numpy(float)
            if "time_s" in test_df.columns
            else np.arange(len(test_true), dtype=float)
        ),
        "true_x": test_true[:, 0],
        "true_y": test_true[:, 1],
    })
    for name in order:
        pred = predictions[name]
        safe = (
            name.lower()
            .replace("+", "_plus_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        pred_frame[f"{safe}_x"] = pred[:, 0]
        pred_frame[f"{safe}_y"] = pred[:, 1]
        pred_frame[f"{safe}_error_m"] = errors(test_true, pred)

    pred_frame.to_csv(
        result_dir / "Fusion_Strategy_Comparison_SourceProtocol_逐时刻预测.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_cdf(
        test_true,
        {name: predictions[name] for name in order},
        result_dir / "Fusion_Strategy_Comparison_SourceProtocol_CDF.png",
    )
    plot_bar(
        summary,
        result_dir / "Fusion_Strategy_Comparison_SourceProtocol_Bar.png",
    )

    protocol = {
        "benchmark_protocol": (
            "Reference to uploaded source code: CSI position prior + raw 4-AP RSS "
            "fingerprint nonlinear measurement for EKF/UKF/PF/FGO; "
            "validation-only covariance-scale selection; source-style uncertainty-aware "
            "DeepFusion; canonical Dynamic-DTKF unchanged."
        ),
        "test_labels_used_for_tuning": False,
        "selected_rss_covariance_scales": calibration.selected_scales,
        "rss1_reconstruction_difference_m": rss1_reconstruction_diff,
        "deepfusion": deep_info,
        "complex_script": str(complex_script),
    }
    (result_dir / "Fusion_Strategy_Comparison_SourceProtocol_config.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Fusion Strategy Comparison: Formal Test ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nImportant protocol note:")
    print("  EKF/UKF/PF/FGO: CSI position prior + raw RSS fingerprint nonlinear observation.")
    print("  FGO is per-time-step nonlinear optimization; no future-trajectory smoothing.")
    print("  DeepFusion: validation-only source-style gated decision fusion.")
    print("  Dynamic-DTKF: canonical saved result, no retuning.")
    print(f"\n结果目录: {result_dir.resolve()}")


def parser():
    p = argparse.ArgumentParser(
        description="Fusion Strategy Comparison — source-protocol reference version"
    )
    p.add_argument(
        "--output-dir",
        default="结果_复杂大场景_MIC-FD动态融合",
    )
    p.add_argument(
        "--complex-script",
        default="复杂大场景_MIC-FD动态融合实验_红线轨迹_验证优化版.py",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--rss-scale-grid",
        default="0.5,1.0,2.0,4.0,8.0",
    )
    p.add_argument("--pf-particles", type=int, default=1200)
    p.add_argument("--fingerprint-field-neighbors", type=int, default=12)
    p.add_argument("--fingerprint-field-length-scale", type=float, default=0.65)

    p.add_argument("--csi-confidence-samples", type=int, default=20)

    p.add_argument("--deep-fusion-epochs", type=int, default=260)
    p.add_argument("--deep-fusion-batch-size", type=int, default=32)
    p.add_argument("--deep-fusion-learning-rate", type=float, default=5e-4)
    p.add_argument("--deep-fusion-restarts", type=int, default=2)
    p.add_argument("--deep-fusion-augmentations", type=int, default=48)

    return p


def main():
    args = parser().parse_args()

    if args.pf_particles < 100:
        raise ValueError("pf-particles至少100")
    if args.fingerprint_field_neighbors < 3:
        raise ValueError("fingerprint-field-neighbors至少3")
    if args.fingerprint_field_length_scale <= 0:
        raise ValueError("fingerprint-field-length-scale必须>0")
    if args.csi_confidence_samples < 2:
        raise ValueError("csi-confidence-samples至少2")

    set_global_determinism(args.seed)
    run(args)


if __name__ == "__main__":
    main()
