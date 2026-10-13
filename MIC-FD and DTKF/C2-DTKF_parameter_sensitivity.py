
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOM_W = 12.0
ROOM_H = 8.0


# ======================================================================
# 基础数值函数：与当前复杂场景 DTKF 主代码保持一致
# ======================================================================

def make_spd(m: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    m = np.asarray(m, float)
    m = 0.5 * (m + m.T)
    eigval, eigvec = np.linalg.eigh(m)
    eigval = np.maximum(eigval, floor)
    return eigvec @ np.diag(eigval) @ eigvec.T


def kf_update(
    xp: np.ndarray,
    pp: np.ndarray,
    z: np.ndarray,
    r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
        float
    )
    s = make_spd(h @ pp @ h.T + r, 1e-8)
    k = pp @ h.T @ np.linalg.pinv(s)
    innovation = np.asarray(z, float) - h @ xp
    x = xp + k @ innovation
    i = np.eye(4)
    # Joseph form
    p = (i - k @ h) @ pp @ (i - k @ h).T + k @ r @ k.T
    return x, make_spd(p, 1e-8)


def deduplicate_common_prior(
    x0: np.ndarray,
    p0: np.ndarray,
    xr: np.ndarray,
    pr: np.ndarray,
    xc: np.ndarray,
    pc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    l0 = np.linalg.pinv(p0)
    lr = np.linalg.pinv(pr)
    lc = np.linalg.pinv(pc)

    info = lr + lc - l0
    eta = lr @ xr + lc @ xc - l0 @ x0

    p = make_spd(np.linalg.pinv(info), 1e-6)
    x = p @ eta
    return x, p


def run_dtkf(
    rss_xy: np.ndarray,
    csi_xy: np.ndarray,
    rss_cov: np.ndarray,
    csi_cov: np.ndarray,
    threshold: float,
    dt: float,
    accel_std: float,
    inflation: float,
) -> np.ndarray:
    """
    与当前主实验一致：
    common prior -> RSS local update + CSI local update
    -> innovation gating -> common-prior information deduplication
    """
    rss_xy = np.asarray(rss_xy, float)
    csi_xy = np.asarray(csi_xy, float)

    f = np.asarray(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        float
    )

    q = accel_std ** 2 * np.asarray(
        [[dt**4/4, 0, dt**3/2, 0],
         [0, dt**4/4, 0, dt**3/2],
         [dt**3/2, 0, dt**2, 0],
         [0, dt**3/2, 0, dt**2]],
        float
    )

    x = np.asarray(
        [
            (rss_xy[0, 0] + csi_xy[0, 0]) / 2.0,
            (rss_xy[0, 1] + csi_xy[0, 1]) / 2.0,
            0.0,
            0.0,
        ],
        float
    )
    p = np.diag([1.0, 1.0, 0.5, 0.5])

    h = np.asarray(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
        float
    )

    out = np.zeros((len(rss_xy), 2), float)

    for i in range(len(rss_xy)):
        if i == 0:
            xp, pp = x.copy(), p.copy()
        else:
            xp = f @ x
            pp = make_spd(f @ p @ f.T + q, 1e-6)

        rr = np.asarray(rss_cov, float).copy()
        rc = np.asarray(csi_cov, float).copy()

        # innovation/disagreement gate
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

        x, p = deduplicate_common_prior(
            xp, pp, xr, pr, xc, pc
        )

        x[0] = np.clip(x[0], 0.0, ROOM_W)
        x[1] = np.clip(x[1], 0.0, ROOM_H)
        out[i] = x[:2]

    return out


# ======================================================================
# 数据与指标
# ======================================================================

def scalar_from_npz(data, key: str) -> float:
    if key not in data:
        raise KeyError(f"DTKF标定参数.npz 中缺少字段: {key}")
    arr = np.asarray(data[key], float).reshape(-1)
    if len(arr) == 0:
        raise ValueError(f"字段 {key} 为空")
    return float(arr[0])


def metrics(true_xy: np.ndarray, pred_xy: np.ndarray) -> dict:
    e = np.linalg.norm(np.asarray(pred_xy) - np.asarray(true_xy), axis=1)
    return {
        "mean_error_m": float(np.mean(e)),
        "median_error_m": float(np.median(e)),
        "rmse_m": float(np.sqrt(np.mean(e ** 2))),
        "p90_m": float(np.percentile(e, 90)),
        "p95_m": float(np.percentile(e, 95)),
        "max_error_m": float(np.max(e)),
        "std_error_m": float(np.std(e, ddof=0)),
    }


def parse_multipliers(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("multipliers不能为空")
    if any(v <= 0 for v in vals):
        raise ValueError("multipliers必须全部 > 0")

    # 确保基准 1.0 一定存在
    if not any(abs(v - 1.0) < 1e-10 for v in vals):
        vals.append(1.0)

    return sorted(set(round(v, 10) for v in vals))


def infer_dt(test_frame: pd.DataFrame, fallback: float) -> float:
    if "time_s" in test_frame.columns and len(test_frame) >= 2:
        t = test_frame["time_s"].to_numpy(float)
        d = np.diff(t)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d):
            return float(np.median(d))
    return float(fallback)


def required_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "DTKF_独立动态验证轨迹预测.csv",
        output_dir / "复杂场景_动态轨迹预测.csv",
        output_dir / "DTKF标定参数.npz",
    ]


def load_inputs(output_dir: Path, fallback_dt: float):
    missing = [p for p in required_files(output_dir) if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "缺少复杂场景主实验输出。\n"
            "请先完成一次主实验，然后再运行参数敏感性分析。\n"
            f"缺少：\n{msg}"
        )

    val = pd.read_csv(output_dir / "DTKF_独立动态验证轨迹预测.csv")
    test = pd.read_csv(output_dir / "复杂场景_动态轨迹预测.csv")
    calib = np.load(output_dir / "DTKF标定参数.npz", allow_pickle=False)

    val_required = [
        "true_x", "true_y",
        "rss_local_k1_x", "rss_local_k1_y",
        "csi_x", "csi_y",
    ]
    test_required = [
        "true_x", "true_y",
        "rss_dtkf_local_k1_x", "rss_dtkf_local_k1_y",
        "csi_x", "csi_y",
    ]

    for c in val_required:
        if c not in val.columns:
            raise KeyError(f"验证轨迹CSV缺少列: {c}")
    for c in test_required:
        if c not in test.columns:
            raise KeyError(f"正式测试轨迹CSV缺少列: {c}")

    val_true = val[["true_x", "true_y"]].to_numpy(float)
    val_rss = val[["rss_local_k1_x", "rss_local_k1_y"]].to_numpy(float)
    val_csi = val[["csi_x", "csi_y"]].to_numpy(float)

    test_true = test[["true_x", "true_y"]].to_numpy(float)
    test_rss = test[
        ["rss_dtkf_local_k1_x", "rss_dtkf_local_k1_y"]
    ].to_numpy(float)
    test_csi = test[["csi_x", "csi_y"]].to_numpy(float)

    rss_cov = np.asarray(calib["rss_covariance"], float)
    csi_cov = np.asarray(calib["csi_covariance"], float)

    baseline = {
        "sigma_a": scalar_from_npz(calib, "selected_process_accel_std"),
        "gamma": scalar_from_npz(calib, "selected_gate_inflation"),
        "tau": scalar_from_npz(calib, "innovation_threshold"),
        "rss_cov_scale": scalar_from_npz(calib, "selected_rss_cov_scale"),
        "csi_cov_scale": scalar_from_npz(calib, "selected_csi_cov_scale"),
    }

    dt = infer_dt(test, fallback_dt)

    return {
        "val_frame": val,
        "test_frame": test,
        "val_true": val_true,
        "val_rss": val_rss,
        "val_csi": val_csi,
        "test_true": test_true,
        "test_rss": test_rss,
        "test_csi": test_csi,
        "rss_cov": rss_cov,
        "csi_cov": csi_cov,
        "baseline": baseline,
        "dt": dt,
    }


# ======================================================================
# 单参数敏感性
# ======================================================================

PARAM_META = {
    "sigma_a": {
        "label": r"$\sigma_a$",
        "title": "Process acceleration standard deviation",
        "unit": "m/s²",
        "file_tag": "sigma_a",
    },
    "tau": {
        "label": r"$\tau$",
        "title": "Innovation threshold",
        "unit": "m",
        "file_tag": "tau",
    },
    "gamma": {
        "label": r"$\gamma$",
        "title": "Gate covariance inflation factor",
        "unit": "",
        "file_tag": "gamma",
    },
}


def evaluate_setting(
    sigma_a: float,
    tau: float,
    gamma: float,
    data: dict,
) -> tuple[dict, dict]:
    rr = data["rss_cov"] * data["baseline"]["rss_cov_scale"]
    rc = data["csi_cov"] * data["baseline"]["csi_cov_scale"]

    pred_val = run_dtkf(
        data["val_rss"],
        data["val_csi"],
        rr,
        rc,
        tau,
        data["dt"],
        sigma_a,
        gamma,
    )
    pred_test = run_dtkf(
        data["test_rss"],
        data["test_csi"],
        rr,
        rc,
        tau,
        data["dt"],
        sigma_a,
        gamma,
    )

    return (
        metrics(data["val_true"], pred_val),
        metrics(data["test_true"], pred_test),
    )


def run_one_parameter(
    param_name: str,
    multipliers: list[float],
    data: dict,
) -> pd.DataFrame:
    b = data["baseline"]

    rows = []

    # CSI参考，只用于 improvement 计算
    csi_val_mean = metrics(data["val_true"], data["val_csi"])["mean_error_m"]
    csi_test_mean = metrics(data["test_true"], data["test_csi"])["mean_error_m"]

    # Full baseline error
    baseline_val, baseline_test = evaluate_setting(
        b["sigma_a"], b["tau"], b["gamma"], data
    )
    base_val_mean = baseline_val["mean_error_m"]
    base_test_mean = baseline_test["mean_error_m"]

    for mul in multipliers:
        sigma_a = b["sigma_a"]
        tau = b["tau"]
        gamma = b["gamma"]

        if param_name == "sigma_a":
            sigma_a = b["sigma_a"] * mul
        elif param_name == "tau":
            tau = b["tau"] * mul
        elif param_name == "gamma":
            gamma = max(1.0, b["gamma"] * mul)
        else:
            raise ValueError(param_name)

        val_m, test_m = evaluate_setting(
            sigma_a, tau, gamma, data
        )

        actual_value = {
            "sigma_a": sigma_a,
            "tau": tau,
            "gamma": gamma,
        }[param_name]

        row = {
            "parameter": param_name,
            "multiplier": float(mul),
            "parameter_value": float(actual_value),
            "is_validation_selected_baseline": int(abs(mul - 1.0) < 1e-10),

            "sigma_a": float(sigma_a),
            "tau_m": float(tau),
            "gamma": float(gamma),

            "validation_mean_error_m": val_m["mean_error_m"],
            "validation_median_error_m": val_m["median_error_m"],
            "validation_rmse_m": val_m["rmse_m"],
            "validation_p95_m": val_m["p95_m"],

            "test_mean_error_m": test_m["mean_error_m"],
            "test_median_error_m": test_m["median_error_m"],
            "test_rmse_m": test_m["rmse_m"],
            "test_p90_m": test_m["p90_m"],
            "test_p95_m": test_m["p95_m"],
            "test_max_error_m": test_m["max_error_m"],
            "test_std_error_m": test_m["std_error_m"],

            "validation_degradation_vs_baseline_pct":
                (val_m["mean_error_m"] - base_val_mean)
                / max(base_val_mean, 1e-12) * 100.0,

            "test_degradation_vs_baseline_pct":
                (test_m["mean_error_m"] - base_test_mean)
                / max(base_test_mean, 1e-12) * 100.0,

            "validation_improvement_vs_csi_pct":
                (csi_val_mean - val_m["mean_error_m"])
                / max(csi_val_mean, 1e-12) * 100.0,

            "test_improvement_vs_csi_pct":
                (csi_test_mean - test_m["mean_error_m"])
                / max(csi_test_mean, 1e-12) * 100.0,
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("parameter_value").reset_index(drop=True)


# ======================================================================
# 绘图
# ======================================================================

def plot_parameter(df: pd.DataFrame, param_name: str, baseline_value: float, path: Path):
    meta = PARAM_META[param_name]

    x = df["parameter_value"].to_numpy(float)
    val_mean = df["validation_mean_error_m"].to_numpy(float)
    test_mean = df["test_mean_error_m"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.plot(x, val_mean, marker="o", linewidth=1.8, label="Validation mean error")
    ax.plot(x, test_mean, marker="s", linewidth=1.8, label="Test mean error")
    ax.axvline(
        baseline_value,
        linestyle="--",
        linewidth=1.3,
        label="Validation-selected baseline",
    )

    unit = f" ({meta['unit']})" if meta["unit"] else ""
    ax.set_xlabel(f"{meta['label']}{unit}")
    ax.set_ylabel("Mean Localization Error (m)")
    ax.set_title(f"DTKF Sensitivity to {meta['title']}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 一致性检查
# ======================================================================

def baseline_consistency_check(data: dict) -> pd.DataFrame:
    b = data["baseline"]

    _, test_m_reconstructed = evaluate_setting(
        b["sigma_a"], b["tau"], b["gamma"], data
    )

    rows = []

    # CSV中若保存了主实验 fusion，则检查重构是否一致
    test_frame = data["test_frame"]
    if {"fusion_x", "fusion_y"}.issubset(test_frame.columns):
        saved_fusion = test_frame[["fusion_x", "fusion_y"]].to_numpy(float)
        saved_m = metrics(data["test_true"], saved_fusion)

        rows.append({
            "source": "saved_main_experiment",
            **saved_m,
        })
        rows.append({
            "source": "reconstructed_from_saved_parameters",
            **test_m_reconstructed,
        })

        mean_diff = abs(
            saved_m["mean_error_m"]
            - test_m_reconstructed["mean_error_m"]
        )
        if mean_diff > 1e-6:
            warnings.warn(
                "注意：用NPZ参数重构的Full DTKF与主实验CSV中的fusion结果不完全一致。"
                f" mean差={mean_diff:.8f} m。"
                " 请确认这些文件来自同一次canonical run。"
            )
    else:
        rows.append({
            "source": "reconstructed_from_saved_parameters",
            **test_m_reconstructed,
        })

    return pd.DataFrame(rows)


# ======================================================================
# 主程序
# ======================================================================

def parser():
    p = argparse.ArgumentParser(
        description="DTKF Parameter Sensitivity Analysis - Complex Scenario"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="结果_复杂大场景_MIC-FD动态融合",
        help="复杂场景主实验结果目录",
    )
    p.add_argument(
        "--multipliers",
        type=str,
        default="0.50,0.75,1.00,1.25,1.50",
        help="OFAT相对扰动倍率，必须包含或将自动补充1.0",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="若正式测试CSV没有time_s列时使用的dt",
    )
    return p


def main():
    args = parser().parse_args()

    output_dir = Path(args.output_dir)
    sensitivity_dir = output_dir / "DTKF参数敏感性"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(output_dir, args.dt)
    multipliers = parse_multipliers(args.multipliers)
    b = data["baseline"]

    print("=" * 96)
    print("DTKF PARAMETER SENSITIVITY ANALYSIS")
    print("OFAT: only one parameter is changed at a time.")
    print("The formal test trajectory is NOT used for parameter selection.")
    print("=" * 96)

    print("\nValidation-selected baseline:")
    print(f"  sigma_a = {b['sigma_a']:.6f}")
    print(f"  tau     = {b['tau']:.6f} m")
    print(f"  gamma   = {b['gamma']:.6f}")
    print(f"  R_rss scale = {b['rss_cov_scale']:.6f}")
    print(f"  R_csi scale = {b['csi_cov_scale']:.6f}")
    print(f"  dt = {data['dt']:.6f} s")
    print(f"  multipliers = {multipliers}")

    baseline_table = pd.DataFrame([{
        "selected_process_accel_std_sigma_a": b["sigma_a"],
        "selected_innovation_threshold_tau_m": b["tau"],
        "selected_gate_inflation_gamma": b["gamma"],
        "selected_rss_cov_scale_locked": b["rss_cov_scale"],
        "selected_csi_cov_scale_locked": b["csi_cov_scale"],
        "dt_s": data["dt"],
    }])
    baseline_table.to_csv(
        sensitivity_dir / "DTKF参数敏感性_基准参数.csv",
        index=False,
        encoding="utf-8-sig",
    )

    consistency = baseline_consistency_check(data)
    consistency.to_csv(
        sensitivity_dir / "DTKF参数敏感性_基准一致性检查.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nBaseline consistency check:")
    print(consistency.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    all_frames = []

    for param in ["sigma_a", "tau", "gamma"]:
        df = run_one_parameter(param, multipliers, data)
        all_frames.append(df)

        tag = PARAM_META[param]["file_tag"]
        df.to_csv(
            sensitivity_dir / f"DTKF参数敏感性_{tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        plot_parameter(
            df,
            param,
            b[param],
            sensitivity_dir / f"DTKF参数敏感性_{tag}.png",
        )

        print("\n" + "-" * 96)
        print(f"Sensitivity: {param}")
        cols = [
            "multiplier",
            "parameter_value",
            "validation_mean_error_m",
            "test_mean_error_m",
            "test_rmse_m",
            "test_p95_m",
            "test_degradation_vs_baseline_pct",
            "test_improvement_vs_csi_pct",
        ]
        print(
            df[cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(
        sensitivity_dir / "DTKF参数敏感性_全部结果.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 论文友好型简表
    paper = full[[
        "parameter",
        "multiplier",
        "parameter_value",
        "is_validation_selected_baseline",
        "test_mean_error_m",
        "test_rmse_m",
        "test_p95_m",
        "test_degradation_vs_baseline_pct",
        "test_improvement_vs_csi_pct",
    ]].copy()
    paper.to_csv(
        sensitivity_dir / "DTKF参数敏感性_论文汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 给出“鲁棒区间”简要统计，但绝不据此重新选测试参数
    robust_rows = []
    for param, df in full.groupby("parameter"):
        base_mean = float(
            df.loc[
                df["is_validation_selected_baseline"] == 1,
                "test_mean_error_m"
            ].iloc[0]
        )
        max_degradation = float(
            np.max(
                (df["test_mean_error_m"].to_numpy(float) - base_mean)
                / max(base_mean, 1e-12)
                * 100.0
            )
        )
        min_improvement_vs_csi = float(df["test_improvement_vs_csi_pct"].min())

        robust_rows.append({
            "parameter": param,
            "baseline_test_mean_error_m": base_mean,
            "max_test_degradation_over_sweep_pct": max_degradation,
            "minimum_test_improvement_vs_csi_over_sweep_pct":
                min_improvement_vs_csi,
        })

    robust = pd.DataFrame(robust_rows)
    robust.to_csv(
        sensitivity_dir / "DTKF参数敏感性_鲁棒性摘要.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 96)
    print("ROBUSTNESS SUMMARY")
    print(robust.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n重要：不要根据正式test曲线重新选择参数。")
    print("论文中1.00x必须明确标记为 independent-validation-selected baseline。")
    print(f"\n结果目录: {sensitivity_dir.resolve()}")
    print("=" * 96)


if __name__ == "__main__":
    main()
