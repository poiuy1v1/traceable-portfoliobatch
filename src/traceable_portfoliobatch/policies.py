from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyConfig:
    batch_size: int = 5
    cost_budget: float = 12.0
    risk_budget: float | None = None
    alpha: float = 1.0
    beta: float = 0.25
    gamma: float = 0.20
    delta: float = 0.15
    cost_scale: float = 1.0
    diversity_distance_scale: float = 2.0
    random_seed: int | None = None


def _diversity_vector(z: np.ndarray, remaining: np.ndarray, selected: list[int], distance_scale: float) -> np.ndarray:
    if not selected:
        return np.zeros(len(remaining), dtype=float)
    diff = z[remaining, None, :] - z[np.asarray(selected), :][None, :, :]
    distances = np.linalg.norm(diff, axis=2) / math.sqrt(z.shape[1])
    minimum = distances.min(axis=1)
    return np.clip(minimum / float(distance_scale), 0.0, 1.0)


def _priority_vector(method: str, p: np.ndarray, u: np.ndarray, c: np.ndarray, div: np.ndarray, cfg: PolicyConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    normalized_cost = c / max(float(cfg.cost_scale), 1e-12)
    score_component = np.zeros_like(p)
    uncertainty_component = np.zeros_like(p)
    cost_component = np.zeros_like(p)
    diversity_component = np.zeros_like(p)
    if method == "top_score":
        score_component = p.copy()
    elif method == "uncertainty":
        uncertainty_component = u.copy()
    elif method == "cost_aware":
        score_component = p / np.maximum(c, 1e-12)
    elif method == "portfolio_batch":
        score_component = cfg.alpha * p
        uncertainty_component = cfg.beta * u
        cost_component = -cfg.gamma * normalized_cost
        diversity_component = cfg.delta * div
    elif method == "random_baseline":
        pass
    else:
        raise ValueError(f"Unknown method: {method}")
    priority = score_component + uncertainty_component + cost_component + diversity_component
    return priority, {
        "score_component": score_component,
        "uncertainty_component": uncertainty_component,
        "cost_component": cost_component,
        "diversity_component": diversity_component,
        "normalized_cost": normalized_cost,
    }


def select_batch(
    frame: pd.DataFrame,
    z: np.ndarray,
    method: str,
    cfg: PolicyConfig,
    *,
    record_details: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Greedy selection with optional stepwise records for high-volume robustness runs."""
    n = len(frame)
    if n != len(z):
        raise ValueError("frame and descriptor matrix length differ")
    ids = frame["candidate_id"].astype(str).to_numpy()
    p = frame["predicted_success"].astype(float).to_numpy()
    u = frame["uncertainty"].astype(float).to_numpy()
    c = frame["proxy_total_cost"].astype(float).to_numpy()
    r = frame["risk_score"].astype(float).to_numpy()
    selected: list[int] = []
    remaining = np.arange(n, dtype=int)
    cost_used = 0.0
    risk_used = 0.0
    steps: list[dict[str, Any]] = []
    rng = np.random.default_rng(cfg.random_seed if cfg.random_seed is not None else 0)

    def record_for(local_pos: int, priority: np.ndarray, comp: dict[str, np.ndarray], div: np.ndarray, new_cost: np.ndarray, new_risk: np.ndarray, feasible: np.ndarray) -> dict[str, Any]:
        idx = int(remaining[local_pos])
        return {
            "candidate_id": ids[idx],
            "feasible": bool(feasible[local_pos]),
            "new_cost": float(new_cost[local_pos]),
            "new_risk": float(new_risk[local_pos]),
            "predicted_success": float(p[idx]),
            "uncertainty": float(u[idx]),
            "proxy_total_cost": float(c[idx]),
            "risk_score": float(r[idx]),
            "diversity_bonus": float(div[local_pos]),
            "score_component": float(comp["score_component"][local_pos]),
            "uncertainty_component": float(comp["uncertainty_component"][local_pos]),
            "cost_component": float(comp["cost_component"][local_pos]),
            "diversity_component": float(comp["diversity_component"][local_pos]),
            "normalized_cost": float(comp["normalized_cost"][local_pos]),
            "priority": float(priority[local_pos]),
        }

    while len(remaining) and len(selected) < cfg.batch_size:
        new_cost = cost_used + c[remaining]
        new_risk = risk_used + r[remaining]
        cost_ok = new_cost <= cfg.cost_budget + 1e-12
        risk_ok = np.ones(len(remaining), dtype=bool) if cfg.risk_budget is None else new_risk <= cfg.risk_budget + 1e-12
        feasible = cost_ok & risk_ok
        if not feasible.any():
            if record_details:
                steps.append({
                    "step": len(selected) + 1,
                    "status": "stopped_no_feasible_candidate",
                    "remaining_count": int(len(remaining)),
                    "cost_used_before": cost_used,
                    "risk_used_before": risk_used,
                    "excluded_by_cost": int((~cost_ok).sum()),
                    "excluded_by_risk": int((cost_ok & ~risk_ok).sum()),
                })
            break
        div = _diversity_vector(z, remaining, selected, cfg.diversity_distance_scale) if method == "portfolio_batch" else np.zeros(len(remaining))
        priority, comp = _priority_vector(method, p[remaining], u[remaining], c[remaining], div, cfg)
        feasible_pos = np.flatnonzero(feasible)
        if method == "random_baseline":
            chosen_pos = int(rng.choice(feasible_pos))
            ranked_pos = (
                feasible_pos[np.argsort(ids[remaining[feasible_pos]])][:5]
                if record_details else np.asarray([chosen_pos], dtype=int)
            )
        else:
            # Stable tie break: descending priority, then candidate ID.
            order = np.lexsort((ids[remaining[feasible_pos]], -priority[feasible_pos]))
            ranked_pos = feasible_pos[order]
            chosen_pos = int(ranked_pos[0])
        chosen_idx = int(remaining[chosen_pos])
        cost_before, risk_before = cost_used, risk_used
        cost_used = float(new_cost[chosen_pos])
        risk_used = float(new_risk[chosen_pos])
        selected.append(chosen_idx)
        if record_details:
            chosen_record = record_for(chosen_pos, priority, comp, div, new_cost, new_risk, feasible)
            steps.append({
                "step": len(selected),
                "status": "selected",
                "method": method,
                "selected_candidate": chosen_record,
                "cost_used_before": cost_before,
                "risk_used_before": risk_before,
                "cost_used_after": cost_used,
                "risk_used_after": risk_used,
                "remaining_cost_budget": float(cfg.cost_budget - cost_used),
                "remaining_risk_budget": None if cfg.risk_budget is None else float(cfg.risk_budget - risk_used),
                "feasible_count": int(feasible.sum()),
                "excluded_by_cost": int((~cost_ok).sum()),
                "excluded_by_risk": int((cost_ok & ~risk_ok).sum()),
                "top_feasible_candidates": [record_for(int(pos), priority, comp, div, new_cost, new_risk, feasible) for pos in ranked_pos[:5]],
            })
        remaining = np.delete(remaining, chosen_pos)

    selected_df = frame.iloc[selected].copy().reset_index(drop=True)

    # Counterfactual alternatives are generated only for archived primary traces.
    if not record_details:
        return selected_df, frame.iloc[[]].copy().reset_index(drop=True), []
    if len(remaining):
        new_cost = cost_used + c[remaining]
        new_risk = risk_used + r[remaining]
        cost_ok = new_cost <= cfg.cost_budget + 1e-12
        risk_ok = np.ones(len(remaining), dtype=bool) if cfg.risk_budget is None else new_risk <= cfg.risk_budget + 1e-12
        feasible = cost_ok & risk_ok
        div = _diversity_vector(z, remaining, selected, cfg.diversity_distance_scale) if method == "portfolio_batch" else np.zeros(len(remaining))
        priority, comp = _priority_vector(method, p[remaining], u[remaining], c[remaining], div, cfg)
        alt = frame.iloc[remaining].copy()
        alt["counterfactual_priority"] = priority
        alt["counterfactual_feasible"] = feasible
        alt["counterfactual_cost_feasible"] = cost_ok
        alt["counterfactual_risk_feasible"] = risk_ok
        alt["counterfactual_diversity_bonus"] = div
        alt = alt.sort_values(["counterfactual_feasible", "counterfactual_priority", "candidate_id"], ascending=[False, False, True]).head(20)
    else:
        alt = frame.iloc[[]].copy()
    return selected_df, alt.reset_index(drop=True), steps


def metrics(selected: pd.DataFrame) -> dict[str, float | int]:
    hits = int((selected["outcome_binary"].astype(float) > 0).sum())
    cost = float(selected["proxy_total_cost"].sum())
    risk = float(selected["risk_score"].sum())
    return {
        "selected_count": int(len(selected)),
        "hits": hits,
        "hit_fraction": round(hits / len(selected), 6) if len(selected) else 0.0,
        "total_proxy_cost": round(cost, 6),
        "proxy_cost_normalized_yield": round(hits / cost, 6) if cost else 0.0,
        "risk_used": round(risk, 6),
        "mean_predicted_success": round(float(selected["predicted_success"].mean()), 6) if len(selected) else float("nan"),
    }


def config_dict(cfg: PolicyConfig) -> dict[str, Any]:
    return asdict(cfg)
