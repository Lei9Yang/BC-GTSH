from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .performance import calPerformance


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sign_codes(values: np.ndarray) -> np.ndarray:
    codes = np.where(values >= 0, 1, -1)
    return codes.astype(np.int8)


def compute_retrieval_metrics(
    query_labels: np.ndarray,
    retrieval_labels: np.ndarray,
    query_codes: np.ndarray,
    retrieval_codes: np.ndarray,
    train_elapsed_time: float,
    hashcode_generate_time: float,
    top_k: list[int],
    ratio_n: list[float],
    ndcg_ks: list[int],
    precision_curve_ks: list[int],
    efficiency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval_start = time.time()
    return calPerformance(
        queryLabel=query_labels.astype(np.float32),
        retrievalLabel=retrieval_labels.astype(np.float32),
        queryB=query_codes.astype(np.float32),
        retrievalB=retrieval_codes.astype(np.float32),
        topK=top_k,
        N=ratio_n,
        ndcgKs=ndcg_ks,
        precisionCurveKs=precision_curve_ks,
        hashcode_generate_time=hashcode_generate_time,
        retrieval_start_time=retrieval_start,
        train_elapsed_time=train_elapsed_time,
        efficiency=efficiency,
    )


def bounded_ks(candidates: list[int], n: int) -> list[int]:
    ks = sorted({max(1, min(int(k), n)) for k in candidates})
    return ks or [max(1, n)]


def save_json(path: Path, payload: Any, save_ap: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload, save_ap=save_ap), f, indent=4)
    tmp.replace(path)


def json_ready(value: Any, save_ap: bool = False) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key == "ap_per_query" and not save_ap:
                continue
            output[str(key)] = json_ready(item, save_ap=save_ap)
        return output
    if isinstance(value, (list, tuple)):
        return [json_ready(item, save_ap=save_ap) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value.__class__.__module__.startswith("torch") and hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value
