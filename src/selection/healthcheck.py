from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class HealthCheckItem:
    name: str
    ok: bool
    detail: str


CORE_PACKAGES = [
    "numpy",
    "tqdm",
    "yaml",
]

DATA_PACKAGES = [
    "datasets",
]

TRAINING_PACKAGES = [
    "torch",
    "transformers",
    "peft",
]

OPTIONAL_PACKAGES = [
    "dotenv",
    "sentence_transformers",
    "matplotlib",
]


def run_healthcheck() -> List[HealthCheckItem]:
    """Run run healthcheck."""
    items: List[HealthCheckItem] = []
    items.extend(_check_packages("core", CORE_PACKAGES))
    items.extend(_check_packages("data", DATA_PACKAGES))
    items.extend(_check_packages("training", TRAINING_PACKAGES))
    items.extend(_check_packages("optional", OPTIONAL_PACKAGES))
    items.append(_check_path("mcp_config.json", "MCP config"))
    items.append(_check_path("models/retriever", "Retriever directory"))
    items.append(_check_retriever_weights())
    return items


def print_healthcheck(items: Iterable[HealthCheckItem]) -> None:
    """Print print healthcheck."""
    print("\n" + "=" * 60)
    print("Project Healthcheck")
    print("=" * 60)
    for item in items:
        status = "OK" if item.ok else "WARN"
        print(f"[{status}] {item.name}: {item.detail}")

    failed_required = [
        item for item in items
        if not item.ok and (item.name.startswith("core:") or item.name.startswith("data:"))
    ]
    if failed_required:
        print("\nRequired dependencies are missing. Run:")
        print("  pip install -r requirements.txt")
    else:
        print("\nHealthcheck complete.")


def _check_packages(group: str, package_names: Iterable[str]) -> List[HealthCheckItem]:
    """Check check packages."""
    return [
        HealthCheckItem(
            name=f"{group}:{package}",
            ok=importlib.util.find_spec(package) is not None,
            detail="installed" if importlib.util.find_spec(package) is not None else "missing",
        )
        for package in package_names
    ]


def _check_path(path: str, label: str) -> HealthCheckItem:
    """Check check path."""
    exists = os.path.exists(path)
    return HealthCheckItem(
        name=label,
        ok=exists,
        detail=path if exists else f"missing: {path}",
    )


def _check_retriever_weights() -> HealthCheckItem:
    """Check check retriever weights."""
    model_path = os.getenv("RETRIEVER_MODEL_PATH", "models/retriever")
    weight_names = {
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack",
        "model.safetensors.index.json",
    }
    has_weights = False
    if os.path.isdir(model_path):
        for _, _, files in os.walk(model_path):
            if weight_names.intersection(files):
                has_weights = True
                break

    return HealthCheckItem(
        name="Retriever weights",
        ok=has_weights,
        detail="found" if has_weights else "missing; lexical fallback will be used",
    )
