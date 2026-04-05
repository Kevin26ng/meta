"""
Model Registry for ChaosLab AI Agents.

Auto-discovers trained models in the `models/` directory and provides
a unified interface for loading and listing them.

Supported model types:
  - stable-baselines3 PPO (.zip files)
  - Tabular Q-Learning (.json files)
  - HeuristicAgent (built-in, always available)

Usage:
    from src.model_registry import ModelRegistry

    registry = ModelRegistry()
    info = registry.list_models()       # metadata for all models
    model = registry.load_model("ppo")  # returns predictor with .predict(obs)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── Path to the models directory ──────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@dataclass
class ModelInfo:
    """Metadata for a single registered model."""
    name: str
    display_name: str
    algorithm: str
    description: str
    file_path: Optional[str] = None
    file_size_mb: float = 0.0
    requires_training: bool = True
    available: bool = False


# ── Static catalog of known models ────────────────────────────────
_MODEL_CATALOG: Dict[str, ModelInfo] = {
    "ppo": ModelInfo(
        name="ppo",
        display_name="PPO Neural Net",
        algorithm="PPO (Proximal Policy Optimization)",
        description="On-policy actor-critic with clipped surrogate objective. "
                    "Good balance of stability and sample efficiency.",
        file_path=os.path.join(MODELS_DIR, "ppo_model.zip"),
        requires_training=True,
    ),
    "qlearning": ModelInfo(
        name="qlearning",
        display_name="Tabular Q-Learning",
        algorithm="Q-Learning (Tabular)",
        description="Classic RL with a Q-value lookup table. No neural network — "
                    "pure Python dict as the brain. Learns state-action values.",
        file_path=os.path.join(MODELS_DIR, "qlearning_model.json"),
        requires_training=True,
    ),
    "heuristic": ModelInfo(
        name="heuristic",
        display_name="Heuristic Expert",
        algorithm="Rule-Based Decision Tree",
        description="Hand-coded SRE expert using keyword detection and "
                    "prioritized action sequences. No training required.",
        file_path=None,
        requires_training=False,
        available=True,  # always available
    ),
}


class ModelRegistry:
    """
    Central registry for discovering, loading, and managing AI models.

    Thread-safe for reads. Models are loaded on-demand and cached.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._refresh_availability()

    def _refresh_availability(self):
        """Check which model files exist on disk."""
        for name, info in _MODEL_CATALOG.items():
            if info.file_path and os.path.exists(info.file_path):
                info.available = True
                try:
                    info.file_size_mb = round(
                        os.path.getsize(info.file_path) / (1024 * 1024), 2
                    )
                except OSError:
                    info.file_size_mb = 0.0
            elif not info.requires_training:
                info.available = True
            else:
                info.available = False

    def list_models(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered models."""
        self._refresh_availability()
        result = []
        for name, info in _MODEL_CATALOG.items():
            result.append({
                "name": info.name,
                "display_name": info.display_name,
                "algorithm": info.algorithm,
                "description": info.description,
                "available": info.available,
                "file_size_mb": info.file_size_mb,
                "requires_training": info.requires_training,
            })
        return result

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific model."""
        info = _MODEL_CATALOG.get(name)
        if not info:
            return None
        self._refresh_availability()
        return {
            "name": info.name,
            "display_name": info.display_name,
            "algorithm": info.algorithm,
            "description": info.description,
            "available": info.available,
            "file_size_mb": info.file_size_mb,
            "requires_training": info.requires_training,
        }

    def load_model(self, name: str) -> Tuple[Any, ModelInfo]:
        """
        Load a model by name.

        Returns (predictor, model_info).
        The predictor has a .predict(obs, deterministic=True) method.

        Falls back to heuristic if the requested model is unavailable.
        """
        self._refresh_availability()

        info = _MODEL_CATALOG.get(name)
        if not info:
            # Unknown model name → fallback
            return self._load_heuristic()

        if name == "heuristic":
            return self._load_heuristic()

        if not info.available:
            # Model file not found → fallback
            return self._load_heuristic()

        # Check cache
        if name in self._cache:
            return self._cache[name], info

        # Load model based on type
        try:
            if name == "ppo":
                from stable_baselines3 import PPO
                model = PPO.load(info.file_path)
            elif name == "qlearning":
                from .qlearning_agent import QLearningAgent
                model = QLearningAgent.load(info.file_path)
            else:
                return self._load_heuristic()

            self._cache[name] = model
            return model, info

        except Exception as e:
            print(f"[ModelRegistry] Error loading {name}: {e}")
            return self._load_heuristic()

    def _load_heuristic(self) -> Tuple[Any, ModelInfo]:
        """Load the built-in heuristic expert agent."""
        if "heuristic" not in self._cache:
            from .heuristic_agent import HeuristicAgent
            self._cache["heuristic"] = HeuristicAgent()
        return self._cache["heuristic"], _MODEL_CATALOG["heuristic"]

    def get_display_name(self, name: str) -> str:
        """Get human-readable display name for a model."""
        info = _MODEL_CATALOG.get(name)
        return info.display_name if info else name.upper()


# ── Singleton instance ────────────────────────────────────────────
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
