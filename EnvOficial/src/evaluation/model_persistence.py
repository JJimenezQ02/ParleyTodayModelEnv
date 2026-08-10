"""Serializacion de modelos entrenados listos para produccion.

Un bundle empaqueta todo lo necesario para predecir mas adelante: el modelo,
el imputer fiteado en train, la lista de features en orden, y los metadatos
que permiten auditar de donde salio (hiperparametros, metricas, fecha).

Las rutas salen del `ModelConfig` (`models_dir`), igual que las metricas y los
configs, asi que otro target cae en su propia carpeta sin tocar codigo.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config.config import ModelConfig
from src.metrics.distribution_metrics import DistFamily, get_mean
from src.training.distribution_models import needs_imputation, predict_params
from src.training.distribution_models import clip_params


def model_stem(
    model_name: str,
    family: DistFamily,
    component: str,
    n_features: Optional[int] = None,
) -> str:
    """Nombre canonico del bundle, sin extension.

    Incluye familia y componente para que Poisson/NegBin y home/away no se
    pisen entre si: `ngboost_poisson_home_42`.
    """
    parts = [model_name.lower(), str(family), component]
    if n_features is not None:
        parts.append(str(n_features))
    return "_".join(p for p in parts if p)


def build_bundle(
    model: Any,
    model_name: str,
    family: DistFamily,
    component: str,
    features: Sequence[str],
    config: ModelConfig,
    imputer: Optional[Any] = None,
    best_params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Empaqueta modelo + preprocesado + metadatos.

    Parameters
    ----------
    features : orden exacto de columnas con el que se entreno. `predict_bundle`
        reindexa con esta lista, asi que un DataFrame con las columnas en otro
        orden sigue funcionando.
    imputer  : el fiteado en train. Obligatorio si el backend lo necesita;
        sin el, las predicciones futuras no reproducen el preprocesado.
    metrics  : metricas de evaluacion a dejar registradas (test_rps, test_nll...).
    """
    if imputer is None and needs_imputation(model_name):
        raise ValueError(
            f"{model_name} requiere imputacion pero no se paso `imputer`. "
            f"Llama a train_and_predict(..., return_imputer=True)."
        )

    target_column: str = config.target.component(component).column

    metadata: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_name,
        "family": family,
        "component": component,
        "target_name": config.target.name,
        "target_column": target_column,
        "feature_names": list(features),
        "n_features": len(list(features)),
        "max_k": config.target.max_k,
        "thresholds": config.target.thresholds,
        "numeric_guards": config.numeric_guards,
        "best_params": dict(best_params) if best_params else {},
        "metrics": dict(metrics) if metrics else {},
    }
    if extra:
        metadata.update(extra)

    return {"metadata": metadata, "model": model, "imputer": imputer}


def save_bundle(
    bundle: Dict[str, Any],
    config: ModelConfig,
    name: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Path:
    """Guarda un bundle en `outputs/clubs/models/<target>/<name>.pkl`."""
    meta: Dict[str, Any] = bundle["metadata"]
    stem: str = name or model_stem(
        meta["model_type"], meta["family"], meta["component"],
        meta.get("n_features"),
    )

    directory: Path = Path(models_dir) if models_dir else config.models_dir
    directory.mkdir(parents=True, exist_ok=True)
    path: Path = directory / f"{stem}.pkl"

    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        size_mb: float = path.stat().st_size / (1024 ** 2)
        print(f" [MODEL] {stem} -> {path}  ({size_mb:.1f} MB)")

    return path


def load_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    """Carga un bundle serializado."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_bundle(
    bundle_or_path: Union[Dict[str, Any], str, Path],
    X: pd.DataFrame,
) -> Dict[str, NDArray]:
    """Predice los parametros de la distribucion con un bundle guardado.

    Reproduce el preprocesado del entrenamiento: selecciona y reordena las
    features, aplica el imputer y las guardas numericas.

    Returns
    -------
    Poisson -> {'lambda': ...}; NegBin -> {'mu': ..., 'alpha': ...}.
    """
    bundle: Dict[str, Any] = (
        bundle_or_path if isinstance(bundle_or_path, dict)
        else load_bundle(bundle_or_path)
    )
    meta: Dict[str, Any] = bundle["metadata"]
    features: List[str] = [str(c) for c in meta["feature_names"]]

    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"Faltan features en el input: {missing}")

    # Reindexado explicito: el orden de columnas importa para los backends.
    X_use: pd.DataFrame = X.loc[:, features]

    imputer = bundle.get("imputer")
    if imputer is not None:
        # `transform` ya devuelve un DataFrame (set_output pandas) pero con las
        # columnas en orden num->cat: se reindexa al orden de entrenamiento en
        # vez de reetiquetar, que mezclaria features silenciosamente.
        X_use = imputer.transform(X_use)[features]

    raw: Dict[str, NDArray] = predict_params(
        bundle["model"], X_use, meta["model_type"], meta["family"],
    )
    return clip_params(raw, meta["family"], meta.get("numeric_guards"))


def predict_distribution(
    bundle_or_path: Union[Dict[str, Any], str, Path],
    X: pd.DataFrame,
    max_k: Optional[int] = None,
) -> pd.DataFrame:
    """PMF por observacion sobre 0..max_k, con la ultima celda acumulando la cola.

    Returns
    -------
    DataFrame con `mean_pred`, `p_0`..`p_<max_k-1>` y `p_ge_<max_k>`.
    """
    from src.metrics.distribution_metrics import get_pmf

    bundle: Dict[str, Any] = (
        bundle_or_path if isinstance(bundle_or_path, dict)
        else load_bundle(bundle_or_path)
    )
    meta: Dict[str, Any] = bundle["metadata"]
    family: DistFamily = meta["family"]

    params: Dict[str, NDArray] = predict_bundle(bundle, X)
    k: int = int(max_k if max_k is not None else meta.get("max_k", 8))

    pmf: NDArray = get_pmf(np.arange(k, dtype=int), params, family)
    frame: Dict[str, NDArray] = {"mean_pred": get_mean(params, family)}
    for idx in range(k):
        frame[f"p_{idx}"] = pmf[:, idx]
    frame[f"p_ge_{k}"] = np.clip(1.0 - pmf.sum(axis=1), 0.0, 1.0)

    return pd.DataFrame(frame, index=X.index)
