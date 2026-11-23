"""harness.py -- loading the DiffMean steering vectors and the water-word temperature table.

    load_weight(ds, L)       DiffMean_weight.pt as a float32 (N, D) array, rows in concept_id order
    load_cloud(ds, L)        one dataset's vectors as a cloud, sorted by its intrinsic coordinate
    h2o_on_temp_axis(L)      the 18 water words with their nominal temperatures (Table 1)
    H2O_TEMP                 the paper's water-word temperature table, degrees C
    unit(v)                  row-wise L2 normalisation

The RELGEOM_MODEL switch (model_switch.py) redirects only the weights to another model's vectors.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from relgeom.paths import AXBENCH
from relgeom import model_switch

DATA_BASE = str(AXBENCH)

# Nominal temperature (degrees C) of each water word: the paper's Table 1.
H2O_TEMP = {
    "ice": -10, "snow": -5, "hail": -5, "frost": -3, "sleet": 0, "slush": 1,
    "icy_water": 1, "cold_water": 8, "rain": 10, "dew": 10, "fog": 10, "puddle": 12,
    "cool_water": 15, "roomtemp_water": 21, "warm_water": 35, "hot_water": 55,
    "boiling_water": 100, "steam": 105,
}

H2O_GROUP = {
    "ice": "frozen", "snow": "frozen", "hail": "frozen", "frost": "frozen",
    "sleet": "frozen", "slush": "frozen",
    "rain": "liquid", "dew": "liquid", "fog": "liquid", "puddle": "liquid",
    "steam": "gas",
    "icy_water": "adj_water", "cold_water": "adj_water", "cool_water": "adj_water",
    "roomtemp_water": "adj_water", "warm_water": "adj_water", "hot_water": "adj_water",
    "boiling_water": "adj_water",
}


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _weight_path(dataset, layer, data_base=None):
    # RELGEOM_MODEL switch (model_switch.py): redirects only the WEIGHTS to another model's
    # vectors; concept indices and every other convention stay canonical. Unset -> unchanged.
    return os.path.join(data_base or model_switch.weight_base() or DATA_BASE, dataset,
                        f"diffmean_l{layer}", "train", "DiffMean_weight.pt")


def load_weight(dataset, layer, data_base=None):
    """Load DiffMean_weight.pt as float32 numpy (N, D). Rows are concept_id-ordered.

    data_base : override the tree the WEIGHTS come from, leaving every other convention
                (concept_index, ordering, labels) untouched.
    """
    import torch
    w = torch.load(_weight_path(dataset, layer, data_base), map_location="cpu")
    return w.float().numpy()


def _concept_index(dataset):
    p = os.path.join(DATA_BASE, dataset, "concept_index.parquet")
    return pd.read_parquet(p)


def _order_array(dataset, df):
    """Intrinsic ordering coordinate for each row of df (concept_id order)."""
    name = dataset
    if name == "temp_c1" and "temp_c" in df.columns:
        return df["temp_c"].astype(float).values, "temp_C"
    if name == "h2o_v1":
        return (np.array([H2O_TEMP.get(g, np.nan) for g in df["concept_group"]], float),
                "nominal_temp_C")
    return None, None


def load_cloud(dataset, layer, perspective="steer", sort=True, normalize=False, data_base=None):
    """Load one concept manifold as a (N, D) cloud + its intrinsic ordering.

    perspective : keep only rows with this perspective (every vector in the paper is 'steer').
    sort        : order rows by the intrinsic coordinate (so X[i] is the i-th level).
    normalize   : re-L2-normalise rows (they already are; default off).
    data_base   : read the WEIGHTS from an alternative tree (the concept_index, ordering
                  and labels still come from the real dataset).

    Returns dict: X (N,D float32), order (N,), order_name, labels (level names), vector_ids,
    groups (h2o phase group or None), name, layer, df (the selected concept_index rows).
    """
    df = _concept_index(dataset).reset_index(drop=True)
    W = load_weight(dataset, layer, data_base)
    if len(df) != len(W):
        raise ValueError(f"{dataset} L{layer}: {len(df)} index rows vs {len(W)} weight rows")
    df["_row"] = np.arange(len(df))
    if perspective is not None and "perspective" in df.columns:
        sel = df["perspective"].astype(str) == perspective
        if sel.any():
            df = df[sel].reset_index(drop=True)
    rows = df["_row"].values
    X = W[rows]
    order, order_name = _order_array(dataset, df)
    if sort and order is not None and np.isfinite(order).all():
        si = np.argsort(order, kind="stable")
        df = df.iloc[si].reset_index(drop=True)
        X = X[si]
        order = order[si]
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    groups = (np.array([H2O_GROUP.get(g) for g in df["concept_group"]])
              if dataset == "h2o_v1" else None)
    return dict(
        X=np.ascontiguousarray(X, dtype=np.float32),
        order=order, order_name=order_name,
        labels=df["concept_group"].astype(str).values,
        vector_ids=df["vector_id"].astype(str).values,
        groups=groups, name=dataset, layer=layer, df=df,
    )


def h2o_on_temp_axis(layer):
    """h2o cloud + its nominal temperature + phase group, for projection-onto-temperature
    tests. Returns dict(X, temp, group, labels)."""
    c = load_cloud("h2o_v1", layer)
    return dict(X=c["X"], temp=c["order"], group=c["groups"], labels=c["labels"])
