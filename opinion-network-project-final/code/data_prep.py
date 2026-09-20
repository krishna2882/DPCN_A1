"""
data_prep.py
------------
Loads the raw survey CSV, cleans it, and encodes Likert responses
numerically for network construction.

Categories (from question ID prefix):
    T = Technology (T01-T15)
    E = Education  (E01-E15)
    S = Ethics / Society (S01-S15)
    V = Environment (V01-V15)
"""

import pandas as pd
import numpy as np

LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
    "No Comments": np.nan,
    "": np.nan,
}

CATEGORY_NAMES = {
    "T": "Technology",
    "E": "Education",
    "S": "Ethics",
    "V": "Environment",
}

MAX_MISSING_ALLOWED = 4  # out of 60 items (~93% completeness required)


def load_raw(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df = df.rename(columns={df.columns[0]: "respondent_id"})
    return df


def question_category(col_name):
    """Return one-letter category code from a column header like 'T03. ...'"""
    return col_name[0]


def clean_and_encode(df, impute="median", max_missing=MAX_MISSING_ALLOWED):
    """
    Args:
        impute: how to fill the sparse remaining missingness among retained
                respondents. "median" (default, used for all reported
                results), "mode" (most frequent response for that item), or
                "none" (leave NaN -- used by sensitivity_analysis.py to build
                a complete-case-only variant).
        max_missing: maximum number of the 60 items a respondent may leave
                unanswered and still be retained.

    Returns:
        encoded  : DataFrame (respondents x items), numeric 1-5, NaN for missing,
                   filtered to respondents with acceptable completeness.
        dropped  : DataFrame of respondents removed for excessive missingness.
        item_meta: DataFrame with columns [item_id, category, text]
    """
    item_cols = [c for c in df.columns if c != "respondent_id"]

    item_meta = pd.DataFrame({
        "column": item_cols,
        "item_id": [c.split(".")[0].strip() for c in item_cols],
        "category_code": [question_category(c) for c in item_cols],
        "text": [c.split(".", 1)[1].strip() if "." in c else c for c in item_cols],
    })
    item_meta["category"] = item_meta["category_code"].map(CATEGORY_NAMES)

    encoded = df.copy()
    for c in item_cols:
        encoded[c] = encoded[c].map(
            lambda v: LIKERT_MAP.get(v.strip(), np.nan) if isinstance(v, str) else np.nan
        )
    encoded = encoded.set_index("respondent_id")
    encoded.columns = item_meta["item_id"].values  # rename to short IDs

    n_missing = encoded.isna().sum(axis=1)
    keep_mask = n_missing <= max_missing
    dropped = encoded.loc[~keep_mask].copy()
    kept = encoded.loc[keep_mask].copy()

    n_cells_imputed = int(kept.isna().sum().sum())

    # Impute remaining sparse missingness with the item's column median,
    # rounded to the nearest integer (appropriate for an ordinal 1-5 scale).
    if impute == "median":
        kept = kept.fillna(kept.median(axis=0, skipna=True).round())
    elif impute == "mode":
        kept = kept.fillna(kept.mode(axis=0, dropna=True).iloc[0])
    elif impute == "none":
        pass  # leave NaN in place (complete-case variant)
    else:
        raise ValueError(f"unknown impute strategy: {impute!r}")

    return kept, dropped, item_meta, n_cells_imputed


if __name__ == "__main__":
    raw = load_raw("../data/Survey_Results_UC.csv")
    kept, dropped, meta, n_imputed = clean_and_encode(raw)
    print(f"Total responses collected : {len(raw)}")
    print(f"Retained after cleaning   : {len(kept)}")
    print(f"Dropped (incomplete)      : {len(dropped)}")
    print(f"Cells imputed (median)    : {n_imputed} / {kept.shape[0]*kept.shape[1]}"
          f" ({100*n_imputed/(kept.shape[0]*kept.shape[1]):.2f}%)")
    print(meta.groupby("category").size())
