"""
validation.py
-------------
Three checks that test interpretive claims made in the report rather than
re-running the pipeline under different settings (that is
sensitivity_analysis.py's job).

1. Split-half replication of the Statement Network clusters.
   The three large cross-category clusters in Section 5.4 are read off a
   single sample of 85 respondents. Splitting the respondents at random,
   rebuilding the statement network on each half independently and measuring
   how much the recovered clusters overlap tells us whether those clusters
   are a property of the cohort's opinions or an artefact of this particular
   sample.

2. Principal component structure of the 60 items.
   If opinions really are organised around latent value dimensions that cut
   across the survey's T/E/S/V labels, the leading principal components
   should load along the detected clusters rather than along the categories.
   This is a weaker test than a confirmatory factor analysis, but it is the
   test the data can actually support at n = 85.

3. Does network centrality measure "mainstream opinion"?
   Weighted degree measures connectivity in a similarity graph. The report
   reads high degree as "holds a mainstream opinion profile", which is an
   interpretation, not a definition. The direct check is whether degree
   tracks proximity to the cohort's average opinion profile (its centroid).

Run:  python validation.py       (writes output/validation.txt via main.py, or
                                  prints to stdout when run directly)
"""

import numpy as np
import pandas as pd
import networkx as nx

from data_prep import load_raw, clean_and_encode
from build_networks import (respondent_similarity, build_respondent_network,
                            build_statement_network)
from analyze_networks import detect_communities, compute_centralities

RANDOM_SEED = 42
N_SPLITS = 50


# ---------------------------------------------------------------- 1. splits
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def split_half_replication(kept, item_meta, threshold=0.35, n_splits=N_SPLITS,
                           n_clusters=3, seed=RANDOM_SEED):
    """
    Rebuild the statement network on each random half of the respondents and
    ask how well the full-sample clusters are recovered. For each of the
    three largest full-sample clusters we take the best-matching cluster in
    each half and record the Jaccard overlap.
    """
    rng = np.random.default_rng(seed)
    G_full, _ = build_statement_network(kept, item_meta, threshold=threshold)
    memb_full, _, _ = detect_communities(G_full)
    full = pd.Series(memb_full)
    target = [sorted(full[full == c].index)
              for c in full.value_counts().index[:n_clusters]]

    scores = {i: [] for i in range(n_clusters)}
    for _ in range(n_splits):
        ids = rng.permutation(kept.index.to_numpy())
        for half in (ids[:len(ids) // 2], ids[len(ids) // 2:]):
            G_h, _ = build_statement_network(kept.loc[half], item_meta,
                                             threshold=threshold)
            memb_h, _, _ = detect_communities(G_h)
            h = pd.Series(memb_h)
            halves = [sorted(h[h == c].index) for c in h.value_counts().index]
            for i, t in enumerate(target):
                scores[i].append(max((jaccard(t, cand) for cand in halves), default=0.0))

    rows = []
    for i, t in enumerate(target):
        s = np.array(scores[i])
        rows.append({"cluster": f"Cluster {i + 1}", "size_full_sample": len(t),
                     "mean_jaccard": round(float(s.mean()), 3),
                     "median_jaccard": round(float(np.median(s)), 3),
                     "p10": round(float(np.percentile(s, 10)), 3),
                     "p90": round(float(np.percentile(s, 90)), 3)})
    return pd.DataFrame(rows), target


def half_clusters_stay_cross_category(kept, item_meta, threshold=0.35,
                                      n_splits=20, seed=RANDOM_SEED):
    """
    Weaker but more relevant companion to the Jaccard test: even if a half
    sample recovers different cluster MEMBERSHIP, are its large clusters still
    mixed across the survey's four categories? That is the claim Section 5.4
    actually makes.
    """
    rng = np.random.default_rng(seed)
    cat_of = item_meta.set_index("item_id")["category"].to_dict()
    mixed, total, shares = 0, 0, []
    for _ in range(n_splits):
        ids = rng.permutation(kept.index.to_numpy())
        for half in (ids[:len(ids) // 2], ids[len(ids) // 2:]):
            G_h, _ = build_statement_network(kept.loc[half], item_meta, threshold=threshold)
            memb_h, _, _ = detect_communities(G_h)
            h = pd.Series(memb_h)
            for c in h.value_counts().index[:3]:
                items = h[h == c].index
                if len(items) < 5:
                    continue
                cats = pd.Series([cat_of[i] for i in items]).value_counts()
                total += 1
                shares.append(cats.iloc[0] / len(items))
                if len(cats) >= 2 and cats.iloc[0] / len(items) < 0.75:
                    mixed += 1
    return {"large_clusters_examined": total,
            "mixed_across_categories": mixed,
            "pct_mixed": round(100 * mixed / total, 1) if total else None,
            "mean_largest_category_share": round(float(np.mean(shares)), 3) if shares else None}


def within_group_correlation(kept, item_meta, memb_stmt):
    """
    Which labelling groups together items that respondents actually answered
    alike -- the survey's categories, or the detected clusters? Compares mean
    |r| between item pairs inside the same group against the all-pairs
    baseline.
    """
    corr = kept.corr(method="pearson").abs()
    items = list(kept.columns)
    cat_of = item_meta.set_index("item_id")["category"].to_dict()

    def mean_within(labels):
        vals = [corr.loc[a, b] for i, a in enumerate(items) for b in items[i + 1:]
                if labels[a] == labels[b]]
        return float(np.mean(vals)), len(vals)

    all_pairs = [corr.loc[a, b] for i, a in enumerate(items) for b in items[i + 1:]]
    cat_r, cat_n = mean_within(cat_of)
    clu_r, clu_n = mean_within(memb_stmt)
    return pd.DataFrame([
        {"grouping": "all item pairs (baseline)", "n_pairs": len(all_pairs),
         "mean_abs_r": round(float(np.mean(all_pairs)), 3)},
        {"grouping": "same survey category (T/E/S/V)", "n_pairs": cat_n,
         "mean_abs_r": round(cat_r, 3)},
        {"grouping": "same detected cluster", "n_pairs": clu_n,
         "mean_abs_r": round(clu_r, 3)},
    ])


# -------------------------------------------------------------------- 2. PCA
def pca_structure(kept, item_meta, memb_stmt, n_components=4):
    """
    PCA on the standardized 60-item matrix. For each leading component we
    report how its top-loading items distribute across the survey's four
    a-priori categories and across the detected statement clusters. If the
    cross-cutting reading is right, cluster labels should be the tidier
    description of the top loadings.
    """
    X = kept.values.astype(float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    # SVD-based PCA (no sklearn dependency)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_ratio = (S ** 2) / (S ** 2).sum()

    cat_of = item_meta.set_index("item_id")["category"].to_dict()
    rows = []
    for pc in range(n_components):
        load = pd.Series(Vt[pc], index=kept.columns)
        top = load.abs().sort_values(ascending=False).head(12).index
        cats = pd.Series([cat_of[i] for i in top]).value_counts()
        clus = pd.Series([memb_stmt[i] for i in top]).value_counts()
        rows.append({
            "component": f"PC{pc + 1}",
            "var_explained": f"{100 * var_ratio[pc]:.1f}%",
            "distinct_categories_in_top12": int(len(cats)),
            "largest_category_share": f"{cats.iloc[0]}/12",
            "distinct_clusters_in_top12": int(len(clus)),
            "largest_cluster_share": f"{clus.iloc[0]}/12",
        })
    return pd.DataFrame(rows), var_ratio


# ------------------------------------------------------- 3. centrality check
def centrality_vs_centroid(kept, G_resp, centrality):
    """
    Distance from the cohort centroid (the mean opinion profile), measured two
    ways, against each centrality measure. A strong negative correlation with
    Euclidean distance -- or positive with correlation-to-centroid -- is what
    licenses reading high degree as "mainstream".
    """
    centroid = kept.mean(axis=0)
    euclid = ((kept - centroid) ** 2).sum(axis=1) ** 0.5
    corr_to_centroid = kept.T.corrwith(centroid)

    df = centrality.copy()
    df["dist_to_centroid"] = euclid.loc[df.index]
    df["corr_to_centroid"] = corr_to_centroid.loc[df.index]

    rows = []
    for measure in ["weighted_degree", "betweenness", "eigenvector"]:
        rows.append({
            "measure": measure,
            "r_with_distance_to_centroid": round(float(df[measure].corr(df["dist_to_centroid"])), 3),
            "r_with_correlation_to_centroid": round(float(df[measure].corr(df["corr_to_centroid"])), 3),
        })
    return pd.DataFrame(rows), df


def run(kept, item_meta, G_resp, centrality, memb_stmt):
    lines = []

    lines.append("=== 1. Split-half replication of the Statement Network clusters ===")
    lines.append(f"{N_SPLITS} random splits x 2 halves = {2 * N_SPLITS} independent rebuilds.")
    lines.append("Jaccard overlap of each full-sample cluster with its best match in a half.\n")
    rep, target = split_half_replication(kept, item_meta)
    lines.append(rep.to_string(index=False))
    lines.append("\n  Jaccard 1.00 = perfectly recovered; 0.50 = half the items shared.")
    lines.append("\n  Companion check -- do a half sample's large clusters still mix categories?")
    lines.append("  " + str(half_clusters_stay_cross_category(kept, item_meta)))

    lines.append("\n\n=== 1b. Which labelling groups co-answered items together? ===")
    lines.append(within_group_correlation(kept, item_meta, memb_stmt).to_string(index=False))

    lines.append("\n\n=== 2. Principal component structure of the 60 items ===")
    lines.append("Do the leading components align with the survey's categories or with")
    lines.append("the detected clusters? Fewer distinct groups among the top loadings =")
    lines.append("the tidier description.\n")
    pca, var_ratio = pca_structure(kept, item_meta, memb_stmt)
    lines.append(pca.to_string(index=False))
    X = kept.values.astype(float)
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    v1 = np.linalg.svd(Xs, full_matrices=False)[2][0]
    same_sign = max((v1 > 0).mean(), (v1 < 0).mean())
    lines.append(f"\n  First 4 components explain {100 * var_ratio[:4].sum():.1f}% of total variance.")
    lines.append(f"  PC1 loadings share a sign on {100 * same_sign:.0f}% of items "
                 f"-> a general agreement factor, not a topical one.")
    lines.append(f"  Scree after PC1 is flat ({', '.join(f'{100*v:.1f}%' for v in var_ratio[1:5])}), "
                 "so no clean factor structure is identifiable at n = 85.")

    lines.append("\n\n=== 3. Does centrality track proximity to the cohort's average opinion? ===")
    cc, _ = centrality_vs_centroid(kept, G_resp, centrality)
    lines.append(cc.to_string(index=False))
    lines.append("\n  Negative r with distance-to-centroid (and positive r with")
    lines.append("  correlation-to-centroid) means central respondents really do sit")
    lines.append("  closer to the cohort's average opinion profile.")

    return "\n".join(lines)


if __name__ == "__main__":
    raw = load_raw("../data/Survey_Results_UC.csv")
    kept, dropped, meta, n_imputed = clean_and_encode(raw)
    G_resp, sim = build_respondent_network(kept)
    G_stmt, corr = build_statement_network(kept, meta)
    cent = compute_centralities(G_resp)
    memb_stmt, _, _ = detect_communities(G_stmt)
    print(run(kept, meta, G_resp, cent, memb_stmt))
