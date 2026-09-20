"""
main.py
-------
End-to-end pipeline: clean data -> build networks -> analyze -> visualize ->
export summary tables. Run this single file to reproduce every number and
figure used in the report.

Usage:
    python main.py
"""

import json
import pandas as pd

from data_prep import load_raw, clean_and_encode
from build_networks import build_respondent_network, build_statement_network
from analyze_networks import (
    network_summary, compute_centralities, detect_communities,
    category_profile, statement_extremes, community_item_deltas,
    top_distinctive_items, statement_cluster_table,
)
import visualize as viz
import validation

DATA_PATH = "../data/Survey_Results_UC.csv"
OUT_DIR = "../output"


def main():
    raw = load_raw(DATA_PATH)
    kept, dropped, meta, n_cells_imputed_actual = clean_and_encode(raw)

    G_resp, sim = build_respondent_network(kept)
    G_stmt, corr = build_statement_network(kept, meta)

    cent = compute_centralities(G_resp)
    memb, mod, n_comm = detect_communities(G_resp)
    profile = category_profile(kept, meta, memb)
    extremes = statement_extremes(kept, meta)
    memb_s, mod_s, n_comm_s = detect_communities(G_stmt)
    deltas = community_item_deltas(kept, meta, memb)
    distinctive = top_distinctive_items(deltas)
    stmt_clusters = statement_cluster_table(memb_s, meta)

    # ---- persist tables -------------------------------------------------
    cent.to_csv(f"{OUT_DIR}/respondent_centrality.csv")
    profile.to_csv(f"{OUT_DIR}/community_profiles.csv")
    extremes.to_csv(f"{OUT_DIR}/statement_extremes.csv")
    pd.Series(memb, name="community").to_csv(f"{OUT_DIR}/respondent_communities.csv")
    deltas.to_csv(f"{OUT_DIR}/community_item_deltas.csv")
    distinctive.to_csv(f"{OUT_DIR}/community_distinctive_items.csv", index=False)
    stmt_clusters.to_csv(f"{OUT_DIR}/statement_clusters.csv", index=False)

    summary = {
        "n_responses_collected": int(len(raw)),
        "n_responses_retained": int(len(kept)),
        "n_responses_dropped": int(len(dropped)),
        "n_cells_imputed": int(n_cells_imputed_actual),
        "respondent_network": network_summary(G_resp, "Respondent"),
        "statement_network": network_summary(G_stmt, "Statement"),
        "respondent_communities": {"count": n_comm, "modularity": round(mod, 4)},
        "statement_communities": {"count": n_comm_s, "modularity": round(mod_s, 4)},
        "overall_mean_by_category": {
            cat: round(float(kept[[c for c in kept.columns if meta.set_index('item_id')['category'][c] == cat]].values.mean()), 3)
            for cat in profile.columns if cat not in ("n_members", "overall_mean")
        },
        "pct_agree_or_strongly_agree": round(float((kept.values >= 4).mean() * 100), 1),
        "pct_disagree_or_strongly_disagree": round(float((kept.values <= 2).mean() * 100), 1),
        "degree_betweenness_corr": round(float(cent["weighted_degree"].corr(cent["betweenness"])), 3),
        "degree_eigenvector_corr": round(float(cent["weighted_degree"].corr(cent["eigenvector"])), 3),
    }
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- validation checks (split-half replication, PCA, centrality meaning)
    with open(f"{OUT_DIR}/validation.txt", "w") as f:
        f.write(validation.run(kept, meta, G_resp, cent, memb_s) + "\n")

    # ---- figures ----------------------------------------------------------
    viz.fig_respondent_network(G_resp, memb, cent)
    viz.fig_statement_network(G_stmt, meta, memb_s)
    viz.fig_statement_cluster_composition(memb_s, meta)
    viz.fig_degree_distribution(cent)
    viz.fig_community_profiles(profile)
    viz.fig_consensus_vs_polarizing(extremes)

    print(json.dumps(summary, indent=2))
    print(f"\nTables written to {OUT_DIR}/, figures written to ../figures/")


if __name__ == "__main__":
    main()
