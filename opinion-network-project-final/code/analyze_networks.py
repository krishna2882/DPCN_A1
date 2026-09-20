"""
analyze_networks.py
--------------------
Computes centrality measures, detects communities (Louvain), and profiles
each community's average opinion by topic category.
"""

import numpy as np
import pandas as pd
import networkx as nx
from data_prep import load_raw, clean_and_encode, CATEGORY_NAMES
from build_networks import build_respondent_network, build_statement_network

RANDOM_SEED = 42


def network_summary(G, name):
    return {
        "network": name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "avg_clustering": round(nx.average_clustering(G, weight="weight"), 4),
        "avg_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        "components": nx.number_connected_components(G),
        "avg_path_length": (
            round(nx.average_shortest_path_length(G), 3)
            if nx.is_connected(G) else None
        ),
    }


def compute_centralities(G):
    deg = dict(G.degree(weight="weight"))

    # Betweenness centrality treats the "weight" attribute as a distance/cost
    # for shortest-path computation (networkx: "weights are interpreted as
    # distances"). Our edge weight is a similarity (Pearson correlation), so
    # using it directly would make *weaker* similarities look like *shorter*
    # paths -- the opposite of the intended meaning. We therefore compute
    # betweenness on a distance-transformed copy of the graph, d_ij = 1 - w_ij
    # (valid because all retained edges have w > 0 by construction), so that
    # strongly similar peers are "close" and weakly similar peers are "far".
    # Degree (a simple sum) and eigenvector centrality (which treats weight as
    # connection strength, not distance) are unaffected by this issue and are
    # computed directly on the similarity-weighted graph.
    G_dist = nx.Graph()
    G_dist.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        G_dist.add_edge(u, v, distance=1.0 - data["weight"])
    btw = nx.betweenness_centrality(G_dist, weight="distance", normalized=True, seed=RANDOM_SEED)

    eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    clus = nx.clustering(G, weight="weight")
    df = pd.DataFrame({
        "weighted_degree": deg,
        "betweenness": btw,
        "eigenvector": eig,
        "clustering": clus,
    })
    return df.sort_values("weighted_degree", ascending=False)


def detect_communities(G):
    """Louvain community detection: a fast greedy heuristic for approximately
    maximizing modularity (it does not guarantee the global optimum)."""
    comms = nx.community.louvain_communities(G, weight="weight", seed=RANDOM_SEED, resolution=1.0)
    membership = {}
    for i, c in enumerate(comms):
        for node in c:
            membership[node] = i
    modularity = nx.community.modularity(G, comms, weight="weight")
    return membership, modularity, len(comms)


def category_profile(kept: pd.DataFrame, item_meta: pd.DataFrame, membership: dict):
    """Average Likert score per category, per community."""
    cat_of_item = item_meta.set_index("item_id")["category"].to_dict()
    cat_cols = {cat: [i for i, c in cat_of_item.items() if c == cat]
                for cat in CATEGORY_NAMES.values()}

    prof = pd.DataFrame(index=sorted(set(membership.values())),
                         columns=list(cat_cols.keys()) + ["n_members", "overall_mean"])
    memb_series = pd.Series(membership)
    for comm_id in prof.index:
        members = memb_series[memb_series == comm_id].index
        sub = kept.loc[members]
        for cat, cols in cat_cols.items():
            prof.loc[comm_id, cat] = round(sub[cols].values.mean(), 2)
        prof.loc[comm_id, "n_members"] = len(members)
        prof.loc[comm_id, "overall_mean"] = round(sub.values.mean(), 2)
    return prof


def statement_cluster_table(membership_s: dict, item_meta: pd.DataFrame):
    """
    Membership of each statement-network Louvain cluster, with the a-priori
    T/E/S/V composition of each cluster, sorted largest first. Written to
    output/ so the cross-category clusters discussed in the report can be
    inspected directly instead of being read off the network figure.
    """
    meta_idx = item_meta.set_index("item_id")
    rows = []
    sizes = pd.Series(membership_s).value_counts()
    for rank, (cluster_id, size) in enumerate(sizes.items(), start=1):
        items = [i for i, c in membership_s.items() if c == cluster_id]
        items = sorted(items)
        comp = meta_idx.loc[items, "category"].value_counts().to_dict()
        rows.append({
            "cluster_rank": rank,
            "cluster_id": cluster_id,
            "size": size,
            "composition": "; ".join(f"{k}:{v}" for k, v in sorted(comp.items())),
            "items": " ".join(items),
        })
    return pd.DataFrame(rows)


def community_item_deltas(kept: pd.DataFrame, item_meta: pd.DataFrame, membership: dict):
    """
    For every one of the 60 statements and every detected community, the
    community's mean score minus the whole-cohort mean for that statement.

    This is the table behind the item-level claims in the report's discussion
    of what distinguishes each community (e.g. "S02 is +0.91 above the cohort
    mean for Community 4"), so those numbers are reproducible from the repo
    rather than quoted without provenance.
    """
    memb = pd.Series(membership)
    cohort_mean = kept.mean(axis=0)
    cols = {}
    for comm_id in sorted(set(membership.values())):
        members = memb[memb == comm_id].index
        cols[f"community_{comm_id + 1}"] = (kept.loc[members].mean(axis=0) - cohort_mean).round(3)
    out = pd.DataFrame(cols)
    out.insert(0, "cohort_mean", cohort_mean.round(3))
    out.insert(0, "category", item_meta.set_index("item_id")["category"].loc[out.index])
    out["text"] = item_meta.set_index("item_id")["text"].loc[out.index]
    return out


def top_distinctive_items(deltas: pd.DataFrame, n=5):
    """The n items on which each community deviates most from the cohort."""
    rows = []
    comm_cols = [c for c in deltas.columns if c.startswith("community_")]
    for c in comm_cols:
        ranked = deltas[c].astype(float).abs().sort_values(ascending=False).head(n)
        for item in ranked.index:
            rows.append({"community": c, "item_id": item,
                         "category": deltas.loc[item, "category"],
                         "delta_vs_cohort": deltas.loc[item, c],
                         "cohort_mean": deltas.loc[item, "cohort_mean"],
                         "text": deltas.loc[item, "text"]})
    return pd.DataFrame(rows)


def statement_extremes(kept: pd.DataFrame, item_meta: pd.DataFrame):
    """Items with highest consensus (low variance) and most polarizing (high variance)."""
    var = kept.var(axis=0).sort_values()
    mean = kept.mean(axis=0)
    text = item_meta.set_index("item_id")["text"]
    cat = item_meta.set_index("item_id")["category"]
    out = pd.DataFrame({"variance": var, "mean": mean.loc[var.index],
                         "category": cat.loc[var.index], "text": text.loc[var.index]})
    return out


if __name__ == "__main__":
    raw = load_raw("../data/Survey_Results_UC.csv")
    kept, dropped, meta, n_imputed = clean_and_encode(raw)

    G_resp, sim = build_respondent_network(kept)
    G_stmt, corr = build_statement_network(kept, meta)

    print("=== Network summaries ===")
    print(pd.DataFrame([network_summary(G_resp, "Respondent"),
                         network_summary(G_stmt, "Statement")]))

    print("\n=== Respondent centrality (top 5) ===")
    cent = compute_centralities(G_resp)
    print(cent.head())

    print("\n=== Respondent centrality (bottom 5 / peripheral) ===")
    print(cent.tail())

    memb, mod, n_comm = detect_communities(G_resp)
    print(f"\n=== Communities: {n_comm} found, modularity={mod:.3f} ===")
    prof = category_profile(kept, meta, memb)
    print(prof)

    print("\n=== Most consensual items (lowest variance) ===")
    ext = statement_extremes(kept, meta)
    print(ext.head(5)[["category", "mean", "variance", "text"]])

    print("\n=== Most polarizing items (highest variance) ===")
    print(ext.tail(5)[["category", "mean", "variance", "text"]])

    memb_s, mod_s, n_comm_s = detect_communities(G_stmt)
    print(f"\n=== Statement-network communities: {n_comm_s} found, modularity={mod_s:.3f} ===")
    # cross-tab: does statement-network community align with a-priori category?
    memb_s_series = pd.Series(memb_s, name="cluster")
    cat_series = meta.set_index("item_id")["category"]
    xt = pd.crosstab(cat_series.loc[memb_s_series.index], memb_s_series)
    print(xt)
