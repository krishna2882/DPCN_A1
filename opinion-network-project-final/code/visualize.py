"""
visualize.py
------------
Generates all figures used in the report:
  1. respondent_network.png   - opinion network colored by community
  2. statement_network.png    - statement/topic network colored by a-priori category
  3. degree_distribution.png  - weighted degree histogram (respondent network)
  4. community_profiles.png   - grouped bar chart: mean score per category per community
  5. consensus_vs_polarizing.png - most agreed-upon vs most divisive statements
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from data_prep import load_raw, clean_and_encode, CATEGORY_NAMES
from build_networks import build_respondent_network, build_statement_network
from analyze_networks import (
    compute_centralities, detect_communities, category_profile, statement_extremes
)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

CATEGORY_COLORS = {
    "Technology": "#4C72B0",
    "Education": "#DD8452",
    "Ethics": "#55A868",
    "Environment": "#8172B2",
}

FIG_DIR = "../figures"


def fig_respondent_network(G, membership, centrality):
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.35, iterations=200)
    n_comms = len(set(membership.values()))
    cmap = plt.get_cmap("tab10")
    node_colors = [cmap(membership[n] % 10) for n in G.nodes()]
    sizes = [80 + 900 * (centrality.loc[n, "weighted_degree"] / centrality["weighted_degree"].max())
             for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(7.2, 6))
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=[1.5 * w for w in edge_weights],
                            edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=sizes,
                            edgecolors="white", linewidths=0.6)
    ax.set_title(f"Respondent Opinion Network\n(85 students, {n_comms} opinion communities, "
                 f"node size = weighted degree)")
    ax.axis("off")

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"Community {i+1}",
                           markerfacecolor=cmap(i % 10), markersize=9)
               for i in range(n_comms)]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/respondent_network.png", bbox_inches="tight")
    plt.close(fig)


def fig_statement_network(G, meta, membership_s=None):
    """
    Statement correlation network, drawn as two panels over an identical
    layout: left coloured by the survey's a-priori T/E/S/V label, right by the
    detected Louvain cluster. Reading the same picture twice is what makes the
    cross-cutting claim visible -- clusters do not line up with categories.

    Only the 50-item giant component is drawn. The 10 items that correlate
    with nothing at |r| >= 0.35 carry no edges, so plotting them adds ten
    floating dots and forces every remaining node closer together; they are
    named in the side note instead.
    """
    giant_nodes = max(nx.connected_components(G), key=len)
    isolates = sorted(set(G.nodes()) - set(giant_nodes))
    H = G.subgraph(giant_nodes).copy()

    # Layout on |correlation| so that strongly related items are pulled
    # together regardless of sign.
    for u, v, d in H.edges(data=True):
        d["abs_weight"] = abs(d["weight"])
    pos = nx.spring_layout(H, weight="abs_weight", seed=11, k=1.35,
                           iterations=900, scale=1.0)

    cat_of = nx.get_node_attributes(H, "category")
    deg = dict(H.degree())
    sizes = [110 + 26 * deg[n] for n in H.nodes()]
    pos_edges = [(u, v) for u, v in H.edges() if H[u][v]["weight"] > 0]
    neg_edges = [(u, v) for u, v in H.edges() if H[u][v]["weight"] < 0]

    if membership_s is None:
        from analyze_networks import detect_communities
        membership_s, _, _ = detect_communities(G)
    # colour the three largest clusters; everything else stays grey
    sizes_by_cluster = pd.Series({n: membership_s[n] for n in H.nodes()}).value_counts()
    top3 = list(sizes_by_cluster.index[:3])
    cluster_palette = ["#4C72B0", "#DD8452", "#55A868"]
    cluster_color = {c: cluster_palette[i] for i, c in enumerate(top3)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))

    for ax, mode in zip(axes, ["category", "cluster"]):
        nx.draw_networkx_edges(H, pos, edgelist=pos_edges, ax=ax, alpha=0.18,
                               edge_color="#888888", width=0.8)
        nx.draw_networkx_edges(H, pos, edgelist=neg_edges, ax=ax, alpha=0.8,
                               edge_color="#D62728", style="dashed", width=1.4)
        if mode == "category":
            colors = [CATEGORY_COLORS[cat_of[n]] for n in H.nodes()]
            title = "a. Coloured by original survey category"
        else:
            colors = [cluster_color.get(membership_s[n], "#C8C8C8") for n in H.nodes()]
            title = "b. Coloured by detected correlation cluster"
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color=colors, node_size=sizes,
                               edgecolors="white", linewidths=0.8)
        nx.draw_networkx_labels(H, pos, ax=ax, font_size=5.2, font_color="#1a1a1a")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        ax.margins(0.08)

    cat_handles = [plt.Line2D([0], [0], marker="o", color="w", label=cat,
                              markerfacecolor=col, markersize=8)
                   for cat, col in CATEGORY_COLORS.items()]
    axes[0].legend(handles=cat_handles, loc="lower left", frameon=False, fontsize=7.5)
    clu_handles = [plt.Line2D([0], [0], marker="o", color="w",
                              label=f"Cluster {i+1} (n={int(sizes_by_cluster.iloc[i])})",
                              markerfacecolor=cluster_palette[i], markersize=8)
                   for i in range(len(top3))]
    clu_handles.append(plt.Line2D([0], [0], marker="o", color="w", label="smaller clusters",
                                  markerfacecolor="#C8C8C8", markersize=8))
    clu_handles.append(plt.Line2D([0], [0], color="#D62728", linestyle="--",
                                  label="negative correlation"))
    axes[1].legend(handles=clu_handles, loc="lower left", frameon=False, fontsize=7.5)

    fig.suptitle("Statement (Topic) Correlation Network — 50-item giant component",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.015,
             "Not shown (no edge at |r| >= 0.35): " + ", ".join(isolates),
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    fig.savefig(f"{FIG_DIR}/statement_network.png", bbox_inches="tight")
    plt.close(fig)


def fig_statement_cluster_composition(membership_s, meta):
    """
    Stacked bar: how each of the three large statement clusters is composed of
    the survey's four a-priori categories. This is the quantitative version of
    the cross-cutting claim that the network drawing can only suggest.
    """
    cat_of = meta.set_index("item_id")["category"].to_dict()
    memb = pd.Series(membership_s)
    top3 = memb.value_counts().index[:3]
    cats = list(CATEGORY_NAMES.values())

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bottoms = np.zeros(len(top3))
    for cat in cats:
        vals = np.array([sum(1 for i, c in membership_s.items()
                             if c == cl and cat_of[i] == cat) for cl in top3], dtype=float)
        ax.bar([f"Cluster {i+1}\n(n={int((memb == cl).sum())})" for i, cl in enumerate(top3)],
               vals, bottom=bottoms, label=cat, color=CATEGORY_COLORS[cat], width=0.55)
        for x, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(x, b + v / 2, int(v), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += vals
    ax.set_ylabel("Number of statements")
    ax.set_title("Composition of the Three Large Statement Clusters")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/statement_cluster_composition.png", bbox_inches="tight")
    plt.close(fig)


def fig_degree_distribution(centrality):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(centrality["weighted_degree"], bins=15, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Weighted degree (sum of similarity weights across all incident edges)")
    ax.set_ylabel("Number of respondents")
    ax.set_title("Weighted Degree Distribution — Respondent Network")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/degree_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_community_profiles(profile: pd.DataFrame):
    cats = list(CATEGORY_NAMES.values())
    n_comm = len(profile)
    x = np.arange(n_comm)
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cat in enumerate(cats):
        vals = profile[cat].astype(float).values
        ax.bar(x + (i - 1.5) * width, vals, width, label=cat, color=CATEGORY_COLORS[cat])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Community {i+1}\n(n={int(profile['n_members'].iloc[i])})"
                         for i in range(n_comm)])
    ax.set_ylabel("Mean Likert score (1-5)")
    ax.set_ylim(0, 5.0)
    ax.set_title("Mean Opinion Score by Category and Community")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/community_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig_consensus_vs_polarizing(extremes: pd.DataFrame):
    low = extremes.head(5)
    high = extremes.tail(5)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, data, title, color in [
        (axes[0], low, "Most Consensual Items\n(lowest variance)", "#55A868"),
        (axes[1], high, "Most Polarizing Items\n(highest variance)", "#C44E52"),
    ]:
        labels = [f"{idx}" for idx in data.index]
        ax.barh(labels, data["variance"].astype(float), color=color)
        ax.set_xlabel("Variance")
        ax.set_title(title)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/consensus_vs_polarizing.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raw = load_raw("../data/Survey_Results_UC.csv")
    kept, dropped, meta, n_imputed = clean_and_encode(raw)

    G_resp, sim = build_respondent_network(kept)
    G_stmt, corr = build_statement_network(kept, meta)

    cent = compute_centralities(G_resp)
    memb, mod, n_comm = detect_communities(G_resp)
    profile = category_profile(kept, meta, memb)
    extremes = statement_extremes(kept, meta)

    memb_s, _, _ = detect_communities(G_stmt)
    fig_respondent_network(G_resp, memb, cent)
    fig_statement_network(G_stmt, meta, memb_s)
    fig_statement_cluster_composition(memb_s, meta)
    fig_degree_distribution(cent)
    fig_community_profiles(profile)
    fig_consensus_vs_polarizing(extremes)
    print("All figures written to", FIG_DIR)
