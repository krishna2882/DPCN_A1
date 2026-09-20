"""
build_networks.py
------------------
Constructs two complementary networks from the cleaned opinion survey:

1. Respondent Opinion Network (primary)
   Nodes  = survey respondents
   Edges  = pairwise Pearson correlation between each pair of respondents'
            60-item opinion vectors, connected via a k-nearest-neighbour (k-NN)
            graph *symmetrized by union*: an edge (i, j) is added if i is
            among j's k most-similar peers, OR j is among i's k most-similar
            peers (not necessarily both). This is a deliberate choice over a
            *mutual* (intersection) k-NN graph, which tends to fragment into
            a disconnected/sparser graph for k this small; the union
            construction keeps the network connected while still bounding
            each node to at most k edges from its own selections (a node's
            total degree can exceed k if other nodes also select it).
   Pearson correlation (rather than raw distance/cosine similarity) is used
   because it measures similarity in the *pattern* of relative agreement
   across topics, controlling for individuals who simply agree/disagree
   with everything (acquiescence / scale-use bias).

2. Statement (Topic) Network (secondary)
   Nodes  = the 60 survey statements
   Edges  = |Pearson correlation| between statements across the 85
            respondents, thresholded.
   Reveals how topics/questions cluster together conceptually based on how
   the cohort actually answered them (as opposed to the a-priori T/E/S/V
   labelling).
"""

import numpy as np
import pandas as pd
import networkx as nx
from data_prep import load_raw, clean_and_encode, CATEGORY_NAMES

K_NEIGHBORS = 8          # for respondent k-NN graph
STATEMENT_CORR_THRESHOLD = 0.35


def respondent_similarity(kept: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation between respondents (rows)."""
    return kept.T.corr(method="pearson")


def build_respondent_network(kept: pd.DataFrame, k=K_NEIGHBORS) -> nx.Graph:
    sim = respondent_similarity(kept)
    ids = sim.index.tolist()
    G = nx.Graph()
    G.add_nodes_from(ids)

    # k-NN graph symmetrized by UNION: connect each node to its k most-similar
    # peers; a node's final degree can exceed k because other nodes may also
    # select it (see module docstring for why union rather than mutual/
    # intersection symmetrization was used).
    for i in ids:
        row = sim.loc[i].drop(index=i)
        top_k = row.sort_values(ascending=False).head(k)
        for j, w in top_k.items():
            if w > 0:  # only keep positive-similarity edges
                if G.has_edge(i, j):
                    # keep the max weight if already added from the other side
                    G[i][j]["weight"] = max(G[i][j]["weight"], float(w))
                else:
                    G.add_edge(i, j, weight=float(w))
    return G, sim


def build_statement_network(kept: pd.DataFrame, item_meta: pd.DataFrame,
                             threshold=STATEMENT_CORR_THRESHOLD) -> nx.Graph:
    corr = kept.corr(method="pearson")
    ids = corr.index.tolist()
    G = nx.Graph()
    cat_lookup = item_meta.set_index("item_id")["category"].to_dict()
    text_lookup = item_meta.set_index("item_id")["text"].to_dict()
    for i in ids:
        G.add_node(i, category=cat_lookup[i], text=text_lookup[i])

    for a_idx, a in enumerate(ids):
        for b in ids[a_idx + 1:]:
            w = corr.loc[a, b]
            if abs(w) >= threshold:
                G.add_edge(a, b, weight=float(w), sign="pos" if w > 0 else "neg")
    return G, corr


if __name__ == "__main__":
    raw = load_raw("../data/Survey_Results_UC.csv")
    kept, dropped, meta, n_imputed = clean_and_encode(raw)

    G_resp, sim = build_respondent_network(kept)
    print("Respondent network:")
    print(f"  nodes={G_resp.number_of_nodes()} edges={G_resp.number_of_edges()}")
    print(f"  density={nx.density(G_resp):.3f}")
    print(f"  connected components={nx.number_connected_components(G_resp)}")

    G_stmt, corr = build_statement_network(kept, meta)
    print("\nStatement network:")
    print(f"  nodes={G_stmt.number_of_nodes()} edges={G_stmt.number_of_edges()}")
    print(f"  density={nx.density(G_stmt):.3f}")
