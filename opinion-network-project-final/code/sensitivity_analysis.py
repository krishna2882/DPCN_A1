"""
sensitivity_analysis.py
------------------------
Checks how stable the two networks' community structure is to the analytic
choices made in build_networks.py:
  - k (number of nearest neighbours) in the Respondent Opinion Network
  - the |correlation| threshold in the Statement (Topic) Network

Also reports, for comparison, what a strict MUTUAL (intersection) k-NN graph
would look like instead of the UNION construction actually used, since this
is a natural robustness question raised during review.
"""

import itertools

import pandas as pd
import networkx as nx
from data_prep import load_raw, clean_and_encode
from build_networks import respondent_similarity, build_statement_network
from analyze_networks import detect_communities

RESP_K_VALUES = [6, 7, 8, 9, 10]
STMT_THRESHOLDS = [0.30, 0.33, 0.35, 0.38, 0.40]


def adjusted_rand_index(labels_a, labels_b):
    """
    Agreement between two partitions of the same node set, corrected for
    chance (1.0 = identical partitions, 0.0 = no better than random).
    Implemented directly to avoid adding a scikit-learn dependency.
    """
    a = pd.Series(labels_a)
    b = pd.Series(labels_b).reindex(a.index)
    ct = pd.crosstab(a, b).values

    def c2(x):
        return x * (x - 1) / 2

    sum_ij = c2(ct).sum()
    sum_i = c2(ct.sum(axis=1)).sum()
    sum_j = c2(ct.sum(axis=0)).sum()
    n = c2(ct.sum())
    expected = sum_i * sum_j / n
    max_index = 0.5 * (sum_i + sum_j)
    return (sum_ij - expected) / (max_index - expected)


def build_union_knn(sim, k):
    G = nx.Graph()
    G.add_nodes_from(sim.index)
    for i in sim.index:
        row = sim.loc[i].drop(index=i)
        for j, w in row.sort_values(ascending=False).head(k).items():
            if w > 0:
                if G.has_edge(i, j):
                    G[i][j]["weight"] = max(G[i][j]["weight"], float(w))
                else:
                    G.add_edge(i, j, weight=float(w))
    return G


def build_mutual_knn(sim, k):
    """Edge only if each is in the other's top-k (intersection/mutual k-NN)."""
    topk = {i: set(sim.loc[i].drop(index=i).sort_values(ascending=False).head(k).index)
            for i in sim.index}
    G = nx.Graph()
    G.add_nodes_from(sim.index)
    for i in sim.index:
        for j in topk[i]:
            if i in topk[j] and sim.loc[i, j] > 0:
                G.add_edge(i, j, weight=float(sim.loc[i, j]))
    return G



def category_means_by_community(data, meta, membership):
    """Mean score per topic category for each community in a given run."""
    cat_of = meta.set_index("item_id")["category"]
    cats = ["Technology", "Education", "Ethics", "Environment"]
    out = {}
    memb = pd.Series(membership)
    for comm in sorted(set(membership.values())):
        members = memb[memb == comm].index
        sub = data.loc[members]
        out[comm] = {c: float(sub[[i for i in sub.columns if cat_of[i] == c]].values.mean())
                     for c in cats}
        out[comm]["n"] = len(members)
    return out


def match_to_baseline(base_memb, other_memb):
    """
    Greedily pair each baseline community with the community in the other run
    that shares the most respondents, so their category profiles can be
    compared like with like. Communities in the other run are used at most once.
    """
    base = pd.Series(base_memb)
    other = pd.Series(other_memb)
    shared = [n for n in other.index if n in base.index]
    pairs, used = {}, set()
    for b in sorted(set(base_memb.values())):
        b_members = set(base[base == b].index) & set(shared)
        best, best_overlap = None, -1
        for o in sorted(set(other_memb.values())):
            if o in used:
                continue
            overlap = len(b_members & set(other[other == o].index))
            if overlap > best_overlap:
                best, best_overlap = o, overlap
        if best is not None and best_overlap > 0:
            pairs[b] = best
            used.add(best)
    return pairs


def profile_stability(base_data, base_memb, variants, meta):
    """
    How far do the COMMUNITY PROFILES (Table 1 of the report) move when the
    analytic settings change? Section 3.2 argues that profiles are the stable
    part of the clustering even though memberships are not -- this measures
    that claim instead of asserting it.
    """
    cats = ["Technology", "Education", "Ethics", "Environment"]
    base_prof = category_means_by_community(base_data, meta, base_memb)
    rows = []
    for label, data, memb in variants:
        prof = category_means_by_community(data, meta, memb)
        pairs = match_to_baseline(base_memb, memb)
        drifts = [abs(prof[o][c] - base_prof[b][c]) for b, o in pairs.items() for c in cats]
        # qualitative claims from Sections 6.2 and 6.4
        # Environment highest or joint-highest (within 0.01) in every community
        env_top = all(pr["Environment"] >= max(pr[c] for c in cats) - 0.01
                      for pr in prof.values())
        all_positive = all(pr[c] > 3.5 for pr in prof.values() for c in cats)
        rows.append({
            "variant": label,
            "n_communities": len(prof),
            "matched": len(pairs),
            "max_profile_drift": round(max(drifts), 3) if drifts else None,
            "mean_profile_drift": round(sum(drifts) / len(drifts), 3) if drifts else None,
            "Env_top_or_tied_in_all": env_top,
            "all_categories_above_3.5": all_positive,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = load_raw("../data/Survey_Results_UC.csv")
    kept, dropped, meta, n_imputed = clean_and_encode(raw)
    sim = respondent_similarity(kept)

    print("=== Respondent network: sensitivity to k (union k-NN, as used) ===")
    rows = []
    for k in RESP_K_VALUES:
        G = build_union_knn(sim, k)
        memb, mod, n_comm = detect_communities(G)
        rows.append({"k": k, "edges": G.number_of_edges(),
                     "connected": nx.is_connected(G),
                     "n_communities": n_comm, "modularity": round(mod, 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Respondent network: union vs. mutual k-NN at k=8 (robustness check) ===")
    G_union = build_union_knn(sim, 8)
    G_mutual = build_mutual_knn(sim, 8)
    for name, G in [("union (used in report)", G_union), ("mutual (stricter alternative)", G_mutual)]:
        connected = nx.is_connected(G)
        memb, mod, n_comm = detect_communities(G) if G.number_of_edges() > 0 else (None, None, None)
        print(f"{name:32s} edges={G.number_of_edges():4d}  connected={connected!s:5s} "
              f"components={nx.number_connected_components(G):2d}  "
              f"communities={n_comm}  modularity={mod}")

    print("\n=== Louvain seed stability on the baseline graph (reference level) ===")
    print("Any ARI comparison below has to be read against how much the Louvain")
    print("heuristic itself moves when only the random seed changes.\n")
    G_base = build_union_knn(sim, 8)
    seed_parts = []
    for s in range(10):
        comms = nx.community.louvain_communities(G_base, weight="weight", seed=s)
        seed_parts.append({n: i for i, c in enumerate(comms) for n in c})
        print(f"  seed={s}  communities={len(comms)}  "
              f"modularity={nx.community.modularity(G_base, comms, weight='weight'):.4f}")
    pairwise = [adjusted_rand_index(a, b)
                for a, b in itertools.combinations(seed_parts, 2)]
    print(f"\n  mean pairwise ARI across seeds = {sum(pairwise)/len(pairwise):.3f} "
          f"(min {min(pairwise):.3f}, max {max(pairwise):.3f})")
    print("  -> the partition is only loosely identified; community COUNT (3-5, "
          "usually 4)\n     and the category-level profiles are the stable part, not "
          "individual memberships.")

    print("\n=== Imputation strategy: does filling the 37 sparse cells matter? ===")
    print("Baseline = median imputation (used for all reported results).")
    print("Variants: mode imputation, and a complete-case subset that keeps only")
    print("respondents who answered all 60 items (so nothing is imputed at all).\n")

    base_memb, base_mod, base_n = detect_communities(build_union_knn(sim, 8))

    variants = []
    # (a) mode imputation, same 85 respondents
    kept_mode, _, _, _ = clean_and_encode(raw, impute="mode")
    # (b) complete cases only: respondents with zero missing items
    kept_cc, _, _, _ = clean_and_encode(raw, impute="none", max_missing=0)

    for label, data in [("median (used in report)", kept),
                        ("mode imputation", kept_mode),
                        ("complete cases only", kept_cc)]:
        sim_v = respondent_similarity(data)
        G_v = build_union_knn(sim_v, 8)
        memb_v, mod_v, n_v = detect_communities(G_v)
        cat_of = meta.set_index("item_id")["category"]
        cat_means = {c: data[[i for i in data.columns if cat_of[i] == c]].values.mean()
                     for c in ["Technology", "Education", "Ethics", "Environment"]}
        shared = [n for n in memb_v if n in base_memb]
        ari = adjusted_rand_index({n: base_memb[n] for n in shared},
                                  {n: memb_v[n] for n in shared})
        variants.append({
            "variant": label,
            "n_respondents": data.shape[0],
            "edges": G_v.number_of_edges(),
            "n_communities": n_v,
            "modularity": round(mod_v, 4),
            "ARI_vs_baseline": round(ari, 4),
            **{c[:4]: round(float(cat_means[c]), 3) for c in
               ["Technology", "Education", "Ethics", "Environment"]},
        })
    print(pd.DataFrame(variants).to_string(index=False))

    print("\n=== Community PROFILE stability across settings ===")
    print("Memberships move (above); the report's substantive claims rest on the")
    print("category profiles in Table 1. Each variant's communities are matched to")
    print("the baseline's by maximum respondent overlap, then profiles compared.\n")
    prof_variants = []
    for k in RESP_K_VALUES:
        m_k, _, _ = detect_communities(build_union_knn(sim, k))
        prof_variants.append((f"k = {k}" + (" (baseline)" if k == 8 else ""), kept, m_k))
    for label, data in [("mode imputation", kept_mode), ("complete cases only", kept_cc)]:
        sim_v = respondent_similarity(data)
        m_v, _, _ = detect_communities(build_union_knn(sim_v, 8))
        prof_variants.append((label, data, m_v))
    for s_ in range(1, 4):
        comms = nx.community.louvain_communities(G_base, weight="weight", seed=s_)
        prof_variants.append((f"Louvain seed {s_}", kept,
                              {n: i for i, c in enumerate(comms) for n in c}))
    print(profile_stability(kept, base_memb, prof_variants, meta).to_string(index=False))

    print("\n=== Statement network: sensitivity to correlation threshold ===")
    rows = []
    for t in STMT_THRESHOLDS:
        G, corr = build_statement_network(kept, meta, threshold=t)
        memb, mod, n_comm = detect_communities(G)
        rows.append({"threshold": t, "edges": G.number_of_edges(),
                     "components": nx.number_connected_components(G),
                     "n_communities": n_comm, "modularity": round(mod, 4)})
    print(pd.DataFrame(rows).to_string(index=False))
