# Opinion Network Formation — Assignment 1

Constructs and analyzes an opinion network from a class survey covering
Technology, Education, Ethics, and Environment (60 five-point Likert items,
96 raw responses).

## Repository structure

```
data/       Survey_Results_UC.csv (raw survey export)
code/       data_prep.py            cleaning + Likert encoding
            build_networks.py       respondent similarity network + statement
                                     correlation network
            analyze_networks.py     centrality, Louvain communities, opinion
                                     profiling, consensus/polarization stats
            visualize.py            all report figures
            sensitivity_analysis.py robustness checks on k, threshold, the
                                     union-vs-mutual k-NN choice, imputation,
                                     Louvain seeds, and community-profile drift
            validation.py           split-half replication of the statement
                                     clusters, PCA on the item matrix, and a
                                     test of whether centrality tracks
                                     proximity to the cohort's mean opinion
            main.py                 runs the full pipeline end-to-end
figures/    generated PNG figures
output/     generated CSV/JSON result tables
report/     report.tex  (LaTeX source — edit this)
            report.html (HTML source, alternative)
            Assignment1_Opinion_Network_Report.pdf (final submission)
```

Generated tables in `output/`: `summary.json` (all headline network metrics),
`respondent_centrality.csv`, `respondent_communities.csv`,
`community_profiles.csv`, `community_item_deltas.csv` and
`community_distinctive_items.csv` (the per-item deviations quoted in report
Section 6.4), `statement_extremes.csv`, `statement_clusters.csv` (membership
and T/E/S/V composition of every statement cluster), and
`sensitivity_analysis.txt` and `validation.txt`.

## Reproducing the analysis

```bash
cd code
pip install pandas numpy networkx matplotlib
python main.py                  # full pipeline: cleaning -> networks -> figures/tables
python sensitivity_analysis.py  # robustness checks (see report Section 6.6)
```

This regenerates every table in `output/` and every figure in `figures/`
used in the final report.

## Method summary

1. **Cleaning:** respondents with more than 4 of 60 items missing/unanswered
   were dropped (96 → 85 respondents; 5 fully blank + 6 partial completions).
   Remaining sparse missingness (37 cells, 0.73% of the retained matrix) was
   imputed with the item's column median.
2. **Respondent Opinion Network:** nodes are respondents; edges connect a
   respondent to their 8 nearest neighbours by Pearson correlation across
   all 60 items — a k-NN graph **symmetrized by union** (edge (i,j) if i is
   in j's top-8 OR j is in i's top-8; not necessarily both, so this is not
   a *mutual* k-NN graph). The union construction was chosen because a
   strict mutual/intersection k-NN graph at k=8 fragments into 17
   disconnected components on this data (see `sensitivity_analysis.py`).
   Pearson correlation was chosen over cosine similarity/raw distance
   because it captures similarity in the *pattern* of relative agreement
   across topics, factoring out individual differences in overall
   agreeableness.
3. **Statement (Topic) Network:** nodes are the 60 survey statements; edges
   connect statement pairs with |Pearson correlation| ≥ 0.35 across
   respondents (168 positive edges, 1 negative).
4. **Analysis:** weighted degree, eigenvector centrality, and betweenness
   centrality (computed on a distance-transformed copy of the graph,
   d = 1 − correlation, since networkx treats edge `weight` as a distance
   for shortest-path routing — using raw similarity directly would invert
   the intended meaning), plus Louvain community detection (a modularity
   heuristic, not a global optimum) on both networks, and category-level
   opinion profiling per community.
5. **Sensitivity checks** (`sensitivity_analysis.py`, output in
   `output/sensitivity_analysis.txt`):
   - *k in the respondent network:* stable — 4 communities and a single
     connected component for k = 6–10, modularity 0.25–0.33.
   - *union vs. mutual k-NN at k = 8:* 549 edges / 1 component vs.
     131 edges / 17 components — the reason the union rule is used.
   - *Louvain seed stability:* ten seeds on the unchanged baseline graph give
     3–5 communities and a mean pairwise adjusted Rand index of 0.60, so
     individual memberships are only loosely identified. Report Sections 5.1
     and 6.4 therefore rely on community profiles, not on who is in which
     community.
   - *Imputation strategy:* median (reported), mode, and a complete-case
     subset with no imputation at all. Category means move by at most 0.05 of
     a Likert point; partition ARI vs. baseline is 0.47 (mode) and 0.14
     (complete cases), i.e. within the algorithm's own seed noise above.
   - *Community profiles:* matching each variant's communities to the
     baseline by maximum respondent overlap, category means drift by 0.12 on
     average and 0.56 at worst. Profiles are stable even though memberships
     are not — which is why the report's claims are stated at profile level.
   - *Statement-network threshold:* genuinely sensitive — 10 communities at
     r = 0.30 rising to 22 at r = 0.40.
6. **Validation checks** (`validation.py`, output in `output/validation.txt`):
   - *Split-half replication:* 50 random splits × 2 halves. Specific cluster
     membership does not replicate (mean Jaccard 0.30–0.40), but 97.5% of
     half-sample clusters remain cross-category, and within-cluster item
     correlation (mean |r| = 0.303) beats within-category (0.228) and the
     all-pairs baseline (0.176).
   - *PCA:* PC1 takes 19.8% of variance with 95% of loadings sharing a sign —
     a general agreement factor, not a topical one. Flat scree after it, so no
     factor structure is identifiable at n = 85.
   - *Centrality meaning:* weighted degree correlates with correlation-to-
     cohort-centroid at r = 0.83, so reading high degree as "mainstream
     opinion profile" is supported rather than assumed.

## Team

Team name: ConsistentProcess

| Name | Roll Number |
| --- | --- |
| Vedant Pahariya | 2023112012 |
| Siddhant Gudwani | 2024102042 |
| Krishna Goel | 2023112009 |

### Individual contribution

Mirrors Section 7 of the report.

**Vedant Pahariya — data pipeline and Respondent Opinion Network (33%)**
Survey parsing and Likert encoding, missingness profiling and the completeness
cutoff, median imputation (`data_prep.py`); respondent similarity matrix and the
union k-NN construction, including the union-vs-mutual comparison that justified
it (`build_networks.py`); network-level metrics — density, components, path
length, clustering. Report Sections 3 and 4 (Steps 1–2), 5.1.

**Siddhant Gudwani — Statement Network, centrality and community analysis (34%)**
Item–item correlation network and threshold selection (`build_networks.py`);
weighted degree, betweenness on the distance-transformed graph, eigenvector
centrality and clustering; Louvain community detection on both networks;
community opinion profiling, per-item community deviations, statement cluster
tables and the consensus/polarization ranking (`analyze_networks.py`). Report
Sections 4 (Steps 3–4), 5.2–5.4, 6.1–6.5.

**Krishna Goel — robustness, validation and visualization (33%)**
Sensitivity analysis across k, correlation threshold, imputation strategy,
Louvain seeds and community-profile drift (`sensitivity_analysis.py`);
split-half cluster replication, within-group correlation comparison, PCA and the
centrality-vs-centroid test (`validation.py`); all six figures (`visualize.py`);
pipeline orchestration and result export (`main.py`); repository documentation.
Report Sections 4 (Step 5), 5.5, 6.6.

Each member reviewed the sections written by the other two; the report text was
drafted jointly against the outputs listed above.

## Building the report

The report exists in two interchangeable sources — edit whichever you prefer.

**LaTeX (recommended for editing).** Team members, roll numbers and the Section 7
contribution table are already filled in. Two placeholders remain in the
`\newcommand` lines at the top of `report/report.tex` — the team name and the
GitHub URL. Fill those, then:

```bash
cd report && pdflatex report.tex && pdflatex report.tex   # run twice
```

Unfilled placeholders are wrapped in `\todo{...}` and print in red, so nothing
gets submitted blank. Once everything is filled in, redefine it as
`\newcommand{\todo}[1]{#1}` to print in normal black.

**HTML (alternative).** Edit `report/report.html`, then:

```bash
cd report && wkhtmltopdf --enable-local-file-access --page-size A4 \
  --margin-top 16mm --margin-bottom 14mm --margin-left 15mm --margin-right 15mm \
  --footer-center "[page]" --footer-font-size 8 \
  report.html Assignment1_Opinion_Network_Report.pdf
```

Both pull figures from `figures/`, so run `cd code && python main.py` first if
you change the analysis.
