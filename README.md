# OMEGA-D-26-00470

Code and data supporting the manuscript *A regional multi-criteria assessment of European education and training systems*, submitted to Omega - The International Journal of Management Science.

## Contents

- `data.xlsx` — criteria values ($g_1$ to $g_7$) for the 200 NUTS regions analysed, one sheet per year (2020-2024).
- `framework.py` — full implementation of the framework: ELECTRE TRI-B credibilities, sorting and ranking based on linear programming, SMAA-TRI category acceptability indices, ranking based on the aggregation of the acceptability indices, and the writing of all result files.
- `results/` — all results, organised by cutting-level interval (`lambda_base` for [4/7, 5/7], `lambda_wide` for ]1/2, 3/4]) and percentile configuration (e.g. `15_30_50_70_85`). Each folder contains the sortings, the SMAA acceptability indices and aggregations, the rankings, and the lexicographic analyses, per scenario and year.

## Reproducing the results

```
pip install -r requirements.txt
python framework.py
```

This regenerates the complete `results/` folder. The run takes roughly 20-30 minutes, most of it in the 50,000-draw SMAA-TRI simulations. The weights and the cutting level are sampled with fixed seeds, so all results are exactly reproducible under the package versions listed in `requirements.txt`.
