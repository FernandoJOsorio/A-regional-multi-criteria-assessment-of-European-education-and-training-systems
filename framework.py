####################
###   1. Setup   ###
####################

import hopsy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parent

criteria_all = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7']
criteria_veto = ['g1', 'g2', 'g3', 'g4']
criteria_min = ['g1', 'g2']

n_draws = 50_000
weight_seed = 0
lambda_seed = 1
weight_floor = 0.085
thinning = (len(criteria_all) - 1) ** 3

years_run = (2020, 2024)
results_dir = repo_root / 'results'

lambdas = {
    'base': (4 / 7, 5 / 7),
    'wide': (np.nextafter(0.5, 1), 3 / 4),
}

perc = {
    '15_30_50_70_85': [0, 0.15, 0.30, 0.50, 0.70, 0.85, 1],
    '20_30_50_70_80': [0, 0.20, 0.30, 0.50, 0.70, 0.80, 1],
    '20_35_50_65_80': [0, 0.20, 0.35, 0.50, 0.65, 0.80, 1],
}

scenarios = {
    'I':   {'veto': False, 'partial_order': False},
    'II':  {'veto': True,  'partial_order': False},
    'III': {'veto': False, 'partial_order': True},
    'IV':  {'veto': True,  'partial_order': True},
}

partial_order_pairs = [(3, 1), (3, 2), (3, 4),
                       (1, 5), (1, 6), (2, 5), (2, 6), (4, 5), (4, 6),
                       (5, 7), (6, 7)]


########################
###   2. Load data   ###
########################


data_path = repo_root / 'data.xlsx'
years = {'2020': 2020, '2021': 2021, '2022': 2022, '2023': 2023, '2024': 2024}

def load_data(path):
    sheets = pd.read_excel(path, sheet_name=list(years), index_col=0)
    data = {years[name]: df[criteria_all] for name, df in sheets.items()}
    data_pooled = pd.concat(data.values())
    return data, data_pooled

data, data_pooled = load_data(data_path)


############################################################
###   3. Category boundaries, thresholds and scenarios   ###
############################################################


def get_boundaries(percentiles):
    n_bounds = len(percentiles)
    boundaries = pd.DataFrame(
        {g: np.quantile(data_pooled[g], percentiles) for g in criteria_all},
        index=[f'b{i}' for i in range(n_bounds)],
    )
    for g in criteria_min:
        boundaries[g] = boundaries[g].values[::-1]
    for g in criteria_all:
        low, high = (100, 0) if g in criteria_min else (0, 100)
        boundaries.loc['b0', g] = low
        boundaries.loc[f'b{n_bounds - 1}', g] = high
    return boundaries.round(1)

def get_thresholds(boundaries, scenario):
    last_inner = f'b{len(boundaries) - 2}'
    n_middle = len(boundaries) - 3
    spread = (boundaries.loc[last_inner] - boundaries.loc['b1']).abs()
    q = pd.Series(0.0, index=criteria_all)
    p = (spread / n_middle).round(1)
    v = pd.Series({g: spread[g] if scenario['veto'] and g in criteria_veto else np.inf
                   for g in criteria_all}).round(1)
    return q, p, v

def get_constraints(scenario):
    n = len(criteria_all)
    A = [-row for row in np.eye(n)]
    b = [-weight_floor] * n
    if scenario['partial_order']:
        for i, j in partial_order_pairs:
            row = np.zeros(n); row[j - 1] = 1.0; row[i - 1] = -1.0
            A.append(row); b.append(0.0)
    return np.array(A, float), np.array(b, float)

def get_config(percentiles, scenario, lam):
    boundaries = get_boundaries(percentiles)
    q, p, v = (ser[criteria_all].to_numpy(float) for ser in get_thresholds(boundaries, scenario))
    A_ub, b_ub = get_constraints(scenario)
    return SimpleNamespace(boundaries=boundaries, q=q, p=p, v=v, A_ub=A_ub, b_ub=b_ub,
                           lambda_lo=lam[0], lambda_hi=lam[1])


####################################
###   4. ELECTRE-TRI functions   ###
####################################


def delta(region, boundary):
    sign = np.where(np.isin(criteria_all, criteria_min), -1.0, 1.0)
    return sign * (region - boundary)

def c(config, region, boundary):
    return np.clip((config.p + delta(region, boundary)) / (config.p - config.q), 0, 1)

def d(config, region, boundary):
    return np.clip((-delta(region, boundary) - config.p) / (config.v - config.p), 0, 1)

def d_max(config, region, boundary):
    return d(config, region, boundary).max(axis=-1)

def c_global(config, region, boundary, weights):
    return c(config, region, boundary) @ weights

def s(config, region, boundary, weights):
    return c_global(config, region, boundary, weights) * (1 - d_max(config, region, boundary))


##############################################
###   5. Sorting from linear programming   ###
##############################################


def opt_s(config, region, boundary, maximise):
    sense = -1.0 if maximise else 1.0
    res = linprog(sense * c(config, region, boundary), A_ub=config.A_ub, b_ub=config.b_ub,
                  A_eq=np.ones((1, len(criteria_all))), b_eq=[1.0], method='highs')
    return s(config, region, boundary, res.x)

def get_credibilities(config, df):
    boundaries = config.boundaries[criteria_all].to_numpy(float)
    n_bounds = boundaries.shape[0]
    s_min = np.empty((len(df), n_bounds)); s_max = np.empty((len(df), n_bounds))
    s_min[:, 0] = s_max[:, 0] = 1.0
    s_min[:, -1] = s_max[:, -1] = 0.0
    for ai, (_, reg) in enumerate(df[criteria_all].iterrows()):
        r = reg.to_numpy(float)
        for k in range(1, n_bounds - 1):
            s_min[ai, k] = opt_s(config, r, boundaries[k], maximise=False)
            s_max[ai, k] = opt_s(config, r, boundaries[k], maximise=True)
    cols = [f'b{i}' for i in range(n_bounds)]
    s_min, s_max = np.round(s_min, 3), np.round(s_max, 3)
    return pd.DataFrame(s_min, index=df.index, columns=cols), pd.DataFrame(s_max, index=df.index, columns=cols)

def sorting_lp(config, s_min, s_max):
    W = (s_min.to_numpy() >= config.lambda_hi).sum(axis=1)
    B = (s_max.to_numpy() >= config.lambda_lo).sum(axis=1)
    return pd.DataFrame({'W': W, 'B': B}, index=s_min.index)


##################################################################################
###   6. Pessimistic and Optimistic rankings based on lexicographic analyses   ###
##################################################################################


def ranking_lexic(config, s_min, s_max):
    W = (s_min.to_numpy() >= config.lambda_hi).sum(axis=1)
    B = (s_max.to_numpy() >= config.lambda_lo).sum(axis=1)
    n = s_min.shape[1]
    cols = [f'b{i}' for i in range(n)]
    rankings = []
    for optimistic, bounds in ((False, s_min), (True, s_max)):
        vals = bounds.to_numpy().copy()
        for ai in range(len(vals)):
            keep = np.zeros(n, bool)
            keep[W[ai] - 1: B[ai] + 1] = True
            vals[ai, ~keep] = np.nan
        show = cols[::-1] if optimistic else cols
        first = {'B': B, 'W': W} if optimistic else {'W': W, 'B': B}
        out = pd.DataFrame(first, index=s_min.index).join(
            pd.DataFrame(vals, index=s_min.index, columns=cols)[show])
        keys = (['B', 'W'] + cols[::-1]) if optimistic else (['W', 'B'] + cols)
        out = out.sort_values(keys, ascending=False, na_position='last')
        pos = pd.Series(np.arange(1, len(out) + 1), index=out.index)
        tie_key = out.fillna(-1.0)
        group = tie_key.ne(tie_key.shift()).any(axis=1).cumsum()
        lo = pos.groupby(group).transform('min'); hi = pos.groupby(group).transform('max')
        out['rank'] = np.where(lo == hi, lo.astype(str), lo.astype(str) + '-' + hi.astype(str))
        rankings.append(out.fillna(''))
    return rankings[0], rankings[1]


#####################################################
###   7. Sampling of weights and cutting levels   ###
#####################################################


def weight_polytope(config):
    prob = hopsy.Problem(config.A_ub, config.b_ub)
    return hopsy.add_equality_constraints(prob, np.ones((1, len(criteria_all))), [1.0])

def sample_weights(n_iter, config, seed=weight_seed, thinning=thinning):
    prob = weight_polytope(config)
    chain = hopsy.MarkovChain(prob, hopsy.UniformHitAndRunProposal,
                              starting_point=hopsy.compute_chebyshev_center(prob))
    _, states = hopsy.sample(chain, hopsy.RandomNumberGenerator(seed=seed),
                             n_samples=n_iter, thinning=thinning)
    return np.array(states).reshape(-1, len(criteria_all))

def sample_lambda(n_iter, config, seed=lambda_seed):
    return np.random.default_rng(seed).uniform(config.lambda_lo, config.lambda_hi, n_iter)


########################################################
###   8. Sorting from SMAA-TRI stochastic analysis   ###
########################################################


def sorting_smaa(config, df):
    weight_samples = sample_weights(n_draws, config)
    lambda_samples = sample_lambda(n_draws, config)

    G = df[criteria_all].to_numpy(float)
    bounds = config.boundaries[criteria_all].to_numpy(float)
    n_categories = bounds.shape[0] - 1

    n_regions = G.shape[0]
    counts = np.zeros((n_regions, n_categories))
    for w, lam in zip(weight_samples, lambda_samples):
        cred = np.round(s(config, G[:, None, :], bounds[None, :, :], w), 3)
        cat = (cred[:, :n_categories] >= lam).sum(axis=1)
        counts[np.arange(n_regions), cat - 1] += 1

    acc = np.round(counts / n_draws, 3)
    cats = np.arange(1, n_categories + 1)
    lo = [int(cats[row > 0].min()) for row in acc]
    hi = [int(cats[row > 0].max()) for row in acc]
    return pd.DataFrame({'W': lo, 'B': hi}, index=df.index).join(
        pd.DataFrame(acc, index=df.index, columns=[f'C{i}' for i in range(1, n_categories + 1)]))


#####################################################################
###   9. Ranking from aggregating category acceptability indices  ###
#####################################################################


def metaweights(m):
    r = np.arange(1, m + 1)
    centroid = np.array([(1 / np.arange(i, m + 1)).sum() for i in r]) / (1 / r).sum()
    linear = (m - r) / (m - 1)
    inverse = 1 / r
    return pd.DataFrame({'Centroid': centroid, 'Linear': linear, 'Inverse': inverse}, index=r)

def ranking_smaa(sorting):
    acc = sorting.filter(regex=r'^C\d')
    alpha = metaweights(acc.shape[1])
    b = acc.to_numpy()[:, ::-1]
    out = pd.DataFrame(index=acc.index)
    for scheme in alpha.columns:
        score = pd.Series(b @ alpha[scheme].to_numpy(), index=acc.index).round(3)
        lo = score.rank(ascending=False, method='min').astype(int)
        hi = score.rank(ascending=False, method='max').astype(int)
        out[scheme] = score
        out[f'rank_{scheme}'] = np.where(lo == hi, lo.astype(str),
                                         lo.astype(str) + '-' + hi.astype(str))
    return out.sort_values('Centroid', ascending=False)


##########################################
###   10. Running and saving results   ###
##########################################


def run_config(config, df):
    s_min, s_max = get_credibilities(config, df)
    sorting_res = sorting_lp(config, s_min, s_max)
    sm = sorting_smaa(config, df).rename(columns={'W': 'W^SMAA', 'B': 'B^SMAA'})
    pess, opt = ranking_lexic(config, s_min, s_max)
    return {
        'sort': pd.concat([sorting_res.rename(columns={'W': 'W^LP', 'B': 'B^LP'}), sm], axis=1),
        'rank_lp_pess': pess,
        'rank_lp_opt': opt,
        'rank_smaa': ranking_smaa(sm),
    }

def run_full_grid():
    grid = {}
    for lam_name, lam in lambdas.items():
        grid[lam_name] = {}
        for perc_name in perc:
            grid[lam_name][perc_name] = {sc_name: {} for sc_name in scenarios}
            for sc_name, sc in scenarios.items():
                config = get_config(perc[perc_name], sc, lam)
                for year in years_run:
                    grid[lam_name][perc_name][sc_name][year] = run_config(config, data[year])
                print(f'{lam_name} / {perc_name} / scenario_{sc_name}: done')
    return grid

def acceptabilities_view(res):
    return res['sort'].filter(regex=r'^C\d').replace(0, np.nan)

def sorting_view(res):
    return res['sort'][['W^LP', 'B^LP', 'W^SMAA', 'B^SMAA']]

def rankings_view(res):
    ranks = pd.concat([res['rank_lp_pess']['rank'].rename('Pessimistic'),
                       res['rank_lp_opt']['rank'].rename('Optimistic')], axis=1)
    return ranks.join(res['rank_smaa'][['rank_Centroid', 'rank_Linear', 'rank_Inverse']]
                      .rename(columns=lambda col: col.removeprefix('rank_')))

def scores_view(res):
    return res['rank_smaa'][['Centroid', 'Linear', 'Inverse']]

views = {'smaa_acceptabilities': acceptabilities_view, 'all_sortings': sorting_view,
         'all_rankings': rankings_view, 'smaa_aggregations': scores_view}

def save_results(grid, outdir=results_dir):
    outdir = Path(outdir)
    for lam_name, percs in grid.items():
        for perc_name, scens in percs.items():
            perc_dir = outdir / f'lambda_{lam_name}' / perc_name
            perc_dir.mkdir(parents=True, exist_ok=True)
            ref = next(iter(scens.values()))
            for fname, view in views.items():
                with pd.ExcelWriter(perc_dir / f'{fname}.xlsx') as writer:
                    for year in ref:
                        regions = ref[year]['sort'].index
                        sheet = pd.concat({sc_name: view(scens[sc_name][year]).reindex(regions)
                                           for sc_name in scens}, axis=1)
                        sheet.to_excel(writer, sheet_name=str(year))
            lex_names = {'W': 'W(.)', 'B': 'B(.)', 'rank': 'Ranking'}
            for kind, key in (('pessimistic', 'rank_lp_pess'), ('optimistic', 'rank_lp_opt')):
                with pd.ExcelWriter(perc_dir / f'lp_lexicographic_{kind}.xlsx') as writer:
                    for sc_name, yrs in scens.items():
                        for year, res in yrs.items():
                            res[key].rename(columns=lex_names).to_excel(
                                writer, sheet_name=f'scenario_{sc_name}_{year}')
    print(f'Saved to {outdir.resolve()}')

if __name__ == '__main__':
    grid = run_full_grid()
    save_results(grid)
