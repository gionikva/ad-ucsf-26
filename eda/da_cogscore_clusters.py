from argparse import ArgumentParser
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from collections import defaultdict
import matplotlib
matplotlib.use("Qt5Agg")   # needs: pip install PyQt5   (or "QtAgg" with PyQt6/PySide6)
import matplotlib.pyplot as plt

ID_COLS = ["PTID", "VISDATE"]

def prep(df, id_cols=ID_COLS):
    feat_cols = [c for c in df.columns
                 if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=feat_cols).reset_index(drop=True)   # positions 0..N-1 now line up with Xs
    F = df[feat_cols]
    std = F.std(ddof=0).replace(0, 1.0)                       # avoid divide-by-zero on constant columns
    Xs = ((F - F.mean()) / std).to_numpy()
    return df, feat_cols, Xs


def _expand(value, n, name):
    """Scalar -> list of n copies; list/tuple/array -> validated list of length n."""
    if np.isscalar(value):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} needs {n} entries (one per component) or a single scalar, got {len(value)}")
    return value

def gmm_subsets(df, id_cols=ID_COLS, min_size=5, min_spread=0.0,
                k_range=range(1, 11), cov_type="full", tol=0.01,
                min_frac=0.0, min_prob=0.0, jitter=0.0,
                reg_covar=1e-3, n_init=3, seed=0):
    id_cols = list(id_cols)
    df, feat_cols, Xs = prep(df, id_cols)
    min_size = _expand(min_size, len(feat_cols), "min_size")
    min_spread = _expand(min_spread, len(feat_cols), "min_spread")
    rng = np.random.default_rng(seed)

    out = []
    for i, comp in enumerate(feat_cols):
        rest = np.delete(Xs, i, axis=1)
        if jitter > 0:                                   # soften the integer lattice for fitting only
            rest = rest + rng.normal(0, jitter, rest.shape)
        ms, sp = min_size[i], min_spread[i]

        # Fit every k, then take the smallest k whose BIC is within tol of the best
        fits = []
        for k in k_range:
            if k * ms > len(rest):
                break
            gm = GaussianMixture(k, covariance_type=cov_type, reg_covar=reg_covar,
                                 n_init=n_init, random_state=seed).fit(rest)
            fits.append((gm.bic(rest), gm))
        if not fits:
            continue
        bics = np.array([b for b, _ in fits])
        best_bic = bics.min()
        ok = np.where(bics - best_bic <= tol * abs(best_bic))[0]
        best = fits[ok[0]][1]                            # smallest acceptable k

        lab = best.predict(rest)
        if min_prob > 0:
            lab = np.where(best.predict_proba(rest).max(1) >= min_prob, lab, -1)

        for c in sorted(set(lab) - {-1}):
            idx = np.where(lab == c)[0]
            if len(idx) < ms or len(idx) < min_frac * len(rest):
                continue
            spread = Xs[idx, i].std()
            if spread < sp:
                continue
            scatter = np.sqrt(((rest[idx] - rest[idx].mean(0))**2).sum(1).mean())
            sub = df.loc[idx, id_cols + feat_cols].copy()
            sub["component"] = comp
            sub["subset"] = int(c)
            sub["spread"] = spread
            sub["rest_scatter"] = scatter
            sub["score"] = spread / (scatter + 1e-9)
            out.append(sub)

    cols = id_cols + feat_cols + ["component", "subset", "spread", "rest_scatter", "score"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

    cols = id_cols + feat_cols + ["component", "subset", "spread", "rest_scatter", "score"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

def dbscan_subsets(df, id_cols=ID_COLS, tau=0.3, min_size=5, min_spread=0.0):
    id_cols = list(id_cols)
    df, feat_cols, Xs = prep(df, id_cols)
    out = []
    
    if type(tau) == list:
        taus = tau
    else:
        taus = [tau] * len(feat_cols)
        
    if type(min_size) == list:
        min_sizes = min_size
    else:
        min_sizes = [min_size] * len(feat_cols)
        
    if type(min_spread) == list:
        min_spreads = min_spread
    else:
        min_spreads = [min_spread] * len(feat_cols)
        
    for i, comp in enumerate(feat_cols):
        tau = taus[i]
        min_size = min_sizes[i]
        min_spread = min_spreads[i]
        
        rest = np.delete(Xs, i, axis=1)
        lab = DBSCAN(eps=tau, min_samples=min_size).fit_predict(rest)
        for c in sorted(set(lab) - {-1}):                  # -1 is DBSCAN noise
            idx = np.where(lab == c)[0]
            spread = Xs[idx, i].std()
            if spread < min_spread:
                continue
            scatter = np.sqrt(((rest[idx] - rest[idx].mean(0))**2).sum(1).mean())
            sub = df.loc[idx, id_cols + feat_cols].copy()  # ids AND cog scores
            sub["component"] = comp
            sub["subset"] = int(c)
            sub["spread"] = spread
            sub["rest_scatter"] = scatter
            sub["score"] = spread / (scatter + 1e-9)
            out.append(sub)
    cols = id_cols + feat_cols + ["component", "subset", "spread", "rest_scatter", "score"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def hdbscan_subsets(df, id_cols=ID_COLS, min_cluster_size=10, min_size=5,
                    min_spread=0.0, cluster_selection_method="leaf"):
    id_cols = list(id_cols)
    df, feat_cols, Xs = prep(df, id_cols)
    out = []
        
    if type(min_size) == list:
        min_sizes = min_size
    else:
        min_sizes = [min_size] * len(feat_cols)
        
    if type(min_spread) == list:
        min_spreads = min_spread
    else:
        min_spreads = [min_spread] * len(feat_cols)
        
    for i, comp in enumerate(feat_cols):
        min_size = min_sizes[i]
        min_spread = min_spreads[i]
        
        rest = np.delete(Xs, i, axis=1)
        lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_size,
                      cluster_selection_method=cluster_selection_method).fit_predict(rest)
        for c in sorted(set(lab) - {-1}):              # -1 is noise
            idx = np.where(lab == c)[0]
            spread = Xs[idx, i].std()
            if spread < min_spread:
                continue
            scatter = np.sqrt(((rest[idx] - rest[idx].mean(0))**2).sum(1).mean())
            sub = df.loc[idx, id_cols + feat_cols].copy()
            sub["component"] = comp
            sub["subset"] = int(c)
            sub["spread"] = spread
            sub["rest_scatter"] = scatter
            sub["score"] = spread / (scatter + 1e-9)
            out.append(sub)
    cols = id_cols + feat_cols + ["component", "subset", "spread", "rest_scatter", "score"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

def x_mean_y(df: pd.DataFrame, comp, feat_cols):
    sims = list(set(feat_cols) - set(comp))
    # means = df.groupby(["component", "subset"])[feat_cols].mean()
    df = df[df["component"] == comp]
 
    return df

def plot(df: pd.DataFrame, feat_cols: list):
    ncols = 4
    nrows = int(np.ceil(len(feat_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    cmap = plt.get_cmap("tab20")

    for ax, col in zip(axes.flat, feat_cols):
        filtered = df[df["component"] == col]
        if filtered.empty:
            ax.set_visible(False)
            continue

        others = [c for c in feat_cols if c != col]       # exclude the target from x
        x = filtered[others].mean(axis=1).to_numpy()
        y = filtered[col].to_numpy()
        labels = filtered["subset"].to_numpy()

        fits = []   # (color, slope, intercept)
        for j, s in enumerate(np.unique(labels)):
            m = labels == s
            color = cmap(j % 20)
            ax.scatter(x[m], y[m], s=20, alpha=0.7, color=color, label=f"subset {s}")

            xs, ys = x[m], y[m]
            ok = ~(np.isnan(xs) | np.isnan(ys))
            xs, ys = xs[ok], ys[ok]
            if len(xs) >= 2 and np.ptp(xs) > 0:
                slope, intercept = np.polyfit(xs, ys, 1)
                fits.append((color, slope, intercept))

        # Freeze the view so the lines don't change the axis limits
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_autoscale_on(False)

        xl = np.array(xlim)
        for color, slope, intercept in fits:
            ax.plot(xl, slope * xl + intercept, color=color, linewidth=1.8)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)      # steep lines get clipped instead of stretching the y-axis

        ax.set_xlabel("mean of other components")
        ax.set_ylabel(col)
        ax.set_title(col)
        if len(np.unique(labels)) <= 10:
            ax.legend(fontsize=7)

    for ax in axes.flat[len(feat_cols):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig("./eda/clusters.png", dpi=150)   # save before show()
    plt.show()
   
def main():
    parser = ArgumentParser()
    # parser.add_argument(
    #     "-m",
    #     "--method",
    #     help='Clustering method: "dbscan" or "graph".',
    #     choices=["dbscan", "graph"],
    #     required=True,
    # )
    parser.add_argument(
        "-i",
        "--input-table",
        help="Location of the ADAS cog scores ADNI table.",
        default="./data/tables/ADAS.csv",
    )

    args = parser.parse_args()

    # method = args.method
    table_path = args.input_table

    feat_cols = [    
        "Q1SCORE",
        "Q2SCORE",
        "Q3SCORE",
        "Q4SCORE",
        "Q5SCORE",
        "Q6SCORE",
        "Q7SCORE",
        "Q8SCORE",
        "Q9SCORE",
        "Q10SCORE",
        "Q11SCORE",
        "Q12SCORE",
        "Q13SCORE",
    ]

    cogscores = pd.read_csv(table_path, usecols=[
        "PTID",
        "VISDATE",
        *feat_cols
    ],
    na_values=[-4])
    
    tau = 1
    
    taus =        [1.3, 1.8, 1.0, 1.4, 2.2, 1.5, 1.0, 1.0, 1.5, 2.0, 1.3, 1.5, 1.0]
    
    min_spreads = [0.9, 0.8, 1.0, 1.0, 0.8, 0.8, 1.5, 1.0, 1.0, 0.3, 1.5, 0.8, 1.1]

    # df = dbscan_subsets(cogscores, id_cols=["PTID", "VISDATE"], tau=taus, min_spread=min_spreads)
    df = gmm_subsets(cogscores, id_cols=["PTID", "VISDATE"],
                 k_range=range(1, 11),      # include 1
                 tol=0.01, min_frac=0.02, jitter=0.1,
                 min_size=10, min_spread=0.5)
    # df = gmm_subsets(cogscores, id_cols=["PTID", "VISDATE"])
    plot(df, feat_cols)
    
    
    
    
if __name__ == "__main__":
    main()