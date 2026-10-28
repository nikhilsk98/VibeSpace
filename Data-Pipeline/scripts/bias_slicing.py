"""Bias detection with data slicing, plus mitigation, using Fairlearn.

MovieLens 25M and the VibeSpace app hold no user demographics, so slices are
built from movie attributes that could lead to unequal treatment of content:
``language_group``, ``era``, ``primary_genre`` and ``popularity_bucket``.

1. Representation: movie counts and Tag Genome label coverage per slice.
2. Label distribution: mean Movie DNA score per slice.
3. Model behaviour: a *diagnostic* Ridge regression (plot embedding + metadata
   -> 12 DNA scores, 70/15/15 split by movie) is trained only to measure
   per-slice error with Fairlearn ``MetricFrame``. It is not the Movie DNA model.
4. Mitigation: inverse-frequency sample weights and slice re-sampling; per-slice
   MAE is measured again and the trade-off in overall MAE is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from scripts.build_dna_labels import DNA_DIMENSIONS, DNA_LABELS_FILE
from scripts.clean import CLEAN_FILE
from scripts.config import PipelineConfig
from scripts.encode_plots import EMBEDDINGS_FILE, EMBEDDINGS_INDEX_FILE
from scripts.exceptions import PipelineError
from scripts.io_utils import read_parquet, write_json, write_text
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

SLICE_COLUMNS = ("language_group", "era", "primary_genre", "popularity_bucket")
BIAS_METRICS_FILE = "bias_metrics.json"
BIAS_REPORT_FILE = "bias_report.md"
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0)
MIN_SLICE_TEST = 15         # slices with fewer test movies are reported but not judged
DISPARITY_RATIO = 1.10      # slice MAE > 110% of overall MAE counts as a significant gap
WEIGHT_CLIP = (0.2, 10.0)


@dataclass
class ModelRun:
    """Per-slice evaluation of one trained diagnostic model."""

    name: str
    overall_mae: float
    by_slice: dict[str, pd.DataFrame]
    alpha: float


def representation(catalog: pd.DataFrame, labels: pd.DataFrame, slice_cols: tuple[str, ...] = SLICE_COLUMNS
                   ) -> dict[str, pd.DataFrame]:
    """Movie count, catalog share and DNA-label coverage per slice."""
    labelled = catalog["movieId"].isin(set(labels["movieId"]))
    out = {}
    for col in slice_cols:
        grp = catalog.assign(_labelled=labelled).groupby(col, dropna=False)
        frame = pd.DataFrame({"movies": grp.size(), "with_labels": grp["_labelled"].sum()})
        frame["share"] = frame["movies"] / max(len(catalog), 1)
        frame["label_coverage"] = frame["with_labels"] / frame["movies"]
        out[col] = frame.sort_values("movies", ascending=False)
    return out


def label_distribution(data: pd.DataFrame, slice_cols: tuple[str, ...] = SLICE_COLUMNS) -> dict[str, pd.DataFrame]:
    """Mean DNA score per slice for labelled movies."""
    return {col: data.groupby(col)[list(DNA_DIMENSIONS)].mean().round(3) for col in slice_cols}


def build_features(data: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    """Plot embedding plus scaled metadata (runtime, year, log votes, rating, language group).

    Tag Genome values are deliberately excluded: they are only used for labels.
    """
    runtime = pd.to_numeric(data["runtime"], errors="coerce")
    year = pd.to_numeric(data["year"], errors="coerce").astype("float64")
    votes = np.log1p(pd.to_numeric(data["vote_count"], errors="coerce").fillna(0))
    rating = pd.to_numeric(data["vote_average"], errors="coerce")
    meta = pd.DataFrame({
        "runtime": runtime.fillna(runtime.median() if runtime.notna().any() else 0),
        "year": year.fillna(year.median() if year.notna().any() else 2000),
        "log_votes": votes,
        "rating": rating.fillna(rating.median() if rating.notna().any() else 0),
    })
    meta = (meta - meta.mean()) / meta.std(ddof=0).replace(0, 1)
    lang = pd.get_dummies(data["language_group"].astype(str), prefix="lang", dtype=float)
    return np.hstack([embeddings.astype(np.float64), meta.to_numpy(), lang.to_numpy()])


def split_by_movie(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 train/validation/test split of row positions."""
    idx = np.arange(n)
    train, rest = train_test_split(idx, test_size=0.30, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


def inverse_frequency_weights(groups: pd.DataFrame) -> np.ndarray:
    """Average over slice columns of n / (k * n_group), clipped and normalised to mean 1."""
    weights = np.zeros(len(groups))
    for col in groups.columns:
        counts = groups[col].map(groups[col].value_counts())
        k = groups[col].nunique()
        weights += (len(groups) / (k * counts)).to_numpy(dtype=float)
    weights /= max(len(groups.columns), 1)
    weights = np.clip(weights, *WEIGHT_CLIP)
    return weights / weights.mean()


def resample_indices(groups: pd.Series, seed: int) -> np.ndarray:
    """Oversample small slices so each has at least half the median slice size."""
    rng = np.random.default_rng(seed)
    counts = groups.value_counts()
    target = int(max(counts.median() // 2, 1))
    positions = [np.arange(len(groups))]
    for name, count in counts.items():
        if count < target:
            members = np.flatnonzero(groups.to_numpy() == name)
            positions.append(rng.choice(members, size=target - count, replace=True))
    return np.concatenate(positions)


def fit_ridge(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
              weights: np.ndarray | None = None) -> tuple[Ridge, float]:
    """Fit Ridge for each alpha and keep the one with the lowest validation MAE."""
    best: tuple[Ridge, float, float] | None = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(x, y, sample_weight=weights)
        mae = mean_absolute_error(y_val, np.clip(model.predict(x_val), 0, 1))
        if best is None or mae < best[1]:
            best = (model, mae, alpha)
    assert best is not None
    return best[0], best[2]


def evaluate(name: str, model: Ridge, alpha: float, x_test: np.ndarray, y_test: np.ndarray,
             slices: pd.DataFrame) -> ModelRun:
    """Per-slice MAE with Fairlearn ``MetricFrame`` for every slice column."""
    pred = np.clip(model.predict(x_test), 0, 1)
    by_slice = {}
    overall = float(mean_absolute_error(y_test, pred))
    for col in slices.columns:
        mf = MetricFrame(metrics={"mae": mean_absolute_error, "count": lambda yt, yp: float(len(yt))},
                         y_true=y_test, y_pred=pred, sensitive_features=slices[col].astype(str).to_numpy())
        frame = mf.by_group.copy()
        frame["count"] = frame["count"].astype(int)
        frame["gap_vs_overall"] = frame["mae"] - overall
        frame["ratio_vs_overall"] = frame["mae"] / overall if overall else np.nan
        by_slice[col] = frame.sort_values("mae", ascending=False)
    return ModelRun(name, overall, by_slice, alpha)


def significant_slices(run: ModelRun) -> dict[str, list[str]]:
    """Slices (with enough test movies) whose MAE exceeds the disparity ratio."""
    out = {}
    for col, frame in run.by_slice.items():
        bad = frame[(frame["count"] >= MIN_SLICE_TEST) & (frame["ratio_vs_overall"] > DISPARITY_RATIO)]
        if len(bad):
            out[col] = bad.index.tolist()
    return out


def worst_gap(run: ModelRun) -> dict[str, float]:
    """Largest MAE gap between judged slices, per slice column."""
    gaps = {}
    for col, frame in run.by_slice.items():
        judged = frame[frame["count"] >= MIN_SLICE_TEST]["mae"]
        gaps[col] = float(judged.max() - judged.min()) if len(judged) > 1 else 0.0
    return gaps


def prepare_dataset(catalog: pd.DataFrame, labels: pd.DataFrame, embeddings: np.ndarray,
                    index: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows with a catalog entry, DNA labels and a plot embedding (aligned)."""
    emb_rows = index.set_index("movieId")["row"]
    data = catalog.merge(labels[["movieId", *DNA_DIMENSIONS]], on="movieId", how="inner")
    data = data[data["movieId"].isin(emb_rows.index)].reset_index(drop=True)
    return data, embeddings[emb_rows.loc[data["movieId"]].to_numpy()]


def run_bias_analysis(catalog: pd.DataFrame, labels: pd.DataFrame, embeddings: np.ndarray, index: pd.DataFrame,
                      seed: int = 42, mitigation_col: str | None = None) -> dict[str, Any]:
    """Full analysis: representation, label distribution, baseline and mitigated models."""
    data, emb = prepare_dataset(catalog, labels, embeddings, index)
    if len(data) < 40:
        raise PipelineError(f"Only {len(data)} labelled movies with plots; need at least 40 for bias analysis")
    x = build_features(data, emb)
    y = data[list(DNA_DIMENSIONS)].to_numpy(dtype=float)
    slices = data[list(SLICE_COLUMNS)].astype(str)
    tr, va, te = split_by_movie(len(data), seed)

    base_model, base_alpha = fit_ridge(x[tr], y[tr], x[va], y[va])
    baseline = evaluate("baseline", base_model, base_alpha, x[te], y[te], slices.iloc[te])

    weights = inverse_frequency_weights(slices.iloc[tr])
    rw_model, rw_alpha = fit_ridge(x[tr], y[tr], x[va], y[va], weights)
    reweighted = evaluate("reweighted", rw_model, rw_alpha, x[te], y[te], slices.iloc[te])

    gaps = worst_gap(baseline)
    target_col = mitigation_col or max(gaps, key=gaps.get)
    rs = resample_indices(slices.iloc[tr][target_col].reset_index(drop=True), seed)
    rs_model, rs_alpha = fit_ridge(x[tr][rs], y[tr][rs], x[va], y[va])
    resampled = evaluate(f"resampled_{target_col}", rs_model, rs_alpha, x[te], y[te], slices.iloc[te])

    return {
        "n_movies": int(len(data)),
        "split": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
        "representation": representation(catalog, labels),
        "label_distribution": label_distribution(data),
        "runs": {"baseline": baseline, "reweighted": reweighted, "resampled": resampled},
        "resample_column": target_col,
        "weight_range": [float(weights.min()), float(weights.max())],
    }


# ----- reporting -------------------------------------------------------------
def _md_table(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a markdown table."""
    frame = frame.reset_index()
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "---|" * len(frame.columns)
    rows = []
    for _, r in frame.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float | np.floating) else str(v) for v in r.tolist()]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def _comparison(result: dict[str, Any], col: str) -> pd.DataFrame:
    """Per-slice MAE for baseline vs mitigated runs."""
    runs = result["runs"]
    frame = pd.DataFrame({
        "test_movies": runs["baseline"].by_slice[col]["count"],
        "mae_baseline": runs["baseline"].by_slice[col]["mae"],
        "mae_reweighted": runs["reweighted"].by_slice[col]["mae"],
        "mae_resampled": runs["resampled"].by_slice[col]["mae"],
    })
    frame.index.name = col
    return frame.sort_values("test_movies", ascending=False)


def metrics_payload(result: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly version of the analysis results."""
    runs = result["runs"]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_movies": result["n_movies"],
        "split": result["split"],
        "resample_column": result["resample_column"],
        "overall_mae": {k: round(r.overall_mae, 6) for k, r in runs.items()},
        "worst_gap": {k: {c: round(v, 6) for c, v in worst_gap(r).items()} for k, r in runs.items()},
        "significant_slices": {k: significant_slices(r) for k, r in runs.items()},
        "per_slice_mae": {k: {c: f["mae"].round(6).to_dict() for c, f in r.by_slice.items()}
                          for k, r in runs.items()},
        "representation": {c: f.reset_index().to_dict(orient="records")
                           for c, f in result["representation"].items()},
    }


def render_report(result: dict[str, Any]) -> str:
    """Markdown bias report with tables and plain-language findings."""
    runs = result["runs"]
    base, rw, rs = runs["baseline"], runs["reweighted"], runs["resampled"]
    sig = significant_slices(base)
    gaps_b, gaps_w, gaps_s = worst_gap(base), worst_gap(rw), worst_gap(rs)
    lines = [
        "# Bias report: data slicing, Fairlearn analysis and mitigation",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} from {result['n_movies']:,} movies that have "
        "a clean catalog entry, Movie DNA labels and a plot embedding.",
        "",
        "## Why slices are movie attributes",
        "",
        "MovieLens 25M ratings are anonymous and the VibeSpace app does not collect user demographics, so there "
        "are no user groups to compare. The realistic risk for a recommender is unequal treatment of *content*: "
        "if the Movie DNA model is less accurate for non-English, older, niche-genre or less-popular films, those "
        "films get recommended for the wrong moods and lose exposure. We therefore slice by `language_group`, "
        "`era`, `primary_genre` and `popularity_bucket`.",
        "",
        "## 1. Representation",
        "",
        "Movie counts and Tag Genome label coverage per slice (whole clean catalog).",
        "",
    ]
    for col, frame in result["representation"].items():
        lines += [f"### {col}", "", _md_table(frame), ""]
    lines += ["## 2. Label distribution per slice", "",
              "Mean Movie DNA percentile per slice (labelled movies).", ""]
    for col in ("language_group", "era", "popularity_bucket"):
        lines += [f"### {col}", "", _md_table(result["label_distribution"][col]), ""]
    lines += [
        "## 3. Model behaviour per slice (diagnostic model)",
        "",
        "A Ridge regression from plot embedding + metadata (runtime, year, log vote count, rating, language group) "
        "to the 12 DNA scores, trained on a 70/15/15 split by movie "
        f"({result['split']['train']:,}/{result['split']['validation']:,}/{result['split']['test']:,}). "
        "It exists only to measure per-slice error with Fairlearn `MetricFrame`; it is not the Movie DNA model. "
        "MAE is averaged over the 12 dimensions on the held-out test movies.",
        "",
        f"Overall test MAE (baseline): **{base.overall_mae:.4f}** (alpha={base.alpha}).",
        "",
        f"A slice is a *significant* gap when it has at least {MIN_SLICE_TEST} test movies and its MAE is more "
        f"than {int((DISPARITY_RATIO - 1) * 100)}% above the overall MAE.",
        "",
    ]
    for col in SLICE_COLUMNS:
        lines += [f"### {col}", "", _md_table(_comparison(result, col), "{:.4f}"), ""]
    lines += [
        "## 4. Mitigation",
        "",
        "* **Re-weighting:** each training movie gets an inverse-frequency weight averaged over the four slice "
        f"columns (clipped to {WEIGHT_CLIP}, mean 1; observed range {result['weight_range'][0]:.2f}-"
        f"{result['weight_range'][1]:.2f}).",
        f"* **Re-sampling:** small `{result['resample_column']}` slices (the column with the largest baseline gap) "
        "are oversampled to at least half the median slice size.",
        "",
        "| Run | Overall MAE | Worst gap language | Worst gap era | Worst gap genre | Worst gap popularity |",
        "|---|---|---|---|---|---|",
    ]
    for name, run, gaps in (("baseline", base, gaps_b), ("reweighted", rw, gaps_w), ("resampled", rs, gaps_s)):
        lines.append(f"| {name} | {run.overall_mae:.4f} | {gaps['language_group']:.4f} | {gaps['era']:.4f} | "
                     f"{gaps['primary_genre']:.4f} | {gaps['popularity_bucket']:.4f} |")
    lines += ["", "## 5. Findings in plain language", ""]
    lines += _findings(result, sig, gaps_b, gaps_w, gaps_s)
    return "\n".join(lines) + "\n"


def _share(frame: pd.DataFrame, group: str) -> float:
    """Catalog share of one slice (0 when absent)."""
    return float(frame.loc[group, "share"]) if group in frame.index else 0.0


def recommend(base: ModelRun, candidate: ModelRun, max_overall_increase: float = 0.02) -> bool:
    """True if a mitigation narrows most worst-case gaps without raising overall MAE by more than 2%."""
    gb, gc = worst_gap(base), worst_gap(candidate)
    narrowed = sum(gc[c] < gb[c] for c in gb)
    return narrowed > len(gb) / 2 and candidate.overall_mae <= base.overall_mae * (1 + max_overall_increase)


def _findings(result: dict[str, Any], sig: dict[str, list[str]], gaps_b: dict[str, float],
              gaps_w: dict[str, float], gaps_s: dict[str, float]) -> list[str]:
    """Plain-language bullet points derived from the numbers."""
    runs = result["runs"]
    base, rw, rs = runs["baseline"], runs["reweighted"], runs["resampled"]
    rep = result["representation"]["language_group"]
    out = [
        f"- **Representation is the main bias.** English-language films are {_share(rep, 'english'):.1%} of the "
        f"catalog, other international films {_share(rep, 'other_international'):.1%}, Hindi "
        f"{_share(rep, 'hindi'):.2%} ({int(rep.loc['hindi', 'movies']) if 'hindi' in rep.index else 0} films) and "
        f"Indian regional languages {_share(rep, 'indian_regional'):.2%} "
        f"({int(rep.loc['indian_regional', 'movies']) if 'indian_regional' in rep.index else 0} films). "
        "Tag Genome labels exist mostly for well-known, frequently rated films, so this skew is inherited by "
        "the Movie DNA labels.",
    ]
    if sig:
        parts = [f"{col}: {', '.join(v)}" for col, v in sig.items()]
        out.append("- **Significant error gaps (baseline):** " + "; ".join(parts) + ".")
    else:
        out.append(f"- **Model behaviour is fairly even.** No slice with at least {MIN_SLICE_TEST} test movies has "
                   f"an MAE more than {int((DISPARITY_RATIO - 1) * 100)}% above the overall MAE "
                   f"({base.overall_mae:.4f}).")
    best_col = max(gaps_b, key=gaps_b.get)
    worst = base.by_slice[best_col]
    judged = worst[worst["count"] >= MIN_SLICE_TEST]
    if len(judged) > 1:
        out.append(f"- The largest baseline gap is in `{best_col}`: {gaps_b[best_col]:.4f} MAE between "
                   f"`{judged['mae'].idxmin()}` ({judged['mae'].min():.4f}) and `{judged['mae'].idxmax()}` "
                   f"({judged['mae'].max():.4f}).")
    else:
        out.append("- The test set is too small to compare slices reliably; run on more movies.")
    small = [f"{col}={g} ({int(f.loc[g, 'count'])})" for col, f in base.by_slice.items()
             for g in f.index if f.loc[g, "count"] < MIN_SLICE_TEST]
    if small:
        out.append(f"- **Too small to judge** (fewer than {MIN_SLICE_TEST} test movies): " + ", ".join(small) + ".")
    for name, run, gaps in (("Re-weighting", rw, gaps_w), ("Re-sampling", rs, gaps_s)):
        delta_overall = run.overall_mae - base.overall_mae
        changes = [f"{c} {gaps_b[c]:.4f}->{gaps[c]:.4f}" for c in SLICE_COLUMNS]
        out.append(f"- **{name}** changes overall MAE by {delta_overall:+.4f} and worst gaps by: "
                   + ", ".join(changes) + ".")
    chosen = [n for n, r in (("re-weighting", rw), ("re-sampling", rs)) if recommend(base, r)]
    out += [
        "- **Trade-off:** mitigation shifts model capacity towards small slices, which usually costs a little "
        "overall accuracy. Rule used: adopt a mitigation only if it narrows most worst-case gaps and raises "
        "overall MAE by at most 2%.",
        f"- **Decision:** {('adopt ' + ' and '.join(chosen)) if chosen else 'no mitigation clears that rule'} "
        "for the Movie DNA model training. Because error gaps are small but representation is very uneven, the "
        "priority is more data for under-represented slices: run TMDB stage `rest` for catalog coverage, keep "
        "inverse-frequency weights available in training, and re-check these tables on every pipeline run.",
    ]
    return out


def run_bias_slicing(cfg: PipelineConfig) -> dict[str, Any]:
    """Run the bias analysis on the current outputs and write metrics + report."""
    catalog = read_parquet(cfg.processed_dir / CLEAN_FILE)
    labels = read_parquet(cfg.processed_dir / DNA_LABELS_FILE)
    embeddings = np.load(cfg.processed_dir / EMBEDDINGS_FILE)
    index = read_parquet(cfg.processed_dir / EMBEDDINGS_INDEX_FILE)
    result = run_bias_analysis(catalog, labels, embeddings, index, seed=cfg.random_seed)
    payload = metrics_payload(result)
    write_json(cfg.metrics_dir / "bias" / BIAS_METRICS_FILE, payload)
    write_text(cfg.run_reports_dir / BIAS_REPORT_FILE, render_report(result))
    logger.info("Bias analysis: overall MAE %s; significant slices %s", payload["overall_mae"],
                payload["significant_slices"]["baseline"])
    return {"overall_mae": payload["overall_mae"], "significant_slices": payload["significant_slices"]}
