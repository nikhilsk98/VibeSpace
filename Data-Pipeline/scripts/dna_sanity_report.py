"""Sanity report for the Movie DNA labels.

For each dimension the report lists the top 10 and bottom 10 movies and runs
three automatic checks that flag dimensions which may be mapped wrongly:

* genre lift: the top 100 movies should over-represent the genres one would
  expect for the dimension (e.g. Comedy for ``humor``);
* spread: the raw scores must vary enough to rank movies meaningfully;
* redundancy: two dimensions should not be almost perfectly correlated.

The flags are hints for the team review, not proof that a mapping is wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from scripts.io_utils import write_text

if TYPE_CHECKING:
    from scripts.build_dna_labels import DnaMapping

EXPECTED_GENRES: dict[str, set[str]] = {
    "energy": {"Action", "Adventure"},
    "intensity": {"Thriller", "Crime", "Horror", "War", "Action"},
    "pace": {"Action", "Thriller", "Adventure"},
    "humor": {"Comedy"},
    "darkness": {"Horror", "Thriller", "Crime", "Film-Noir", "War"},
    "hopefulness": {"Comedy", "Children", "Animation", "Romance", "Musical"},
    "emotional_intensity": {"Drama", "Romance", "War"},
    "romance": {"Romance"},
    "suspense": {"Thriller", "Mystery", "Horror", "Crime"},
    "complexity": {"Mystery", "Sci-Fi", "Thriller", "Film-Noir"},
    "realism": {"Documentary", "Drama", "War", "Crime"},
    "inspiration": {"Drama", "Documentary", "Children"},
}
TOP_N = 10
LIFT_SAMPLE = 100
MIN_LIFT = 1.2
MIN_RAW_STD = 0.03
MAX_CORRELATION = 0.85


def genre_lift(labels: pd.DataFrame, movies: pd.DataFrame, dim: str, sample: int = LIFT_SAMPLE) -> float:
    """Share of top-``sample`` movies with an expected genre, divided by the base rate."""
    expected = EXPECTED_GENRES.get(dim, set())
    merged = labels[["movieId", dim]].merge(movies[["movieId", "genres"]], on="movieId", how="left")
    has = merged["genres"].fillna("").str.split("|").map(lambda g: bool(expected & set(g)))
    base = has.mean()
    if base == 0 or not expected:
        return float("nan")
    top = has[merged[dim].rank(ascending=False, method="first") <= sample]
    return float(top.mean() / base)


def find_flags(labels: pd.DataFrame, movies: pd.DataFrame, dims: tuple[str, ...]) -> dict[str, list[str]]:
    """Return {dimension: [reasons]} for dimensions that look suspicious."""
    flags: dict[str, list[str]] = {}
    for dim in dims:
        reasons = []
        lift = genre_lift(labels, movies, dim)
        if pd.notna(lift) and lift < MIN_LIFT:
            reasons.append(f"weak genre signal (lift {lift:.2f} < {MIN_LIFT})")
        raw_std = float(labels[f"{dim}_raw"].std()) if f"{dim}_raw" in labels else float("nan")
        if pd.notna(raw_std) and raw_std < MIN_RAW_STD:
            reasons.append(f"raw scores barely vary (std {raw_std:.3f})")
        if reasons:
            flags[dim] = reasons
    corr = labels[list(dims)].corr(method="spearman")
    for i, a in enumerate(dims):
        for b in dims[i + 1:]:
            if abs(corr.loc[a, b]) > MAX_CORRELATION:
                msg = f"highly correlated with {b} (rho {corr.loc[a, b]:.2f})"
                flags.setdefault(a, []).append(msg)
    return flags


def _title_list(frame: pd.DataFrame) -> str:
    """Comma-separated titles for a markdown cell."""
    return "; ".join(str(t).replace("|", "/") for t in frame["title"].tolist())


def render_report(labels: pd.DataFrame, movies: pd.DataFrame, mapping: DnaMapping,
                  flags: dict[str, list[str]], dims: tuple[str, ...]) -> str:
    """Build the markdown report text."""
    merged = labels.merge(movies[["movieId", "title"]], on="movieId", how="left")
    lines = [
        "# Movie DNA sanity report",
        "",
        f"Mapping version: **{mapping.version}** ({mapping.status or 'reviewed by the team'}).  ",
        f"Movies labelled: **{len(labels):,}**. Generated: {datetime.now(UTC):%Y-%m-%d %H:%M UTC}.",
        "",
        "Scores are percentiles (0 = lowest, 1 = highest) of the raw tag-relevance score.",
        "The Tag Genome is used only to build these labels; it is never a model input.",
        "",
        "## Automatic checks",
        "",
        "| Dimension | Genre lift (top 100) | Raw std | Status |",
        "|---|---|---|---|",
    ]
    for dim in dims:
        lift = genre_lift(labels, movies, dim)
        std = labels[f"{dim}_raw"].std() if f"{dim}_raw" in labels else float("nan")
        status = "FLAG: " + "; ".join(flags[dim]) if dim in flags else "ok"
        lines.append(f"| {dim} | {lift:.2f} | {std:.3f} | {status} |")
    lines += ["", f"Flagged dimensions: **{', '.join(flags) if flags else 'none'}**.", ""]

    for dim in dims:
        ordered = merged.sort_values([dim, "movieId"], ascending=[False, True])
        pos = ", ".join(f"`{t}`" for t in mapping.positive[dim])
        inv = ", ".join(f"`{t}`" for t in mapping.inverse.get(dim, [])) or "none"
        lines += [
            f"## {dim}",
            "",
            f"Positive tags: {pos}. Inverse tags: {inv}.",
            "",
            f"- **Top {TOP_N}:** {_title_list(ordered.head(TOP_N))}",
            f"- **Bottom {TOP_N}:** {_title_list(ordered.tail(TOP_N).iloc[::-1])}",
            "",
        ]
    return "\n".join(lines)


def write_sanity_report(labels: pd.DataFrame, movies: pd.DataFrame, mapping: DnaMapping, path: Path) -> list[str]:
    """Write the report to ``path`` and return the flagged dimension names."""
    from scripts.build_dna_labels import DNA_DIMENSIONS

    flags = find_flags(labels, movies, DNA_DIMENSIONS)
    write_text(Path(path), render_report(labels, movies, mapping, flags, DNA_DIMENSIONS))
    return sorted(flags)
