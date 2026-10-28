"""Regenerate the tiny, deterministic fixture dataset used by the tests.

Run from ``Data-Pipeline/``::

    python -m tests.fixtures.build_fixtures

It writes a miniature MovieLens 25M layout (``tests/fixtures/movielens``) and
fake TMDB responses (``tests/fixtures/tmdb``). The handcrafted movies cover the
edge cases (missing TMDB id, duplicate TMDB id, 404, adult title, missing plot,
runtime 0, runtime outlier, year mismatch, unknown language, Hindi and Telugu
titles); the generated movies give the bias analysis enough rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ML_DIR = HERE / "movielens"
TMDB_DIR = HERE / "tmdb"
N_GENERATED = 44
EXTRA_TAGS = ["007", "atmospheric", "boring", "classic"]

# movieId, title, genres, tmdbId (None = missing), genome?, tmdb payload overrides (None = 404)
HANDCRAFTED = [
    (1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy", 862, True,
     {"title": "Toy Story", "original_language": "en", "release_date": "1995-10-30", "runtime": 81,
      "genres": ["Animation", "Adventure", "Family", "Comedy"], "vote_count": 18000, "popularity": 90.1,
      "overview": "A cowboy doll feels threatened when a flashy space toy joins the bedroom."}),
    (2, "Jumanji (1995)", "Adventure|Children|Fantasy", 8844, True,
     {"title": "Jumanji", "original_language": "en", "release_date": "1995-12-15", "runtime": 104,
      "genres": ["Adventure", "Fantasy", "Family"], "vote_count": 10500, "popularity": 40.2,
      "overview": "Two kids find a board game that releases jungle hazards with every roll of the dice."}),
    (3, "Heat (1995)", "Action|Crime|Thriller", 949, True,
     {"title": "Heat", "original_language": "en", "release_date": "1995-12-15", "runtime": 170,
      "genres": ["Crime", "Drama", "Action", "Thriller"], "vote_count": 7000, "popularity": 30.5,
      "overview": "A detective and a master thief circle each other across Los Angeles."}),
    (4, "Dilwale Dulhania Le Jayenge (1995)", "Comedy|Drama|Romance", 19404, True,
     {"title": "Dilwale Dulhania Le Jayenge", "original_language": "hi", "release_date": "1995-10-20",
      "runtime": 190, "genres": ["Comedy", "Drama", "Romance"], "vote_count": 4300, "popularity": 25.0,
      "overview": "Two young travellers fall in love in Europe and must win over a strict father."}),
    (5, "Baahubali: The Beginning (2015)", "Action|Adventure|Drama", 256040, True,
     {"title": "Baahubali: The Beginning", "original_language": "te", "release_date": "2015-07-10",
      "runtime": 159, "genres": ["Action", "Drama", "Fantasy"], "vote_count": 900, "popularity": 15.3,
      "overview": "A young man raised by villagers learns of his royal lineage and an ancient feud."}),
    (6, "Amelie (2001)", "Comedy|Romance", 194, True,
     {"title": "Amelie", "original_language": "fr", "release_date": "2001-04-25", "runtime": 122,
      "genres": ["Comedy", "Romance"], "vote_count": 11000, "popularity": 28.0,
      "overview": "A shy waitress in Paris decides to quietly change the lives of the people around her."}),
    (7, "Metropolis (1927)", "Drama|Sci-Fi", 19, True,
     {"title": "Metropolis", "original_language": "de", "release_date": "1927-01-10", "runtime": 153,
      "genres": ["Drama", "Science Fiction"], "vote_count": 2600, "popularity": 14.0,
      "overview": "In a futuristic city split between workers and planners, a robot double sows chaos."}),
    (8, "No Link Movie (1999)", "Drama", None, True, None),
    (9, "Toy Story Duplicate Link (1995)", "Comedy", 862, True, None),
    (10, "Missing At TMDB (2003)", "Horror", 999999, True, None),
    (11, "Silent Unknown (2005)", "Drama", 11111, True,
     {"title": "Silent Unknown", "original_language": "xx", "release_date": "2005-03-01", "runtime": 0,
      "genres": ["Drama"], "vote_count": 3, "popularity": 0.6, "overview": ""}),
    (12, "Adult Only (2004)", "Drama", 12121, True,
     {"title": "Adult Only", "original_language": "en", "release_date": "2004-01-01", "runtime": 90,
      "genres": ["Drama"], "vote_count": 10, "popularity": 1.0, "adult": True,
      "overview": "A fixture record that must be quarantined because it is flagged adult."}),
    (13, "Very Long Epic (2000)", "Documentary|War", 13131, True,
     {"title": "Very Long Epic", "original_language": "en", "release_date": "2010-05-05", "runtime": 500,
      "genres": ["Documentary", "War", "History"], "vote_count": 50, "popularity": 2.0,
      "overview": "An enormous archival chronicle of a long war, told across many hours of footage."}),
    (14, "Spirited Away (2001)", "Adventure|Animation|Fantasy", 129, True,
     {"title": "Spirited Away", "original_language": "ja", "release_date": "2001-07-20", "runtime": 125,
      "genres": ["Animation", "Family", "Fantasy"], "vote_count": 15000, "popularity": 60.0,
      "overview": "A girl wanders into a world of spirits and works in a bathhouse to free her parents."}),
    (15, "The Godfather (1972)", "Crime|Drama", 238, False,
     {"title": "The Godfather", "original_language": "en", "release_date": "1972-03-14", "runtime": 175,
      "genres": ["Drama", "Crime"], "vote_count": 19000, "popularity": 95.0,
      "overview": "The ageing head of a crime family hands control of his empire to a reluctant son."}),
    (16, "Parasite (2019)", "Comedy|Drama|Thriller", 496243, False,
     {"title": "Parasite", "original_language": "ko", "release_date": "2019-05-30", "runtime": 133,
      "genres": ["Comedy", "Thriller", "Drama"], "vote_count": 16000, "popularity": 70.0,
      "overview": "A poor family schemes its way into the household of a wealthy one."}),
]

GEN_GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller"]
GEN_LANGS = ["en", "en", "en", "fr", "es", "hi", "ta", "ja"]


def mapping_tags() -> list[str]:
    """All tags used by the real Movie DNA mapping."""
    with open(ROOT / "configs" / "movie_dna_mapping_v1.yaml", encoding="utf-8") as fh:
        dims = yaml.safe_load(fh)["dimensions"]
    tags: list[str] = []
    for spec in dims.values():
        for tag in (spec.get("positive") or []) + (spec.get("inverse") or []):
            if tag not in tags:
                tags.append(tag)
    return tags


def payload(tmdb_id: int, fields: dict) -> dict:
    """A TMDB-shaped response with keywords and videos."""
    genres = [{"id": i, "name": g} for i, g in enumerate(fields.pop("genres"), start=1)]
    return {
        "id": tmdb_id, "imdb_id": f"tt{tmdb_id:07d}", "adult": fields.pop("adult", False),
        "original_title": fields["title"], "spoken_languages": [{"iso_639_1": fields["original_language"]}],
        "production_countries": [{"iso_3166_1": "US"}], "genres": genres, "vote_average": 7.1,
        "poster_path": f"/poster{tmdb_id}.jpg",
        "keywords": {"keywords": [{"id": 1, "name": "fixture"}]},
        "videos": {"results": [
            {"site": "YouTube", "type": "Teaser", "key": f"teaser{tmdb_id}", "official": True,
             "published_at": "2020-01-01T00:00:00Z", "size": 1080},
            {"site": "YouTube", "type": "Trailer", "key": f"trailer{tmdb_id}", "official": True,
             "published_at": "2019-01-01T00:00:00Z", "size": 1080},
        ]},
        **fields,
    }


def generated_movies(rng: np.random.Generator) -> list[tuple]:
    """Synthetic but well-formed movies (all with genome data)."""
    rows = []
    for i in range(N_GENERATED):
        movie_id = 100 + i
        tmdb_id = 500000 + i
        year = int(rng.integers(1950, 2020))
        genre = GEN_GENRES[i % len(GEN_GENRES)]
        lang = GEN_LANGS[i % len(GEN_LANGS)]
        fields = {
            "title": f"Fixture Film {i}", "original_language": lang, "release_date": f"{year}-06-15",
            "runtime": int(rng.integers(80, 150)), "genres": [genre], "vote_count": int(rng.integers(5, 20000)),
            "popularity": float(round(rng.uniform(1, 50), 2)),
            "overview": f"Fixture plot number {i}: a {genre.lower()} story about people facing event {i}.",
        }
        rows.append((movie_id, f"Fixture Film {i} ({year})", genre, tmdb_id, True, fields))
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Write a small CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    """Write every fixture file."""
    rng = np.random.default_rng(7)
    movies = HANDCRAFTED + generated_movies(rng)
    write_csv(ML_DIR / "movies.csv", ["movieId", "title", "genres"], [[m[0], m[1], m[2]] for m in movies])
    write_csv(ML_DIR / "links.csv", ["movieId", "imdbId", "tmdbId"],
              [[m[0], f"{m[0]:07d}", "" if m[3] is None else m[3]] for m in movies])
    write_csv(ML_DIR / "ratings.csv", ["userId", "movieId", "rating", "timestamp"],
              [[1, m[0], 4.0, 1_000_000_000] for m in movies[:5]])
    write_csv(ML_DIR / "tags.csv", ["userId", "movieId", "tag", "timestamp"], [[1, 1, "funny", 1_000_000_000]])

    tags = mapping_tags() + EXTRA_TAGS
    write_csv(ML_DIR / "genome-tags.csv", ["tagId", "tag"], [[i + 1, t] for i, t in enumerate(tags)])
    scores = []
    for m in movies:
        if m[4]:
            for i in range(len(tags)):
                scores.append([m[0], i + 1, f"{rng.uniform(0, 1):.5f}"])
    write_csv(ML_DIR / "genome-scores.csv", ["movieId", "tagId", "relevance"], scores)

    TMDB_DIR.mkdir(parents=True, exist_ok=True)
    for old in TMDB_DIR.glob("*.json"):
        old.unlink()
    for m in movies:
        if m[3] is not None and m[5] is not None and m[0] != 9:
            data = payload(m[3], dict(m[5]))
            (TMDB_DIR / f"{m[3]}.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(movies)} movies, {len(tags)} tags, {len(scores)} genome scores")


if __name__ == "__main__":
    main()
