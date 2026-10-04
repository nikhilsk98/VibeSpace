# VibeSpace

> **Emotion-aware, context-aware, and continuously learning movie discovery.**

VibeSpace is an intelligent movie recommendation platform that goes beyond static genres and generic "because you watched..." suggestions. It combines a user's **current mood, desired emotional outcome, personal taste, contextual preferences, and real-time swipe feedback** to recommend movies that fit what the user actually wants to watch **right now**.

The platform supports both **solo discovery** and **group movie rooms**, where multiple users swipe independently and VibeSpace finds the best movie for the group using fairness-aware recommendation logic.

---

## Table of Contents

- [Why VibeSpace?](#why-vibespace)
- [Core Product Experience](#core-product-experience)
- [Key Features](#key-features)
- [High-Level Architecture](#high-level-architecture)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Model 1 — Movie DNA](#model-1--movie-dna)
- [Model 2 — Candidate Retrieval](#model-2--candidate-retrieval)
- [Model 3 — Personalized Ranking](#model-3--personalized-ranking)
- [Real-Time Session Learning](#real-time-session-learning)
- [Group Recommendation Rooms](#group-recommendation-rooms)
- [Data Sources](#data-sources)
- [Data Pipeline](#data-pipeline)
- [Training Pipeline](#training-pipeline)
- [Online Recommendation Flow](#online-recommendation-flow)
- [Feedback Loop](#feedback-loop)
- [MLOps Architecture](#mlops-architecture)
- [Monitoring and Drift Detection](#monitoring-and-drift-detection)
- [Evaluation Metrics](#evaluation-metrics)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Local Development](#local-development)
- [Environment Variables](#environment-variables)
- [Project Roadmap](#project-roadmap)
- [Success Criteria](#success-criteria)
- [Team Workflow](#team-workflow)
- [Future Extensions](#future-extensions)

---

# Why VibeSpace?

Traditional movie discovery is usually based on:

- Genre
- Popularity
- Historical watch behavior
- Collaborative filtering
- Static user profiles

These approaches often miss a major question:

> **What does the user want to experience tonight?**

A person may usually love complex thrillers but, on a stressful evening, want something light and comforting. Another user may feel anxious and want something hopeful or motivational. A group of friends may individually have very different tastes but still need to agree on one movie.

VibeSpace treats recommendation as a combination of:

```text
Long-Term Taste
      +
Current Mood
      +
Desired Feeling
      +
Session Preferences
      +
Real-Time Swipe Feedback
      +
Context / Constraints
      ↓
Personalized Movie Ranking
```

---

# Core Product Experience

## 1. Choose the experience

Instead of starting with only genres, the user can choose an intent such as:

- Beast Mode
- Comfort Me
- Make Me Laugh
- Inspire Me
- Mind-Bending
- Adrenaline
- Romantic
- Emotional
- Easy Watch
- Surprise Me

The user can also refine the session using controls such as:

```text
Intensity       Low  ─────────●── High
Pace            Slow ────────●── Fast
Complexity      Easy ─────●───── Deep
Emotionality    Light ──────●── Heavy
Realism         Realistic ───●── Fantasy
```

Additional constraints can include:

- Language
- Hollywood / Bollywood / regional cinema
- Runtime
- Release period
- Streaming platform
- Content preferences

---

## 2. Swipe through movie cards

Movies are shown as swipeable cards.

```text
┌─────────────────────────────┐
│                             │
│         MOVIE POSTER        │
│                             │
│          CREED III          │
│                             │
│  Intense • Motivational     │
│  Fast-paced • Emotional     │
│                             │
│        ▶ Trailer            │
│                             │
│   ← Skip         Like →     │
└─────────────────────────────┘
```

Possible signals:

- Left swipe → negative preference signal
- Right swipe → positive preference signal
- Trailer click → weak positive signal
- Trailer completion → stronger positive signal
- Movie selection → very strong positive signal

The system updates the current session profile after every interaction.

---

## 3. Recommendations improve during the same session

If a user selects **Beast Mode**, VibeSpace does not assume that every action movie is equally relevant.

For example:

```text
Right swipe:
- Creed
- Warrior
- John Wick

Left swipe:
- Deadpool
- Fast & Furious
```

The system may infer that the current session favors:

```text
Serious tone        ↑
Motivation          ↑
Grounded action     ↑
High intensity      ↑
Comedy              ↓
Over-the-top style  ↓
```

The next recommendations are then re-ranked immediately.

---

# Key Features

## Mood-first discovery

Users describe the experience they want instead of selecting only a genre.

## Current mood → desired mood

VibeSpace can support queries such as:

> "I feel anxious, but I want to feel hopeful."

The recommender searches for movies that better match that emotional destination.

## Movie DNA

Each movie is represented using richer characteristics than genre alone.

Examples:

- Energy
- Intensity
- Pace
- Humor
- Darkness
- Hopefulness
- Emotional intensity
- Romance
- Suspense
- Complexity
- Realism
- Inspiration

## Real-time swipe learning

Every swipe modifies the session profile without waiting for full model retraining.

## Long-term personalization

Historical likes, dislikes, watch decisions, and preference patterns contribute to a persistent user profile.

## Group movie rooms

Friends create a room, join using a code or link, swipe independently, and receive a group recommendation.

## YouTube trailer integration

Trailer IDs are stored with movie metadata and rendered using embedded YouTube players instead of downloading or re-hosting videos.

## Explainable recommendations

The UI can show simple reasons such as:

```text
Why this movie?

✓ High intensity
✓ Matches your fast-paced preference
✓ Similar to movies you liked this session
✓ Fits your preferred language
```

---

# High-Level Architecture

```text
                         EXTERNAL DATA
               ┌──────────────────────────┐
               │ TMDB                     │
               │ MovieLens                │
               │ MovieLens Tag Genome     │
               │ YouTube Trailer IDs      │
               └────────────┬─────────────┘
                            │
                            ▼
                  DATA INGESTION PIPELINE
                            │
                            ▼
                   VALIDATION / CLEANING
                            │
                            ▼
                  GCS / BIGQUERY / STORAGE
                            │
                            ▼
                    FEATURE ENGINEERING
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
          ▼                 ▼                  ▼
   Text Embeddings    Poster Embeddings    Metadata / Tags
          │                 │                  │
          └─────────────────┼──────────────────┘
                            ▼
                      MOVIE DNA MODEL
                            │
                            ▼
                  MOVIE REPRESENTATIONS
                            │
                            ▼
                POSTGRESQL + PGVECTOR
                            │
                            │
                            ▼
USER ──► SESSION PROFILE ──► RETRIEVAL MODEL
                            │
                            ▼
                   CANDIDATE MOVIES
                            │
                            ▼
                     RANKING MODEL
                            │
                            ▼
                FILTERING + DIVERSITY
                            │
                            ▼
                      MOVIE CARDS
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
          LEFT SWIPE      TRAILER       RIGHT SWIPE
              │             │              │
              └─────────────┼──────────────┘
                            ▼
                    INTERACTION SERVICE
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
            REDIS                       PUB/SUB
              │                            │
              ▼                            ▼
     SESSION PROFILE UPDATE            BIGQUERY
              │                            │
              ▼                            ▼
      REAL-TIME RE-RANKING          TRAINING DATA
                                           │
                                           ▼
                                        AIRFLOW
                                           │
                                           ▼
                                      RETRAINING
                                           │
                                           ▼
                                         MLFLOW
                                           │
                                           ▼
                                  MODEL REGISTRY
                                           │
                                           ▼
                                      DEPLOYMENT
                                           │
                                           ▼
                                      MONITORING
```

---

# Machine Learning Architecture

VibeSpace uses **three core trainable models**.

```text
MODEL 1
Movie DNA Predictor
"What does this movie feel like?"

          ↓

MODEL 2
Two-Tower Retrieval
"Which movies could this user/session like?"

          ↓

MODEL 3
Ranking Model
"In what exact order should we show them?"
```

Pretrained encoders may be used for feature extraction, but they are not counted as core models trained by the project.

---

# Model 1 — Movie DNA

## Goal

Represent a movie using a multidimensional emotional and semantic profile instead of only genre labels.

Example:

```text
Movie: Creed

Energy          0.86
Intensity       0.88
Pace            0.74
Humor           0.22
Darkness        0.31
Hopefulness     0.82
Emotional       0.79
Romance         0.24
Suspense        0.57
Complexity      0.42
Realism         0.78
Inspiration     0.94
```

## Inputs

Possible features include:

```text
Plot / Overview
Genres
Keywords
Runtime
Language
Release Year
Country
Poster Embedding
MovieLens Tags
Tag Genome Relevance
```

## Feature encoders

### Text

Plot descriptions can be encoded using a pretrained sentence embedding model.

```text
Plot
  ↓
Sentence Transformer
  ↓
Text Embedding
```

### Poster

Posters can optionally be encoded using CLIP.

```text
Poster
  ↓
CLIP
  ↓
Visual Embedding
```

### Structured features

Examples:

- Genre
- Language
- Runtime
- Release year
- Popularity
- Tag Genome relevance

## Output

The model produces a fixed-dimensional Movie DNA vector.

## Training labels

MovieLens Tag Genome relevance scores can be grouped into higher-level dimensions.

Example:

```text
Humor =
weighted combination of
funny + humorous + comedy

Darkness =
dark + bleak + disturbing

Inspiration =
inspiring + uplifting + motivational + feel-good
```

A smaller manually reviewed subset can be used as a higher-quality validation set.

## Evaluation

Possible metrics:

- MAE
- RMSE
- R²
- Per-dimension MAE
- Per-dimension RMSE

---

# Model 2 — Candidate Retrieval

## Goal

Efficiently retrieve a relatively small set of promising movies from the complete catalog.

Example:

```text
50,000 movies
      ↓
Two-Tower Retrieval
      ↓
~100–200 candidates
```

## Architecture

```text
USER / SESSION TOWER

Current Mood
Desired Feeling
Long-Term User Profile
Recent Swipes
Language Preferences
Runtime Preferences
Context
        │
        ▼
Neural Network
        │
        ▼
User / Session Embedding


MOVIE TOWER

Movie DNA
Plot Embedding
Genres
Language
Metadata
Popularity
        │
        ▼
Neural Network
        │
        ▼
Movie Embedding
```

Similarity can then be computed using:

```text
Dot Product
or
Cosine Similarity
```

## Training data

Initial interactions can come from MovieLens ratings.

Possible mapping:

```text
Rating >= 4 → Positive
Rating <= 2 → Negative
```

Later, VibeSpace interactions provide richer labels:

```text
Left Swipe         Negative
Right Swipe        Positive
Trailer Click      Weak Positive
Trailer Completion Positive
Movie Selection    Strong Positive
```

## Evaluation

- Recall@K
- HitRate@K
- Candidate coverage

---

# Model 3 — Personalized Ranking

## Goal

Take retrieved candidates and decide the exact order in which they should be shown.

Recommended first implementation:

> **LightGBM with learning-to-rank / LambdaRank**

## Ranking features

Possible features include:

```text
Two-Tower Similarity
Movie DNA Similarity
Session Mood Match
Long-Term User Similarity
Recent Swipe Similarity
Language Match
Runtime Difference
Genre Affinity
Actor Affinity
Director Affinity
Popularity
Novelty
Recently Skipped Similarity
```

Example:

```text
200 candidates
      ↓
Ranking Model
      ↓
Top 20
      ↓
Diversity Layer
      ↓
Cards shown to user
```

## Evaluation

- NDCG@K
- Precision@K
- MAP@K
- HitRate@K

---

# Real-Time Session Learning

VibeSpace uses two personalization layers.

## Long-term profile

Represents what the user usually likes.

```text
Sci-Fi              ↑
Complex plots       ↑
Thrillers           ↑
Slow romance        ↓
Serious tone        ↑
```

## Session profile

Represents what the user wants right now.

Example:

```text
"I'm tired.
I want something funny,
easy to follow,
and under 100 minutes."
```

The session profile can temporarily override long-term taste.

## Updating the session profile

The production model is **not retrained after every swipe**.

Instead, the current session embedding is updated in memory.

Conceptually:

```text
new_session_vector
=
old_session_vector
+ liked_movie_signal
- disliked_movie_signal
```

The updated session state can be stored in Redis.

This provides immediate personalization with very low latency.

---

# Group Recommendation Rooms

VibeSpace supports collaborative movie discovery.

## Flow

```text
User creates room
      ↓
Room code / invite link
      ↓
Friends join
      ↓
Each user provides:
- Long-term profile
- Tonight's mood
- Constraints
      ↓
Everyone swipes independently
      ↓
Group recommendation engine
      ↓
Best compromise
      ↓
Final match
```

## Why simple averaging is not enough

Suppose:

| Movie | User A | User B | User C | User D |
|---|---:|---:|---:|---:|
| Movie 1 | 95 | 92 | 88 | 10 |
| Movie 2 | 90 | 83 | 80 | 78 |
| Movie 3 | 30 | 91 | 84 | 88 |

A simple average can favor a movie that one person strongly dislikes.

VibeSpace can use a fairness-aware group score such as:

```text
Group Score
=
Average Satisfaction
+
Worst-User Satisfaction Component
```

The objective is:

> Find a movie the group likes without sacrificing one member completely.

Possible strategies to compare:

- Average satisfaction
- Majority vote
- Least misery
- Fairness-aware ranking

## Group metrics

- Average group satisfaction
- Worst-member satisfaction
- Group acceptance rate
- Cards until group match

---

# Data Sources

## MovieLens

Planned uses:

- User ratings
- Movie IDs
- Tags
- Tag Genome relevance
- Initial retrieval training data
- Recommendation evaluation

## TMDB

Planned uses:

- Movie titles
- Plot overviews
- Genres
- Posters
- Runtime
- Release dates
- Language
- Cast / crew
- Keywords
- Trailer references where available

## YouTube

YouTube videos are embedded through the official player.

VibeSpace should store trailer video IDs rather than downloading trailer files.

---

# Data Pipeline

Movie ingestion is orchestrated as a repeatable pipeline.

```text
Start
  ↓
Fetch Movie Updates
  ↓
Validate Raw Data
  ↓
Remove Duplicates
  ↓
Normalize Metadata
  ↓
Resolve Movie IDs
  ↓
Generate Text Features
  ↓
Generate Poster Features
  ↓
Generate / Predict Movie DNA
  ↓
Generate Movie Embeddings
  ↓
Store Warehouse Features
  ↓
Store Serving Features
  ↓
Run Data Quality Checks
  ↓
Finish
```

Potential orchestration:

> Apache Airflow

---

# Training Pipeline

## Movie DNA training

```text
Dataset Snapshot
      ↓
Data Validation
      ↓
Train / Validation / Test Split
      ↓
Feature Generation
      ↓
Train Movie DNA Model
      ↓
Evaluate
      ↓
Log Metrics to MLflow
      ↓
Compare with Champion
      ↓
Register Candidate
```

## Retrieval training

```text
MovieLens + VibeSpace Interactions
      ↓
Positive / Negative Sampling
      ↓
User Features
      ↓
Movie Features
      ↓
Train Two-Tower Model
      ↓
Evaluate Recall@K
      ↓
Register Candidate
```

## Ranking training

```text
Historical Impressions
      +
User Context
      +
Movie Features
      +
Swipe / Selection Outcomes
      ↓
Build Ranking Dataset
      ↓
Train Ranker
      ↓
Evaluate NDCG@K
      ↓
Register Candidate
```

---

# Online Recommendation Flow

```text
User opens VibeSpace
      ↓
Selects mood / intent / constraints
      ↓
API creates session
      ↓
Load long-term profile
      ↓
Create session vector
      ↓
Apply hard constraints
      ↓
Two-Tower retrieval
      ↓
Candidate set
      ↓
Ranking model
      ↓
Diversity re-ranking
      ↓
Movie card shown
      ↓
User interacts
      ↓
Session state updated
      ↓
Next recommendations re-ranked
```

Hard constraints should be enforced separately from ML.

Example:

```text
Language ∈ {English, Hindi}
Runtime < 120 minutes
Horror = False
```

The recommendation model should not be allowed to "approximately" satisfy those constraints.

---

# Feedback Loop

Every interaction becomes useful ML data.

Example event:

```text
event_id
user_id
session_id
timestamp

movie_id
rank_position

selected_mood
desired_mood

event_type
model_version

language
runtime
device
```

Possible `event_type` values:

```text
IMPRESSION
SWIPE_LEFT
SWIPE_RIGHT
TRAILER_OPEN
TRAILER_25
TRAILER_50
TRAILER_75
TRAILER_COMPLETE
MOVIE_SELECT
```

## Event flow

```text
Frontend
   ↓
FastAPI
   ↓
Interaction Service
   ├──────────────► Redis
   │                  ↓
   │           Session Update
   │
   └──────────────► Pub/Sub
                      ↓
                   BigQuery
                      ↓
                Training Dataset
```

---

# MLOps Architecture

VibeSpace is designed as a complete ML lifecycle rather than only a model notebook.

```text
DATA
  ↓
INGESTION
  ↓
VALIDATION
  ↓
FEATURE ENGINEERING
  ↓
DATASET VERSIONING
  ↓
TRAINING
  ↓
EXPERIMENT TRACKING
  ↓
MODEL REGISTRY
  ↓
OFFLINE EVALUATION
  ↓
STAGING
  ↓
CANARY / A-B TEST
  ↓
PRODUCTION
  ↓
MONITORING
  ↓
DRIFT DETECTION
  ↓
RETRAINING
```

## Experiment tracking

MLflow can track:

- Parameters
- Hyperparameters
- Dataset version
- Git commit
- Metrics
- Model artifacts
- Feature configuration
- Training duration

## Model registry

Example registry:

```text
movie-dna
  v12 Champion
  v13 Candidate

retrieval-model
  v21 Champion
  v22 Candidate

ranking-model
  v31 Champion
  v32 Candidate
```

## Champion / Challenger

A new model must outperform the current production model before promotion.

Example:

```text
Ranker v31
NDCG@10 = 0.712

Ranker v32
NDCG@10 = 0.746
```

If the candidate passes offline gates, a small portion of production traffic can be routed to it before full promotion.

---

# Monitoring and Drift Detection

Monitoring is divided into four layers.

## Infrastructure

- CPU
- Memory
- Pod health
- Restart count
- Network
- Resource utilization

## API

- Requests per second
- p50 latency
- p95 latency
- p99 latency
- HTTP 5xx rate
- Timeout rate

## Model

- Recall@K
- NDCG@K
- Precision@K
- Right-swipe rate
- Movie selection rate
- Cards until match

## Data

- Language distribution
- Genre distribution
- Movie DNA distribution
- Mood distribution
- Embedding drift
- Missing-value rate
- Unknown categories

## Drift detection

Possible methods:

- Population Stability Index
- Kolmogorov–Smirnov test
- KL divergence
- Embedding centroid shift

If drift exceeds a configured threshold:

```text
Detect Drift
    ↓
Generate Alert
    ↓
Trigger Retraining DAG
    ↓
Train Candidate
    ↓
Evaluate
    ↓
Promote if Better
```

---

# Evaluation Metrics

## Movie DNA

```text
MAE
RMSE
R²
Per-dimension MAE
```

## Retrieval

```text
Recall@50
Recall@100
HitRate@K
Coverage
```

## Ranking

```text
NDCG@10
Precision@10
MAP@K
```

## Product

```text
Right-Swipe Rate
Trailer Click Rate
Movie Selection Rate
Session Completion Rate
Cards Until Match
```

## Group recommendation

```text
Average Group Satisfaction
Worst-Member Satisfaction
Group Acceptance Rate
Cards Until Group Match
```

### Primary product metric

A key VibeSpace metric is:

> **Cards Until Match**

This measures how many recommendations a user or group must see before finding something they want to watch.

The goal is for VibeSpace personalization to reduce this number relative to genre-only or mood-only baselines.

---

# Technology Stack

| Layer | Planned Technology |
|---|---|
| Frontend | React / Next.js |
| Backend | FastAPI |
| Language | Python |
| Application Database | PostgreSQL |
| Vector Search | pgvector |
| Session State | Redis |
| Event Streaming | Google Pub/Sub |
| Data Warehouse | BigQuery |
| Object Storage | Google Cloud Storage |
| Pipeline Orchestration | Apache Airflow |
| Experiment Tracking | MLflow |
| Model Registry | MLflow |
| Model Training | PyTorch / LightGBM |
| Containers | Docker |
| Orchestration | Kubernetes / GKE |
| CI/CD | GitHub Actions |
| Infrastructure as Code | Terraform |
| Monitoring | Prometheus + Grafana |
| Cloud Logging | Google Cloud Logging |
| Data Validation | Pandera / Great Expectations |

The exact production stack may evolve during implementation.

---

# Repository Structure

```text
VibeSpace/
│
├── apps/
│   └── web/                     # Frontend application
│
├── services/
│   └── api/                     # FastAPI backend
│
├── ml/
│   ├── common/
│   ├── movie_dna/               # Model 1
│   ├── retrieval/               # Model 2
│   └── ranking/                 # Model 3
│
├── pipelines/
│   └── airflow/                 # Data + training DAGs
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── samples/
│   └── schemas/
│
├── features/                    # Feature engineering
│
├── evaluation/
│   ├── offline/
│   └── product/
│
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   ├── drift/
│   └── alerts/
│
├── infra/
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── ml/
│   ├── pipelines/
│   └── e2e/
│
├── notebooks/
├── scripts/
├── docs/
│
├── .github/
│   └── workflows/
│
├── .env.example
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

# Local Development

> The commands below describe the intended project workflow. They will be finalized as implementation progresses.

## Clone

```bash
git clone https://github.com/nikhilsk98/VibeSpace.git
cd VibeSpace
```

## Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install backend dependencies

```bash
pip install -r services/api/requirements.txt
```

## Configure environment

```bash
cp .env.example .env
```

Add local credentials and API keys to `.env`.

Never commit `.env`.

## Run the API

Planned command:

```bash
uvicorn services.api.app.main:app --reload
```

## Run the full local stack

Once Docker services are configured:

```bash
docker compose up --build
```

---

# Environment Variables

Example configuration:

```text
APP_ENV=development
APP_NAME=VibeSpace

DATABASE_URL=
REDIS_URL=

TMDB_API_KEY=

GCP_PROJECT_ID=
GCP_REGION=
GCS_BUCKET=
BIGQUERY_DATASET=

PUBSUB_INTERACTION_TOPIC=

MLFLOW_TRACKING_URI=

NEXT_PUBLIC_API_URL=
```

Real secrets must never be committed to Git.

---

# Project Roadmap

## Phase 1 — Data foundation

- Set up MovieLens ingestion
- Integrate TMDB metadata
- Define schemas
- Add validation
- Build sample dataset
- Add reproducible preprocessing

## Phase 2 — Movie understanding

- Define Movie DNA dimensions
- Map Tag Genome features
- Build text embeddings
- Train Movie DNA baseline
- Track experiments in MLflow

## Phase 3 — Recommendation baseline

- Build content-based baseline
- Train two-tower retrieval model
- Add vector search
- Implement evaluation metrics

## Phase 4 — Personalized ranking

- Build ranking dataset
- Train LightGBM ranker
- Implement session profile logic
- Add diversity re-ranking

## Phase 5 — Product

- Build mood selection UI
- Build swipe cards
- Integrate trailers
- Add user accounts
- Add feedback logging

## Phase 6 — Group rooms

- Create room sessions
- Add invite codes
- Collect independent swipes
- Implement group ranking
- Add group match reveal

## Phase 7 — Production MLOps

- Airflow pipelines
- MLflow registry
- Docker
- CI/CD
- Cloud deployment
- Prometheus / Grafana
- Drift detection
- Retraining workflow

## Phase 8 — Evaluation

- Compare against genre-only baseline
- Compare against mood-only baseline
- Evaluate retrieval and ranking
- Measure Cards Until Match
- Evaluate group satisfaction
- Document results

---

# Success Criteria

VibeSpace will be considered successful when it demonstrates:

1. A reproducible data ingestion and preprocessing pipeline.
2. A trained Movie DNA model with measurable offline performance.
3. A retrieval system capable of efficiently generating relevant candidates.
4. A ranking model that improves recommendation quality over simple baselines.
5. Real-time adaptation after swipe feedback.
6. A working solo recommendation experience.
7. A working group recommendation room.
8. End-to-end event logging for future retraining.
9. Experiment tracking and model versioning.
10. Production monitoring with model and data metrics.
11. A reproducible CI/CD and deployment process.
12. Clear evidence that personalization reduces the number of cards required to find a movie.

---

# Team Workflow

Recommended branch strategy:

```text
main
  │
  ├── feature/data-pipeline
  ├── feature/movie-dna
  ├── feature/retrieval
  ├── feature/ranking
  ├── feature/backend
  ├── feature/frontend
  ├── feature/group-rooms
  ├── feature/monitoring
  └── feature/infrastructure
```

Development workflow:

```text
Create Issue
   ↓
Create Feature Branch
   ↓
Implement + Test
   ↓
Open Pull Request
   ↓
CI Checks
   ↓
Code Review
   ↓
Merge to Main
```

---

# Future Extensions

Once the core platform is stable, possible extensions include:

## Emotional Arc Modeling

Instead of representing a movie with only one static emotional vector, model how the emotional experience changes over time.

```text
Beginning → Middle → Ending

Dark → Emotional → Hopeful
```

## Natural-language intent

Example:

> "I want something intelligent and intense, but not depressing."

The text can be converted into session preferences while the core recommender still performs retrieval and ranking.

## Streaming availability

Filter movies based on services available to the user.

## Social discovery

Allow friends to:

- Share profiles
- Compare Movie DNA preferences
- Create recurring groups
- Save shared watchlists

## Better group optimization

Research more advanced social-choice and fairness-aware recommendation algorithms.

---

# Project Vision

VibeSpace is not intended to be another static movie catalog.

The long-term goal is to build a recommendation system that understands:

```text
What do you usually like?

What do you want tonight?

How do you feel right now?

How do you want to feel afterward?

What are you rejecting?

What are you responding positively to?

Who are you watching with?
```

and continuously adapts until it finds the right movie.

> **VibeSpace — stop searching for a movie. Let your vibe find it.**
