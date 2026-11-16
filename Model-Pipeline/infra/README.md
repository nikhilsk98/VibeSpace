# Google Cloud resources for the model registry and CI

All resources live in project **`vibespace-mlops`** (region **`us-east1`**) and are created by
[`gcp_setup.sh`](gcp_setup.sh), which is idempotent (safe to re-run; it checks before creating and IAM bindings
are additive):

```bash
gcloud config configurations activate vibespace      # account with Owner / Editor on the project
bash Model-Pipeline/infra/gcp_setup.sh
```

| Resource | Name | Purpose | Settings |
|---|---|---|---|
| Artifact Registry (generic) | `vibespace-models` | versioned model bundles (`.tar.gz` + `.sha256`): package `movie-dna` (every pushed version) and package `movie-dna-champions` (protected copy of every version that has ever been champion) | `movie-dna`: keep the latest 10 versions, older ones deleted after 1 day; `movie-dna-champions`: **never deleted** |
| Artifact Registry (Docker) | `vibespace-images` | trainer image `us-east1-docker.pkg.dev/vibespace-mlops/vibespace-images/trainer` | keep the latest 3 images |
| Cloud Storage bucket | `gs://vibespace-model-registry-nsk` | registry metadata: `champion.json`, `champion_history.jsonl`, `versions/<version>.json`, `champions/<version>.json` (protected copy) | private (uniform access, public access prevention), versioning on, noncurrent object versions deleted after 30 days |
| Service account | `vibespace-ci@vibespace-mlops.iam.gserviceaccount.com` | identity of the GitHub Actions training workflow | **no keys**; roles below |
| Workload Identity pool | `github-pool` | trust GitHub's OIDC tokens | global |
| OIDC provider | `github-provider` | issuer `https://token.actions.githubusercontent.com` | maps `repository`, `ref`, `actor`; condition `assertion.repository == 'nikhilsk98/VibeSpace'` |

Existing resource reused: the DVC bucket `gs://vibespace-dvc-nsk` (only a read binding for CI was added).

## Protected champion store (past champions are never deleted)

Every successful CI run pushes a new version, and the cleanup policy keeps only the newest 10 versions of
`movie-dna`. Without protection an old champion recorded in `champion_history.jsonl` could be deleted and a
`rollback` to it would fail. Therefore:

1. Whenever a version becomes champion (`promote`, `rollback`, `set_champion`), its verified bundle is
   copied to the package **`movie-dna-champions`** under the same version name (idempotent: copied once) and
   its metadata to **`champions/<version>.json`** in the registry bucket.
2. The repository's cleanup policy has a **Keep** rule for that package. In Artifact Registry, Keep rules
   always win over Delete rules, so the repository itself guarantees those versions survive; no client code
   or schedule has to remember to skip them.
3. `pull` and `rollback` look in `movie-dna` first and fall back to `movie-dna-champions`, so rolling back to
   any past champion works even after its regular version was cleaned up (checksums are verified either way).
4. The bucket's lifecycle rule only deletes *noncurrent* object versions, so live metadata (including
   `champions/`) is never removed by it.

Applied policy on `vibespace-models` (`gcloud artifacts repositories list-cleanup-policies vibespace-models
--location us-east1`):

| Rule | Action | Condition |
|---|---|---|
| `keep-champions` | Keep | package name prefix `movie-dna-champions` (all versions, forever) |
| `keep-latest-10` | Keep | the 10 most recent versions per package with prefix `movie-dna` |
| `delete-older` | Delete | any version older than 1 day not kept by a rule above |

Backfill (`python -m vibespace_model.cli protect`): copies every champion in the history (TEST entries
excluded) that is not yet protected. Current protected versions: `v20261115-005730-e63b305` and
`v20261115-011119-e63b305` (the TEST candidate `v20261115-005829-e63b305` was skipped).

## CI service account permissions (least privilege)

| Scope | Role | Why |
|---|---|---|
| bucket `gs://vibespace-dvc-nsk` | `roles/storage.objectViewer` | `dvc pull` of the processed outputs |
| bucket `gs://vibespace-model-registry-nsk` | `roles/storage.objectAdmin` | read / write champion and version metadata |
| repository `vibespace-models` | `roles/artifactregistry.writer` | upload / download model bundles |
| repository `vibespace-images` | `roles/artifactregistry.writer` | push the trainer image |
| service account itself | `roles/iam.workloadIdentityUser` for `principalSet://…/attribute.repository/nikhilsk98/VibeSpace` | lets workflows of that repository (and only it) impersonate the account |

There are **no project-level roles** and **no service-account keys**: GitHub Actions exchanges its OIDC token
for short-lived credentials through Workload Identity Federation (`google-github-actions/auth`).

Values for the GitHub repository variables (set later, see `../docs/ci_setup_for_later.md`):

```text
GCP_WORKLOAD_IDENTITY_PROVIDER = projects/218127030715/locations/global/workloadIdentityPools/github-pool/providers/github-provider
GCP_SERVICE_ACCOUNT            = vibespace-ci@vibespace-mlops.iam.gserviceaccount.com
```

## Cost (near zero)

| Item | Pricing (list, us-east1) | Expected use | Monthly |
|---|---|---|---|
| Artifact Registry storage | first 0.5 GB free, then $0.10 / GB-month | measured: trainer image 563 MiB + model bundles ≈ 5 MiB; at most 3 images (≈ 1.7 GB), 10 regular bundles (≈ 13 MB) and one protected copy per champion (≈ 1.3 MB each, so ≈ 1 MB per month even with weekly champion changes) | now ≈ $0.01, at most ≈ $0.12 |
| Cloud Storage (registry bucket) | $0.020 / GB-month (Standard) | measured: 7.6 KB of JSON | < $0.01 |
| Cloud Storage operations / egress | cents per 10k operations; egress inside `us-east1` and to GitHub runners is small | a few runs per week | < $0.05 |
| Workload Identity Federation, IAM, service accounts | free | – | $0 |
| **Total (new resources)** | | | **≈ $0.02–0.20 / month**, covered by the existing $10 budget alert |

(The existing DVC bucket adds ≈ $0.03 / month for its 1.5 GB.)

The image repository dominates the cost; its cleanup policy keeps at most 3 images. Deleting the trainer
images (`gcloud artifacts docker images delete …`) brings the total to a few cents.

## Local machine changes

Pushing the trainer image from a laptop (`docker push us-east1-docker.pkg.dev/...`) needs Docker to obtain
Google credentials, so `gcloud auth configure-docker us-east1-docker.pkg.dev` was run once. It added one entry
to `~/.docker/config.json`:

```json
"credHelpers": {
  "us-east1-docker.pkg.dev": "gcloud"
}
```

- **What it is:** a credential *helper* mapping. When Docker talks to `us-east1-docker.pkg.dev`, it asks
  `gcloud` for a short-lived token of the active gcloud account; no password, key or token is stored in the
  file.
- **Why:** to push the trainer image to the `vibespace-images` repository (Part 4). CI does not need it on
  laptops; GitHub Actions configures its own runner.
- **Harmless:** it only affects that one registry host; other registries keep using Docker Desktop's
  credential store (`credsStore`).
- **Removing it:** delete the `"us-east1-docker.pkg.dev": "gcloud"` line (and the `credHelpers` object if it
  becomes empty) from `~/.docker/config.json`, or run:

```bash
python3 - <<'EOF'
import json, os
path = os.path.expanduser("~/.docker/config.json")
config = json.load(open(path))
config.get("credHelpers", {}).pop("us-east1-docker.pkg.dev", None)
if not config.get("credHelpers"):
    config.pop("credHelpers", None)
json.dump(config, open(path, "w"), indent=2)
EOF
```

  (`gcloud auth configure-docker` has no "remove" option.) Pushing to the image repository from that
  laptop then needs the command again.

## Removing everything

```bash
gcloud artifacts repositories delete vibespace-models --location us-east1
gcloud artifacts repositories delete vibespace-images --location us-east1
gcloud storage rm --recursive gs://vibespace-model-registry-nsk
gcloud iam workload-identity-pools delete github-pool --location global
gcloud iam service-accounts delete vibespace-ci@vibespace-mlops.iam.gserviceaccount.com
gcloud storage buckets remove-iam-policy-binding gs://vibespace-dvc-nsk \
  --member serviceAccount:vibespace-ci@vibespace-mlops.iam.gserviceaccount.com --role roles/storage.objectViewer
```
