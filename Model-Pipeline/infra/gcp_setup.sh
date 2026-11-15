#!/usr/bin/env bash
# Idempotent setup of the VibeSpace model registry and CI access in Google Cloud.
# Re-running it is safe: every resource is checked before it is created, and IAM bindings are additive.
#
#   bash Model-Pipeline/infra/gcp_setup.sh            # uses the active gcloud configuration
#   PROJECT_ID=vibespace-mlops bash Model-Pipeline/infra/gcp_setup.sh
#
# Creates (see infra/README.md for what each one is for and what it costs):
#   - Artifact Registry generic repository  vibespace-models  (model bundles: package movie-dna keeps the latest
#                                            10 versions; package movie-dna-champions is never cleaned up)
#   - Artifact Registry Docker repository   vibespace-images  (trainer image, keep latest 3 images)
#   - Cloud Storage bucket                  gs://vibespace-model-registry-nsk (champion + version metadata)
#   - Service account                       vibespace-ci (least privilege, no keys)
#   - Workload Identity pool + provider     github-pool / github-provider (GitHub repo nikhilsk98/VibeSpace only)
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-east1}"
MODELS_REPO="${MODELS_REPO:-vibespace-models}"
MODELS_PACKAGE="${MODELS_PACKAGE:-movie-dna}"
CHAMPIONS_PACKAGE="${CHAMPIONS_PACKAGE:-movie-dna-champions}"
IMAGES_REPO="${IMAGES_REPO:-vibespace-images}"
REGISTRY_BUCKET="${REGISTRY_BUCKET:-vibespace-model-registry-nsk}"
DVC_BUCKET="${DVC_BUCKET:-vibespace-dvc-nsk}"
CI_SA_NAME="${CI_SA_NAME:-vibespace-ci}"
POOL_ID="${POOL_ID:-github-pool}"
PROVIDER_ID="${PROVIDER_ID:-github-provider}"
GITHUB_REPO="${GITHUB_REPO:-nikhilsk98/VibeSpace}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "No project set: run 'gcloud config set project <id>' or pass PROJECT_ID." >&2
  exit 1
fi
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
CI_SA="${CI_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

log() { printf '==> %s\n' "$*"; }

# ------------------------------------------------------------------ APIs
log "Enabling APIs (no-op when already enabled)"
gcloud services enable artifactregistry.googleapis.com iam.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com storage.googleapis.com --project "${PROJECT_ID}"

# ------------------------------------------------------------------ Artifact Registry
create_repo() {  # name format description
  local name="$1" format="$2" description="$3"
  if gcloud artifacts repositories describe "${name}" --location "${REGION}" --project "${PROJECT_ID}" \
      >/dev/null 2>&1; then
    log "Repository ${name} exists"
  else
    log "Creating ${format} repository ${name} in ${REGION}"
    gcloud artifacts repositories create "${name}" --repository-format "${format}" --location "${REGION}" \
      --description "${description}" --project "${PROJECT_ID}"
  fi
}

cleanup_policy() {  # file keep_count  -> keep the newest N versions of every package, delete the rest
  cat >"$1" <<EOF
[
  {"name": "keep-latest-$2", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": $2}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any", "olderThan": "1d"}}
]
EOF
}

models_policy() {  # file -> Keep rules win over Delete rules, so the champion package is never deleted
  cat >"$1" <<EOF
[
  {"name": "keep-champions", "action": {"type": "Keep"},
   "condition": {"packageNamePrefixes": ["${CHAMPIONS_PACKAGE}"]}},
  {"name": "keep-latest-10", "action": {"type": "Keep"},
   "mostRecentVersions": {"packageNamePrefixes": ["${MODELS_PACKAGE}"], "keepCount": 10}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any", "olderThan": "1d"}}
]
EOF
}

create_repo "${MODELS_REPO}" generic "VibeSpace Movie DNA model bundles"
create_repo "${IMAGES_REPO}" docker "VibeSpace trainer images"
models_policy "${WORKDIR}/models-policy.json"
cleanup_policy "${WORKDIR}/images-policy.json" 3
log "Setting cleanup policies (models: keep 10 versions + every champion forever, images: keep 3)"
gcloud artifacts repositories set-cleanup-policies "${MODELS_REPO}" --location "${REGION}" \
  --project "${PROJECT_ID}" --policy "${WORKDIR}/models-policy.json" --no-dry-run >/dev/null
gcloud artifacts repositories set-cleanup-policies "${IMAGES_REPO}" --location "${REGION}" \
  --project "${PROJECT_ID}" --policy "${WORKDIR}/images-policy.json" --no-dry-run >/dev/null

# ------------------------------------------------------------------ registry bucket
if gcloud storage buckets describe "gs://${REGISTRY_BUCKET}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  log "Bucket gs://${REGISTRY_BUCKET} exists"
else
  log "Creating bucket gs://${REGISTRY_BUCKET}"
  gcloud storage buckets create "gs://${REGISTRY_BUCKET}" --project "${PROJECT_ID}" --location "${REGION}" \
    --default-storage-class STANDARD --uniform-bucket-level-access --public-access-prevention
fi
cat >"${WORKDIR}/lifecycle.json" <<'EOF'
{"rule": [{"action": {"type": "Delete"}, "condition": {"isLive": false, "daysSinceNoncurrentTime": 30}}]}
EOF
log "Bucket: versioning on, noncurrent versions deleted after 30 days"
gcloud storage buckets update "gs://${REGISTRY_BUCKET}" --versioning \
  --lifecycle-file "${WORKDIR}/lifecycle.json" >/dev/null

# ------------------------------------------------------------------ CI service account (no keys)
if gcloud iam service-accounts describe "${CI_SA}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  log "Service account ${CI_SA} exists"
else
  log "Creating service account ${CI_SA}"
  gcloud iam service-accounts create "${CI_SA_NAME}" --project "${PROJECT_ID}" \
    --display-name "VibeSpace CI (GitHub Actions)" \
    --description "Keyless CI identity: reads DVC data, writes model bundles, images and registry metadata"
fi

log "Granting least-privilege roles (scoped to single resources)"
gcloud storage buckets add-iam-policy-binding "gs://${DVC_BUCKET}" \
  --member "serviceAccount:${CI_SA}" --role roles/storage.objectViewer >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${REGISTRY_BUCKET}" \
  --member "serviceAccount:${CI_SA}" --role roles/storage.objectAdmin >/dev/null
for repo in "${MODELS_REPO}" "${IMAGES_REPO}"; do
  gcloud artifacts repositories add-iam-policy-binding "${repo}" --location "${REGION}" \
    --project "${PROJECT_ID}" --member "serviceAccount:${CI_SA}" --role roles/artifactregistry.writer >/dev/null
done

# ------------------------------------------------------------------ Workload Identity Federation (GitHub OIDC)
if gcloud iam workload-identity-pools describe "${POOL_ID}" --location global --project "${PROJECT_ID}" \
    >/dev/null 2>&1; then
  log "Workload Identity pool ${POOL_ID} exists"
else
  log "Creating Workload Identity pool ${POOL_ID}"
  gcloud iam workload-identity-pools create "${POOL_ID}" --location global --project "${PROJECT_ID}" \
    --display-name "GitHub Actions"
fi

if gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" --location global \
    --workload-identity-pool "${POOL_ID}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  log "Provider ${PROVIDER_ID} exists"
else
  log "Creating OIDC provider ${PROVIDER_ID} (only ${GITHUB_REPO} may use it)"
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" --location global \
    --workload-identity-pool "${POOL_ID}" --project "${PROJECT_ID}" \
    --display-name "GitHub OIDC" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref,attribute.actor=assertion.actor" \
    --attribute-condition "assertion.repository == '${GITHUB_REPO}'"
fi

log "Allowing ${GITHUB_REPO} workflows to impersonate ${CI_SA}"
gcloud iam service-accounts add-iam-policy-binding "${CI_SA}" --project "${PROJECT_ID}" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  >/dev/null

PROVIDER_NAME="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"
log "Done. Values for the GitHub repository variables:"
echo "  GCP_WORKLOAD_IDENTITY_PROVIDER=${PROVIDER_NAME}"
echo "  GCP_SERVICE_ACCOUNT=${CI_SA}"
