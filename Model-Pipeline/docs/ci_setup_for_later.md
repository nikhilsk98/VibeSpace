# CI/CD setup to do after the team push (Part 6)

Everything on the Google Cloud side already exists (see `../infra/README.md`). The workflows
`.github/workflows/model-ci.yml` and `.github/workflows/model-train.yml` were built and tested locally
(`act`, actionlint) but have **never run on GitHub**. Nothing was pushed and no GitHub setting was changed.
Do the steps below **after** the workflow files and `Model-Pipeline/` are on `main`.

## 1. Repository variables (required for `model-train.yml`)

Run from any folder with `gh` logged in as a repository admin:

```bash
gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER --repo nikhilsk98/VibeSpace \
  --body "projects/218127030715/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
gh variable set GCP_SERVICE_ACCOUNT --repo nikhilsk98/VibeSpace \
  --body "vibespace-ci@vibespace-mlops.iam.gserviceaccount.com"
gh variable list --repo nikhilsk98/VibeSpace
```

These are **variables, not secrets**: they are resource names, not credentials. There is no key to store;
GitHub's OIDC token is exchanged for short-lived Google credentials (Workload Identity Federation), and only
workflows of `nikhilsk98/VibeSpace` are accepted (`assertion.repository == 'nikhilsk98/VibeSpace'`).

## 2. Optional: Slack notifications

Create an incoming webhook in Slack (Apps → Incoming Webhooks → Add to a channel), then:

```bash
gh secret set SLACK_WEBHOOK_URL --repo nikhilsk98/VibeSpace     # paste the webhook URL when prompted
```

Without this secret the Slack step is skipped (never an error). The run still writes a job summary, and
GitHub emails the person who triggered a failed run (Settings → Notifications → Actions).

## 3. First real run

```bash
gh workflow run model-train.yml --repo nikhilsk98/VibeSpace --ref main -f dry_run=true    # safe first run
gh run watch --repo nikhilsk98/VibeSpace                                                    # follow it
gh workflow run model-train.yml --repo nikhilsk98/VibeSpace --ref main                      # real promotion
```

Or: GitHub → **Actions** → **model-train** → **Run workflow** (tick "Dry run" for the first one).
`model-ci.yml` runs by itself on every pull request or push that touches `Model-Pipeline/**`.

**What success looks like** (job "train, gate and promote", about 10–15 min):

1. Install, Workload Identity login, `dvc pull` of the 4 processed files (≈ 30 MB, not the raw data).
2. Train: the tuned MLP trains in a few seconds; the split hash matches `ci.expected_split_hash`.
3. Validation gate passes (MAE improvement ≈ 38% ≥ 15%, R² ≈ 0.51 ≥ 0.30).
4. Bias gate passes (worst gated slice ≈ 1.10 × overall ≤ 1.25).
5. Promote: the candidate is compared with the champion on the same split. Expected result: identical
   validation MAE (same data, parameters and seed), so **"promoted"** within the 0.002 tolerance, or
   "not promoted (rollback)" if it is worse; either way the job is green.
6. Artifacts `model-train-<run id>` (metrics, bias report, model card, plots, decision) are attached.

**What a not-promoted run looks like.** The job is **green** (exit code 0): the pipeline worked, the new
model was just not better than the champion by the rule (validation MAE more than 0.002 worse, both evaluated
on the same split). To make sure nobody mistakes it for a new champion:

- the run page (and the commit's checks) shows a yellow **warning annotation**
  `NOT PROMOTED: champion unchanged (rollback)` with the candidate version, the unchanged champion and the
  reasons, written by the step "Flag a not-promoted result";
- the job summary starts with the bold line `NOT PROMOTED: champion unchanged (rollback): <champion> stays
  champion` followed by the reasons;
- the notification message (and Slack, when configured) has the title
  `NOT PROMOTED: champion unchanged (rollback)` and lists the same reasons;
- the candidate is still stored in the registry with status `not promoted (rollback)`, so it can be inspected
  (`cli list-versions`, `cli pull --version …`); `champion.json` is untouched.

A successful promotion instead starts the summary with `PROMOTED: <version> is the new champion`; failed
gates start it with `VALIDATION GATE FAILED` / `BIAS GATE FAILED` (and the job is red).

**Reading the job summary** (run page → Summary): a table of steps with status and seconds, validation and
test metrics, the validation and bias gate results with reasons, and the promotion decision with the
candidate version, the champion after the run and the reasons. The "Notification message" step log shows the
exact text that Slack would receive.

On a push to `main`, a second job **trainer image** builds `Model-Pipeline/Dockerfile` and pushes
`us-east1-docker.pkg.dev/vibespace-mlops/vibespace-images/trainer:<short sha>` and `:latest`.

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `google-github-actions/auth` fails: "unable to generate access token", `iam.serviceAccounts.getAccessToken` denied | The `roles/iam.workloadIdentityUser` binding or the provider condition does not match. Check: `gcloud iam service-accounts get-iam-policy vibespace-ci@vibespace-mlops.iam.gserviceaccount.com` shows `principalSet://…/attribute.repository/nikhilsk98/VibeSpace`; the repository was not renamed or forked (forks are rejected on purpose); re-run `bash Model-Pipeline/infra/gcp_setup.sh`. |
| auth fails: "the target service account does not exist" / variables empty | Repository variables not set (step 1) or misspelled: `gh variable list`. |
| `dvc pull` fails with 403 | The CI account needs `roles/storage.objectViewer` on `gs://vibespace-dvc-nsk` (granted by the setup script): `gcloud storage buckets get-iam-policy gs://vibespace-dvc-nsk`. |
| `dvc pull` "missing cache files" | Somebody changed `Data-Pipeline/dvc.lock` without `dvc push`: run `dvc push` from the machine that produced the data. |
| Train step fails with "split hash … differs from ci.expected_split_hash" | The data changed (new `dvc.lock`). Retrain Parts 1–3 locally, then update `ci.expected_split_hash` and `ci.params` in `configs/model_config.yaml`. |
| Promote fails with `PERMISSION_DENIED` on Artifact Registry or the bucket | Check `roles/artifactregistry.writer` on `vibespace-models` / `vibespace-images` and `roles/storage.objectAdmin` on `gs://vibespace-model-registry-nsk` (setup script). |
| Validation or bias gate fails | The job stops there and nothing is pushed; read the reasons in the job summary and in `bias_report.md` (artifact). |
| Run is slow or Actions minutes run out | Public repos are free; private repos get 2,000 min/month. One training run uses ≈ 15 min, `model-ci` ≈ 10 min. Pip downloads are cached by `setup-python`. |

## 5. Manual rollback from a laptop

```bash
cd Model-Pipeline
gcloud config configurations activate vibespace && gcloud auth application-default login
.venv/bin/python -m vibespace_model.cli champion                       # current champion + history
.venv/bin/python -m vibespace_model.cli list-versions                  # all versions and their status
.venv/bin/python -m vibespace_model.cli rollback                       # back to the previous champion
.venv/bin/python -m vibespace_model.cli rollback --to-version v20261115-005730-e63b305
```

`rollback` downloads the target bundle and verifies its SHA-256 checksums before switching
`champion.json`, appends the change to `champion_history.jsonl`, and moves the local MLflow alias
`champion` when that version exists in the local store. Add `--dry-run` to practise on a local folder.
