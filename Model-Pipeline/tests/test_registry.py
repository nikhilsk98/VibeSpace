"""Part 4 registry: bundles and checksums, local and (faked) GCP backends, champion and history."""

from __future__ import annotations

import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from vibespace_model.registry import (
    ChecksumError,
    GcpBackend,
    LocalBackend,
    ModelRegistry,
    RegistryError,
    create_bundle,
    make_version,
    unpack_bundle,
    verify_archive,
)
from vibespace_model.select import load_bundle


@pytest.fixture(scope="module")
def bundle_dir(part1_tree) -> Path:
    return part1_tree["cfg"].artifacts_subdir("part1") / "selected"


@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(LocalBackend(tmp_path / "registry"))


def _pack(bundle_dir: Path, tmp_path: Path, version: str = "v20260101-000000-abc1234"):
    return create_bundle(bundle_dir, tmp_path / "packed", version, {"mlflow_run_id": "run-1"})


def test_version_naming() -> None:
    when = datetime(2026, 11, 14, 19, 5, 7, tzinfo=UTC)
    assert make_version("e63b305137abcdef", when) == "v20261114-190507-e63b305"
    assert make_version(None, when) == "v20261114-190507-nogit"
    assert make_version("unknown", when).endswith("-nogit")
    assert make_version("abc").startswith("v20")


def test_bundle_contents_and_checksums(bundle_dir, tmp_path) -> None:
    bundle = _pack(bundle_dir, tmp_path)
    assert bundle.archive.name == "v20260101-000000-abc1234.tar.gz"
    assert (
        bundle.checksum_file.read_text().split()[0]
        == bundle.sha256
        == verify_archive(bundle.archive, bundle.checksum_file)
    )
    with tarfile.open(bundle.archive) as tar:
        names = {
            Path(n).relative_to("v20260101-000000-abc1234").as_posix()
            for n in tar.getnames()
            if n != "v20260101-000000-abc1234"
        }
    for member in (
        "model_card.json",
        "features.json",
        "metrics.json",
        "bias_summary.json",
        "split.json",
        "data_version.json",
        "manifest.json",
        "checksums.sha256",
        "model/params.json",
    ):
        assert member in names, member
    assert bundle.manifest["mlflow_run_id"] == "run-1"
    assert bundle.manifest["split_hash"]


def test_checksum_mismatch_is_detected(bundle_dir, tmp_path) -> None:
    bundle = _pack(bundle_dir, tmp_path)
    with bundle.archive.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ChecksumError, match="checksum mismatch"):
        verify_archive(bundle.archive, bundle.checksum_file)
    with pytest.raises(ChecksumError, match="missing"):
        verify_archive(tmp_path / "nope.tar.gz", bundle.checksum_file)


def test_corrupted_member_is_detected(bundle_dir, tmp_path) -> None:
    bundle = _pack(bundle_dir, tmp_path)
    extracted = tmp_path / "x"
    with tarfile.open(bundle.archive) as tar:
        tar.extractall(extracted, filter="data")
    root = extracted / bundle.version
    (root / "features.json").write_text("{}")
    rebuilt = tmp_path / "rebuilt.tar.gz"
    with tarfile.open(rebuilt, "w:gz") as tar:
        tar.add(root, arcname=bundle.version)
    from vibespace_model.registry import sha256_file

    sha = tmp_path / "rebuilt.tar.gz.sha256"
    sha.write_text(f"{sha256_file(rebuilt)}  rebuilt.tar.gz\n")
    with pytest.raises(ChecksumError, match="features.json"):
        unpack_bundle(rebuilt, sha, tmp_path / "out")


def test_not_a_bundle(tmp_path) -> None:
    with pytest.raises(RegistryError, match="not a saved model bundle"):
        create_bundle(tmp_path, tmp_path / "out", "v1")


def test_push_pull_round_trip(part1_tree, bundle_dir, registry, tmp_path) -> None:
    bundle = _pack(bundle_dir, tmp_path)
    meta = registry.push(bundle, {"local_registry_version": "1"})
    assert meta["status"] == "registered" and meta["sha256"] == bundle.sha256
    assert registry.version_meta(bundle.version)["local_registry_version"] == "1"
    root = registry.pull(bundle.version, tmp_path / "pulled")
    original, builder, _ = load_bundle(bundle_dir)
    pulled, pulled_builder, card = load_bundle(root)
    data = part1_tree["summary"]
    assert card["split_hash"] == data["split_hash"]
    X = np.random.default_rng(0).normal(size=(5, len(builder.feature_names))).astype(np.float32)
    np.testing.assert_array_equal(original.predict(X), pulled.predict(X))
    assert pulled_builder.feature_names == builder.feature_names
    assert registry.verify_remote(bundle.version) == bundle.version
    with pytest.raises(RegistryError, match="already exists"):
        registry.push(bundle)


def test_pull_errors(bundle_dir, registry, tmp_path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        registry.pull("v-missing", tmp_path / "p")
    bundle = _pack(bundle_dir, tmp_path)
    registry.push(bundle)
    registry.update_version_meta(bundle.version, sha256="0" * 64)
    with pytest.raises(ChecksumError, match="metadata"):
        registry.pull(bundle.version, tmp_path / "p2")
    stored = registry.backend.root / "artifacts" / bundle.version / f"{bundle.version}.tar.gz.sha256"
    stored.unlink()
    with pytest.raises(ChecksumError, match="missing its archive"):
        registry.pull(bundle.version, tmp_path / "p3")


def test_champion_and_history(bundle_dir, registry, tmp_path) -> None:
    assert registry.get_champion() is None and registry.history() == []
    first, second = _pack(bundle_dir, tmp_path, "v1"), _pack(bundle_dir, tmp_path / "b", "v2")
    registry.push(first)
    registry.push(second)
    with pytest.raises(RegistryError, match="not in"):
        registry.set_champion("v3", "nope")
    registry.set_champion("v1", "first")
    champion = registry.set_champion("v2", "better", details={"val_mae": 0.1})
    assert champion["previous_version"] == "v1" and champion["val_mae"] == 0.1
    assert registry.get_champion()["version"] == "v2"
    assert [h["version"] for h in registry.history()] == ["v1", "v2"]
    assert registry.version_meta("v1")["status"] == "superseded"
    assert registry.version_meta("v2")["status"] == "champion"
    registry.set_champion("v1", "back", event="rollback")
    assert registry.version_meta("v2")["status"] == "rolled back"
    assert [m["version"] for m in registry.list_versions()] == ["v1", "v2"]
    registry.delete_version("v2")
    assert not registry.backend.bundle_exists("v2")
    # v2 was champion: its protected copy and metadata survive the deletion of the regular version
    assert registry.is_protected("v2") and registry.version_meta("v2")["protected_at"]
    assert len(registry.history()) == 3  # history is kept


def test_local_backend_text_objects(tmp_path) -> None:
    backend = LocalBackend(tmp_path)
    assert backend.list_keys("versions/") == [] and backend.read_text("x.json") is None
    backend.write_text("versions/a.json", "1")
    backend.write_text("champion.json", "2")
    assert backend.list_keys("versions/") == ["versions/a.json"]
    backend.delete_text("versions/a.json")
    backend.delete_text("versions/a.json")  # no error when missing
    with pytest.raises(RegistryError):
        backend.download_bundle("nope", tmp_path / "d")


class FakeRunner:
    """Records gcloud commands and answers them from a script of (match, returncode, stdout, stderr)."""

    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or []

    def __call__(self, command, input=None, capture_output=True, text=True, check=False):  # noqa: A002
        self.calls.append(command)
        for match, code, out, err in self.answers:
            if match in " ".join(command):
                if code == 0 and "download" in command:
                    dest = Path(next(a for a in command if a.startswith("--destination=")).split("=", 1)[1])
                    (dest / "v1.tar.gz").write_text("x")
                return subprocess.CompletedProcess(command, code, out, err)
        return subprocess.CompletedProcess(command, 0, "", "")


def test_gcp_backend_commands(tmp_path) -> None:
    runner = FakeRunner(
        [
            (
                "storage cat gs://bkt/missing.json",
                1,
                "",
                "ERROR: The following URLs matched no objects or files",
            ),
            ("storage cat gs://bkt/broken.json", 1, "", "permission denied"),
            ("storage cat gs://bkt/champion.json", 0, '{"version": "v1"}', ""),
            ("versions describe v-missing", 1, "", "NOT_FOUND"),
            (
                "storage ls",
                0,
                "gs://bkt/versions/\ngs://bkt/versions/v1.json\ngs://bkt/versions/v2.json\n",
                "",
            ),
            ("generic upload", 0, "", ""),
            ("generic download", 0, "", ""),
        ]
    )
    backend = GcpBackend("proj", "us-east1", "repo", "movie-dna", "bkt", runner=runner)
    (tmp_path / "a.tar.gz").write_text("a")
    backend.upload_bundle("v1", [tmp_path / "a.tar.gz"])
    upload = runner.calls[-1]
    assert upload[:4] == ["gcloud", "artifacts", "generic", "upload"]
    assert "--repository=repo" in upload and "--package=movie-dna" in upload and "--version=v1" in upload
    assert "--project=proj" in upload and "--location=us-east1" in upload
    assert [p.name for p in backend.download_bundle("v1", tmp_path / "dl")] == ["v1.tar.gz"]
    assert backend.bundle_exists("v1") and not backend.bundle_exists("v-missing")
    assert backend.read_text("champion.json") == '{"version": "v1"}'
    assert backend.read_text("missing.json") is None
    with pytest.raises(RegistryError, match="permission denied"):
        backend.read_text("broken.json")
    backend.write_text("versions/v1.json", "{}")
    assert (
        runner.calls[-1][:3] == ["gcloud", "storage", "cp"]
        and runner.calls[-1][4] == "gs://bkt/versions/v1.json"
    )
    assert backend.list_keys("versions/") == ["versions/v1.json", "versions/v2.json"]
    backend.delete_bundle("v1")
    assert runner.calls[-1][1:4] == ["artifacts", "versions", "delete"]
    backend.delete_text("versions/v1.json")
    assert "gs://bkt/versions/v1.json" in runner.calls[-1]


def test_gcp_backend_failures() -> None:
    runner = FakeRunner(
        [
            ("generic upload", 1, "", "PERMISSION_DENIED: artifactregistry.repositories.uploadArtifacts"),
            ("storage ls", 1, "", "not found"),
        ]
    )
    backend = GcpBackend("proj", "us-east1", "repo", "movie-dna", "bkt", runner=runner)
    with pytest.raises(RegistryError, match="PERMISSION_DENIED"):
        backend.upload_bundle("v1", [])
    assert backend.list_keys("versions/") == []


def test_champion_json_shape(bundle_dir, registry, tmp_path) -> None:
    registry.push(_pack(bundle_dir, tmp_path, "v9"))
    registry.set_champion("v9", "reason text")
    stored = json.loads(registry.backend.read_text("champion.json"))
    assert {"version", "set_at", "event", "reason", "previous_version"} <= set(stored)
