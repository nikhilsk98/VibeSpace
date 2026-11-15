"""Item 2: the protected champion store keeps every past champion; cleanup never deletes it."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.test_registry import FakeRunner
from vibespace_model.promote import rollback
from vibespace_model.registry import (
    ChecksumError,
    GcpBackend,
    LocalBackend,
    ModelRegistry,
    RegistryError,
    create_bundle,
    is_test_entry,
)


@pytest.fixture(scope="module")
def bundle_dir(part1_tree) -> Path:
    return part1_tree["cfg"].artifacts_subdir("part1") / "selected"


@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(LocalBackend(tmp_path / "registry"), champion_package="movie-dna-champions")


def _push(registry, bundle_dir, tmp_path, version):
    bundle = create_bundle(bundle_dir, tmp_path / "packed" / version, version)
    registry.push(bundle)
    return bundle


def test_becoming_champion_copies_the_bundle(registry, bundle_dir, tmp_path) -> None:
    bundle = _push(registry, bundle_dir, tmp_path, "v20260101-000000-aaaaaaa")
    assert not registry.is_protected(bundle.version)
    registry.set_champion(bundle.version, "first")
    assert registry.is_protected(bundle.version)
    assert registry.protected_versions() == [bundle.version]
    stored = registry.backend.root / "protected" / "movie-dna-champions" / bundle.version
    assert sorted(p.name for p in stored.iterdir()) == [
        f"{bundle.version}.tar.gz",
        f"{bundle.version}.tar.gz.sha256",
    ]
    meta = json.loads(registry.backend.read_text(f"champions/{bundle.version}.json"))
    assert meta["sha256"] == bundle.sha256 and meta["protected_at"]


def test_protect_is_idempotent(registry, bundle_dir, tmp_path) -> None:
    bundle = _push(registry, bundle_dir, tmp_path, "v20260101-000000-aaaaaaa")
    assert registry.protect(bundle.version) is True
    assert registry.protect(bundle.version) is False  # already there: nothing copied again
    registry.set_champion(bundle.version, "promote")
    registry.set_champion(bundle.version, "promote the same version again")
    assert registry.protected_versions() == [bundle.version]  # no duplicates
    with pytest.raises(RegistryError, match="cannot protect"):
        registry.protect("v-missing")


def test_cleanup_never_deletes_champions(registry, bundle_dir, tmp_path) -> None:
    versions = [f"v20260101-0000{i:02d}-aaaaaaa" for i in range(14)]
    for v in versions:
        _push(registry, bundle_dir, tmp_path, v)
    champions = [versions[0], versions[2], versions[5]]
    for v in champions:
        registry.set_champion(v, "promote")
    deleted = registry.backend.apply_cleanup_policy(keep_latest=10)
    assert deleted == versions[:4]  # oldest 4 regular versions go ...
    assert registry.backend.list_bundles() == versions[4:]
    assert registry.protected_versions() == sorted(champions)  # ... but every champion stays protected
    for v in champions:
        assert registry.locate(v) in ("regular", "protected")
    assert registry.locate(versions[1]) is None  # a non-champion that was cleaned up is gone


def test_rollback_to_a_cleaned_up_champion(part1_tree, registry, bundle_dir, tmp_path) -> None:
    cfg = part1_tree["cfg"]
    old = _push(registry, bundle_dir, tmp_path, "v20260101-000000-aaaaaaa")
    registry.set_champion(old.version, "first champion")
    new = _push(registry, bundle_dir, tmp_path, "v20260101-000100-bbbbbbb")
    registry.set_champion(new.version, "second champion")
    registry.delete_version(old.version)  # what the cleanup policy would do to an old regular version
    assert registry.locate(old.version) == "protected"
    root = registry.pull(old.version, tmp_path / "pulled")  # checksums verified from the protected copy
    assert json.loads((root / "manifest.json").read_text())["version"] == old.version
    result = rollback(cfg, registry)
    assert result["champion"]["version"] == old.version and result["champion"]["event"] == "rollback"
    assert registry.version_meta(old.version)["status"] == "champion"
    assert registry.protected_versions() == [old.version, new.version]


def test_corrupted_protected_copy_is_rejected(part1_tree, registry, bundle_dir, tmp_path) -> None:
    old = _push(registry, bundle_dir, tmp_path, "v20260101-000000-aaaaaaa")
    registry.set_champion(old.version, "first")
    new = _push(registry, bundle_dir, tmp_path, "v20260101-000100-bbbbbbb")
    registry.set_champion(new.version, "second")
    registry.delete_version(old.version)
    archive = (
        registry.backend.root / "protected" / "movie-dna-champions" / old.version / f"{old.version}.tar.gz"
    )
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(ChecksumError):
        rollback(part1_tree["cfg"], registry)
    assert registry.get_champion()["version"] == new.version


def test_backfill_skips_test_entries(registry, bundle_dir, tmp_path) -> None:
    a = _push(registry, bundle_dir, tmp_path, "v20260101-000000-aaaaaaa")
    t = _push(registry, bundle_dir, tmp_path, "v20260101-000100-test000")
    registry.append_history({"version": a.version, "event": "promote", "reason": "first"})
    registry.append_history({"version": t.version, "event": "test", "reason": "TEST: temporary"})
    registry.append_history({"version": a.version, "event": "rollback", "reason": "back"})
    registry.append_history({"version": "v-gone", "event": "promote", "reason": "old"})
    outcome = registry.backfill_protected()
    assert outcome == {
        a.version: "copied",
        t.version: "skipped (TEST entry)",
        "v-gone": "missing (regular version already deleted)",
    }
    assert registry.backfill_protected()[a.version] == "already protected"
    assert registry.protected_versions() == [a.version]
    assert is_test_entry({"event": "promote", "reason": "TEST candidate"}) is True
    assert is_test_entry({"event": "promote", "reason": "real"}) is False


def test_protect_detects_missing_or_mismatched_files(registry, bundle_dir, tmp_path) -> None:
    bundle = _push(registry, bundle_dir, tmp_path, "v20260101-000000-aaaaaaa")
    registry.update_version_meta(bundle.version, sha256="0" * 64)
    with pytest.raises(ChecksumError, match="metadata"):
        registry.protect(bundle.version)
    (registry.backend.root / "artifacts" / bundle.version / f"{bundle.version}.tar.gz.sha256").unlink()
    with pytest.raises(ChecksumError, match="missing"):
        registry.protect(bundle.version)


def test_gcp_backend_uses_the_champion_package(tmp_path) -> None:
    runner = FakeRunner([("versions list", 0, "v1\nv2\n", "")])
    backend = GcpBackend("proj", "us-east1", "repo", "movie-dna", "bkt", runner=runner)
    (tmp_path / "a.tar.gz").write_text("a")
    backend.upload_bundle("v1", [tmp_path / "a.tar.gz"], package="movie-dna-champions")
    assert "--package=movie-dna-champions" in runner.calls[-1]
    assert backend.bundle_exists("v1", package="movie-dna-champions")
    assert "--package=movie-dna-champions" in runner.calls[-1]
    assert backend.list_bundles("movie-dna-champions") == ["v1", "v2"]
    assert "--package=movie-dna-champions" in runner.calls[-1]
    backend.delete_bundle("v1", package="movie-dna-champions")
    assert "--package=movie-dna-champions" in runner.calls[-1]
    failing = GcpBackend(
        "proj",
        "us-east1",
        "repo",
        "movie-dna",
        "bkt",
        runner=lambda c, **k: subprocess.CompletedProcess(c, 1, "", "denied"),
    )
    assert failing.list_bundles() == []


def test_protect_cli(tree, capsys) -> None:
    from vibespace_model.cli import EXIT_OK, main
    from vibespace_model.config import load_config

    args = ["--config", str(tree["config"]), "--dry-run", "--log-level", "WARNING"]
    assert main(["train", "--config", str(tree["config"]), "--log-level", "WARNING"]) == EXIT_OK
    bundle = str(load_config(tree["config"]).artifacts_subdir("part1") / "selected")
    assert main(["push", *args, "--bundle", bundle, "--version", "v-cli"]) == EXIT_OK
    capsys.readouterr()
    assert main(["protect", *args, "--version", "v-cli"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["outcome"] == {"v-cli": "copied"}
    assert main(["protect", *args]) == EXIT_OK  # backfill from (empty) history
    assert json.loads(capsys.readouterr().out)["protected_versions"] == ["v-cli"]
    assert main(["list-versions", *args]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["protected_champions"] == ["v-cli"]
