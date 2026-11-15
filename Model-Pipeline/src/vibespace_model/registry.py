"""Part 4: model registry on GCP (Artifact Registry generic repository + a metadata bucket).

Bundle format: ``<version>.tar.gz`` holding the saved model folder, ``features.json``, ``model_card.json``,
``metrics.json``, ``bias_summary.json``, ``split.json`` (split hash), ``data_version.json``, ``manifest.json``
and ``checksums.sha256`` (one SHA-256 per member), plus a separate ``<version>.tar.gz.sha256`` file with the
archive's own SHA-256. Versions are named ``v<YYYYMMDD-HHMMSS>-<git sha>`` (UTC).

Storage layout:
- Artifact Registry generic repository ``vibespace-models``, package ``movie-dna``, one version per bundle
  (both files).
- Bucket ``gs://vibespace-model-registry-nsk``: ``versions/<version>.json`` (metadata, promotion decision),
  ``champion.json`` and ``champion_history.jsonl`` (every champion change, newest last).
- **Protected champion store:** every version that becomes champion is also copied to the package
  ``movie-dna-champions`` (same version name) and its metadata to ``champions/<version>.json``. The
  repository's cleanup policy keeps the newest 10 versions of ``movie-dna`` but has a Keep rule for
  ``movie-dna-champions``, so past champions are never deleted and ``rollback`` / ``pull`` always find them.

All storage calls go through a :class:`Backend`: :class:`GcpBackend` shells out to ``gcloud`` (the same CLI
the CI job authenticates), :class:`LocalBackend` writes to a local folder (``--dry-run``, ``act``, tests).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess  # noqa: S404  (fixed gcloud commands, no shell)
import tarfile
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vibespace_model.errors import VibeSpaceModelError

LOGGER = logging.getLogger(__name__)
CHAMPION_KEY = "champion.json"
HISTORY_KEY = "champion_history.jsonl"
VERSIONS_PREFIX = "versions/"
CHAMPIONS_PREFIX = "champions/"
CHECKSUMS_MEMBER = "checksums.sha256"
DEFAULT_CHAMPION_PACKAGE = "movie-dna-champions"


class RegistryError(VibeSpaceModelError):
    """A registry operation failed (missing version, storage error)."""


class ChecksumError(RegistryError):
    """A bundle does not match its recorded SHA-256."""


# ---------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_version(git_sha: str | None, when: datetime | None = None) -> str:
    """``v<YYYYMMDD-HHMMSS>-<7-char git sha>`` in UTC (``nogit`` when unknown)."""
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    short = (git_sha or "")[:7] if git_sha and git_sha != "unknown" else "nogit"
    return f"v{stamp}-{short}"


def utc_now() -> str:
    """Current UTC time, ISO 8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------- bundles


@dataclass
class BundleFile:
    """A packed bundle on local disk."""

    version: str
    archive: Path
    checksum_file: Path
    sha256: str
    manifest: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def create_bundle(
    bundle_dir: Path, out_dir: Path, version: str, extra: dict[str, Any] | None = None
) -> BundleFile:
    """Pack a saved model bundle (model/, features.json, model_card.json) into ``<version>.tar.gz``."""
    card_path = bundle_dir / "model_card.json"
    if (
        not card_path.is_file()
        or not (bundle_dir / "model").is_dir()
        or not (bundle_dir / "features.json").is_file()
    ):
        raise RegistryError(
            f"{bundle_dir} is not a saved model bundle (model/, features.json, model_card.json)"
        )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="bundle-", dir=out_dir))
    try:
        root = staging / version
        shutil.copytree(bundle_dir / "model", root / "model")
        shutil.copy2(bundle_dir / "features.json", root / "features.json")
        shutil.copy2(card_path, root / "model_card.json")
        _write_json(root / "metrics.json", card.get("metrics", {}))
        _write_json(
            root / "bias_summary.json",
            {
                "bias_gate": card.get("bias_gate"),
                "bias_gate_test": card.get("bias_gate_test"),
                "slice_summary_val": card.get("slice_summary_val"),
            },
        )
        _write_json(root / "split.json", {"split_hash": card.get("split_hash")})
        _write_json(root / "data_version.json", card.get("data_version", {}))
        manifest = {
            "version": version,
            "created_at": utc_now(),
            "model_type": card.get("model_type"),
            "stage": card.get("stage"),
            "split_hash": card.get("split_hash"),
            "data_version_id": (card.get("data_version") or {}).get("data_version_id"),
            "git_commit": (card.get("data_version") or {}).get("git_commit"),
            **(extra or {}),
        }
        _write_json(root / "manifest.json", manifest)
        lines = [
            f"{sha256_file(p)}  {p.relative_to(root).as_posix()}"
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ]
        (root / CHECKSUMS_MEMBER).write_text("\n".join(lines) + "\n", encoding="utf-8")

        archive = out_dir / f"{version}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=version)
        digest = sha256_file(archive)
        checksum_file = out_dir / f"{version}.tar.gz.sha256"
        checksum_file.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return BundleFile(version, archive, checksum_file, digest, manifest)


def verify_archive(archive: Path, checksum_file: Path) -> str:
    """Check the archive against its ``.sha256`` file; return the digest."""
    if not archive.is_file() or not checksum_file.is_file():
        raise ChecksumError(f"bundle or checksum file missing: {archive}, {checksum_file}")
    expected = checksum_file.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(archive)
    if actual != expected:
        raise ChecksumError(f"checksum mismatch for {archive.name}: expected {expected}, got {actual}")
    return actual


def unpack_bundle(archive: Path, checksum_file: Path, dest: Path) -> Path:
    """Verify the archive, extract it and check every member against ``checksums.sha256``."""
    verify_archive(archive, checksum_file)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    roots = [p for p in dest.iterdir() if p.is_dir() and (p / CHECKSUMS_MEMBER).is_file()]
    if len(roots) != 1:
        raise ChecksumError(f"{archive.name} does not contain exactly one bundle folder")
    root = roots[0]
    for line in (root / CHECKSUMS_MEMBER).read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        member = root / rel
        if not member.is_file() or sha256_file(member) != digest:
            raise ChecksumError(f"member {rel} of {archive.name} is missing or corrupted")
    return root


# ---------------------------------------------------------------------------- storage backends


class Backend(ABC):
    """Storage operations the registry needs: bundle files (Artifact Registry) and small text objects."""

    description = "backend"

    @abstractmethod
    def upload_bundle(self, version: str, files: list[Path], package: str | None = None) -> None:
        """Store the bundle files under ``version`` (default package unless ``package`` is given)."""

    @abstractmethod
    def download_bundle(self, version: str, dest: Path, package: str | None = None) -> list[Path]:
        """Download all files of ``version`` into ``dest``."""

    @abstractmethod
    def delete_bundle(self, version: str, package: str | None = None) -> None:
        """Remove ``version`` from the bundle store."""

    @abstractmethod
    def bundle_exists(self, version: str, package: str | None = None) -> bool:
        """True when ``version`` is in the bundle store."""

    @abstractmethod
    def list_bundles(self, package: str | None = None) -> list[str]:
        """Version names stored in a package, sorted."""

    @abstractmethod
    def read_text(self, key: str) -> str | None:
        """Text object, or None when it does not exist."""

    @abstractmethod
    def write_text(self, key: str, text: str) -> None:
        """Create or replace a text object."""

    @abstractmethod
    def delete_text(self, key: str) -> None:
        """Remove a text object (no error if missing)."""

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """Keys starting with ``prefix``."""


class LocalBackend(Backend):
    """Filesystem stand-in for GCP.

    Default package: ``<root>/artifacts/<version>/``; any other package (the protected champion store):
    ``<root>/protected/<package>/<version>/``; text objects: ``<root>/bucket/<key>``.
    """

    def __init__(self, root: Path, package: str = "movie-dna") -> None:
        self.root = Path(root)
        self.package = package
        self.description = f"local dry-run folder {self.root}"

    def _package_dir(self, package: str | None) -> Path:
        if package in (None, self.package):
            return self.root / "artifacts"
        return self.root / "protected" / package

    def _bundle_dir(self, version: str, package: str | None = None) -> Path:
        return self._package_dir(package) / version

    def upload_bundle(self, version: str, files: list[Path], package: str | None = None) -> None:
        target = self._bundle_dir(version, package)
        if target.exists():
            raise RegistryError(f"version {version} already exists")
        target.mkdir(parents=True)
        for file in files:
            shutil.copy2(file, target / file.name)

    def download_bundle(self, version: str, dest: Path, package: str | None = None) -> list[Path]:
        source = self._bundle_dir(version, package)
        if not source.is_dir():
            raise RegistryError(f"version {version} not found in {self.description}")
        dest.mkdir(parents=True, exist_ok=True)
        return [Path(shutil.copy2(f, dest / f.name)) for f in sorted(source.iterdir())]

    def delete_bundle(self, version: str, package: str | None = None) -> None:
        shutil.rmtree(self._bundle_dir(version, package), ignore_errors=True)

    def bundle_exists(self, version: str, package: str | None = None) -> bool:
        return self._bundle_dir(version, package).is_dir()

    def list_bundles(self, package: str | None = None) -> list[str]:
        folder = self._package_dir(package)
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def apply_cleanup_policy(self, keep_latest: int = 10) -> list[str]:
        """Simulate the Artifact Registry cleanup policy: keep the newest versions of the default package,
        never touch the protected champion package. Returns the deleted versions."""
        versions = self.list_bundles()  # version names start with a UTC timestamp: name order = age order
        doomed = versions[:-keep_latest] if keep_latest > 0 else versions
        for version in doomed:
            self.delete_bundle(version)
        return doomed

    def read_text(self, key: str) -> str | None:
        path = self.root / "bucket" / key
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def write_text(self, key: str, text: str) -> None:
        path = self.root / "bucket" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def delete_text(self, key: str) -> None:
        (self.root / "bucket" / key).unlink(missing_ok=True)

    def list_keys(self, prefix: str) -> list[str]:
        base = self.root / "bucket"
        if not base.is_dir():
            return []
        keys = [p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()]
        return sorted(k for k in keys if k.startswith(prefix) and not k.endswith(".tmp"))


Runner = Callable[..., subprocess.CompletedProcess]


def _run_gcloud(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """The one place that starts gcloud (tests replace it to prove dry runs never reach GCP)."""
    return subprocess.run(command, **kwargs)  # noqa: S603


class GcpBackend(Backend):
    """Artifact Registry generic repository + Cloud Storage bucket, through the ``gcloud`` CLI."""

    def __init__(
        self,
        project: str,
        location: str,
        repository: str,
        package: str,
        bucket: str,
        runner: Runner | None = None,
        gcloud: str = "gcloud",
    ) -> None:
        self.project, self.location, self.repository = project, location, repository
        self.package, self.bucket = package, bucket
        self.runner, self.gcloud = runner, gcloud
        self.description = f"Artifact Registry {location}/{repository} (package {package}) + gs://{bucket}"

    def _run(
        self, args: list[str], input_text: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        command = [self.gcloud, *args, f"--project={self.project}", "--quiet"]
        runner = self.runner or _run_gcloud
        result = runner(command, input=input_text, capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            raise RegistryError(f"gcloud {' '.join(args[:3])} failed: {result.stderr.strip()[-500:]}")
        return result

    def _repo_args(self) -> list[str]:
        return [f"--location={self.location}", f"--repository={self.repository}"]

    def upload_bundle(self, version: str, files: list[Path], package: str | None = None) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for file in files:
                shutil.copy2(file, Path(tmp) / file.name)
            self._run(
                [
                    "artifacts",
                    "generic",
                    "upload",
                    *self._repo_args(),
                    f"--package={package or self.package}",
                    f"--version={version}",
                    f"--source-directory={tmp}",
                ]
            )

    def download_bundle(self, version: str, dest: Path, package: str | None = None) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        before = set(dest.iterdir())
        self._run(
            [
                "artifacts",
                "generic",
                "download",
                *self._repo_args(),
                f"--package={package or self.package}",
                f"--version={version}",
                f"--destination={dest}",
            ]
        )
        return sorted(set(dest.rglob("*")) - before - {p for p in dest.rglob("*") if p.is_dir()})

    def delete_bundle(self, version: str, package: str | None = None) -> None:
        self._run(
            [
                "artifacts",
                "versions",
                "delete",
                version,
                *self._repo_args(),
                f"--package={package or self.package}",
            ]
        )

    def bundle_exists(self, version: str, package: str | None = None) -> bool:
        result = self._run(
            [
                "artifacts",
                "versions",
                "describe",
                version,
                *self._repo_args(),
                f"--package={package or self.package}",
            ],
            check=False,
        )
        return result.returncode == 0

    def list_bundles(self, package: str | None = None) -> list[str]:
        result = self._run(
            [
                "artifacts",
                "versions",
                "list",
                *self._repo_args(),
                f"--package={package or self.package}",
                "--format=value(name.basename())",
            ],
            check=False,
        )
        if result.returncode != 0:
            return []
        return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())

    def _url(self, key: str) -> str:
        return f"gs://{self.bucket}/{key}"

    def read_text(self, key: str) -> str | None:
        result = self._run(["storage", "cat", self._url(key)], check=False)
        if result.returncode != 0:
            stderr = result.stderr.lower()
            if any(text in stderr for text in ("no urls matched", "matched no objects", "404", "not found")):
                return None
            raise RegistryError(f"reading {self._url(key)} failed: {result.stderr.strip()[-300:]}")
        return result.stdout

    def write_text(self, key: str, text: str) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=Path(key).suffix, delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
        try:
            self._run(["storage", "cp", tmp.name, self._url(key)])
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def delete_text(self, key: str) -> None:
        self._run(["storage", "rm", self._url(key)], check=False)

    def list_keys(self, prefix: str) -> list[str]:
        result = self._run(["storage", "ls", self._url(prefix)], check=False)
        if result.returncode != 0:
            return []
        base = f"gs://{self.bucket}/"
        return sorted(
            line.strip()[len(base) :]
            for line in result.stdout.splitlines()
            if line.strip().startswith(base) and not line.strip().endswith("/")
        )


# ---------------------------------------------------------------------------- registry


class ModelRegistry:
    """Versioned bundles, version metadata, champion pointer and champion history."""

    def __init__(self, backend: Backend, champion_package: str = DEFAULT_CHAMPION_PACKAGE) -> None:
        self.backend = backend
        self.champion_package = champion_package

    def locate(self, version: str) -> str | None:
        """Package holding ``version``: the regular package first, then the protected champion store."""
        if self.backend.bundle_exists(version):
            return "regular"
        if self.backend.bundle_exists(version, self.champion_package):
            return "protected"
        return None

    # versions -----------------------------------------------------------------
    def push(self, bundle: BundleFile, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Upload the bundle files and write ``versions/<version>.json``."""
        verify_archive(bundle.archive, bundle.checksum_file)
        self.backend.upload_bundle(bundle.version, [bundle.archive, bundle.checksum_file])
        meta = {
            **bundle.manifest,
            "sha256": bundle.sha256,
            "pushed_at": utc_now(),
            "status": "registered",
            **(metadata or {}),
        }
        self.write_version_meta(bundle.version, meta)
        LOGGER.info("Pushed %s to %s", bundle.version, self.backend.description)
        return meta

    def pull(self, version: str, dest: Path) -> Path:
        """Download, verify and unpack ``version`` into ``dest`` (protected store as fallback)."""
        where = self.locate(version)
        if where is None:
            raise RegistryError(f"version {version} not found in {self.backend.description}")
        download = dest / "_download"
        package = self.champion_package if where == "protected" else None
        files = self.backend.download_bundle(version, download, package)
        archive = next((f for f in files if f.name == f"{version}.tar.gz"), None)
        checksum = next((f for f in files if f.name == f"{version}.tar.gz.sha256"), None)
        if archive is None or checksum is None:
            raise ChecksumError(f"version {version} is missing its archive or checksum file")
        meta = self.version_meta(version)
        if meta and meta.get("sha256") and meta["sha256"] != verify_archive(archive, checksum):
            raise ChecksumError(f"version {version} does not match the SHA-256 recorded in its metadata")
        return unpack_bundle(archive, checksum, dest / "bundle")

    def verify_remote(self, version: str) -> str:
        """Download ``version`` to a temporary folder, verify every checksum; return the manifest version."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self.pull(version, Path(tmp))
            return json.loads((root / "manifest.json").read_text(encoding="utf-8"))["version"]

    def delete_version(self, version: str) -> None:
        """Remove a regular bundle and its metadata. The protected champion copy and history are kept."""
        self.backend.delete_bundle(version)
        self.backend.delete_text(f"{VERSIONS_PREFIX}{version}.json")

    def version_meta(self, version: str) -> dict[str, Any] | None:
        """Metadata of one version (protected champion copy as fallback; None when unknown)."""
        text = self.backend.read_text(f"{VERSIONS_PREFIX}{version}.json")
        if not text:
            text = self.backend.read_text(f"{CHAMPIONS_PREFIX}{version}.json")
        return json.loads(text) if text else None

    # protected champion store -----------------------------------------------------------
    def is_protected(self, version: str) -> bool:
        """True when ``version`` is in the protected champion package."""
        return self.backend.bundle_exists(version, self.champion_package)

    def protected_versions(self) -> list[str]:
        """All versions in the protected champion package."""
        return self.backend.list_bundles(self.champion_package)

    def protect(self, version: str) -> bool:
        """Copy a regular bundle (verified) into the protected store; False if it is already there."""
        if self.is_protected(version):
            return False
        if not self.backend.bundle_exists(version):
            raise RegistryError(f"cannot protect {version}: it is not in {self.backend.description}")
        with tempfile.TemporaryDirectory() as tmp:
            files = self.backend.download_bundle(version, Path(tmp))
            archive = next((f for f in files if f.name == f"{version}.tar.gz"), None)
            checksum = next((f for f in files if f.name == f"{version}.tar.gz.sha256"), None)
            if archive is None or checksum is None:
                raise ChecksumError(f"version {version} is missing its archive or checksum file")
            digest = verify_archive(archive, checksum)
            meta = self.version_meta(version) or {"version": version}
            if meta.get("sha256") and meta["sha256"] != digest:
                raise ChecksumError(f"version {version} does not match the SHA-256 recorded in its metadata")
            self.backend.upload_bundle(version, [archive, checksum], self.champion_package)
        self.backend.write_text(
            f"{CHAMPIONS_PREFIX}{version}.json",
            json.dumps({**meta, "protected_at": utc_now(), "sha256": digest}, indent=2, default=str),
        )
        LOGGER.info("Protected champion %s in package %s", version, self.champion_package)
        return True

    def backfill_protected(self) -> dict[str, str]:
        """Protect every past champion in the history (TEST entries excluded); returns version -> outcome."""
        outcome: dict[str, str] = {}
        for entry in self.history():
            version = entry["version"]
            if version in outcome:
                continue
            if is_test_entry(entry):
                outcome[version] = "skipped (TEST entry)"
            elif self.is_protected(version):
                outcome[version] = "already protected"
            elif self.backend.bundle_exists(version):
                self.protect(version)
                outcome[version] = "copied"
            else:
                outcome[version] = "missing (regular version already deleted)"
        return outcome

    def write_version_meta(self, version: str, meta: dict[str, Any]) -> None:
        """Create or replace ``versions/<version>.json``."""
        self.backend.write_text(f"{VERSIONS_PREFIX}{version}.json", json.dumps(meta, indent=2, default=str))

    def update_version_meta(self, version: str, **fields: Any) -> dict[str, Any]:
        """Merge fields into a version's metadata, and into its protected copy when there is one."""
        regular_key, protected_key = f"{VERSIONS_PREFIX}{version}.json", f"{CHAMPIONS_PREFIX}{version}.json"
        regular, protected = self.backend.read_text(regular_key), self.backend.read_text(protected_key)
        updated: dict[str, Any] = {}
        if regular is not None or protected is None:
            updated = {**(json.loads(regular) if regular else {"version": version}), **fields}
            self.write_version_meta(version, updated)
        if protected is not None:
            protected_meta = {**json.loads(protected), **fields}
            self.backend.write_text(protected_key, json.dumps(protected_meta, indent=2, default=str))
            updated = updated or protected_meta
        return updated

    def list_versions(self) -> list[dict[str, Any]]:
        """All version metadata, oldest first."""
        metas = []
        for key in self.backend.list_keys(VERSIONS_PREFIX):
            text = self.backend.read_text(key)
            if text:
                metas.append(json.loads(text))
        return sorted(metas, key=lambda m: (m.get("created_at", ""), m.get("version", "")))

    # champion -----------------------------------------------------------------
    def get_champion(self) -> dict[str, Any] | None:
        """Current ``champion.json`` (None when no champion has been set)."""
        text = self.backend.read_text(CHAMPION_KEY)
        return json.loads(text) if text else None

    def history(self) -> list[dict[str, Any]]:
        """Champion history, oldest first."""
        text = self.backend.read_text(HISTORY_KEY) or ""
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def set_champion(
        self, version: str, reason: str, event: str = "promote", details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Protect ``version``, point ``champion.json`` at it and append the change to the history."""
        where = self.locate(version)
        if where is None:
            raise RegistryError(f"cannot make {version} champion: it is not in {self.backend.description}")
        if where == "regular":
            self.protect(version)  # no-op when it is already protected
        previous = self.get_champion()
        champion = {
            "version": version,
            "set_at": utc_now(),
            "event": event,
            "reason": reason,
            "previous_version": previous["version"] if previous else None,
            **(details or {}),
        }
        self.append_history(champion)
        self.backend.write_text(CHAMPION_KEY, json.dumps(champion, indent=2, default=str))
        self.update_version_meta(version, status="champion")
        if previous and previous["version"] != version:
            self.update_version_meta(
                previous["version"], status="rolled back" if event == "rollback" else "superseded"
            )
        LOGGER.info("Champion is now %s (%s)", version, event)
        return champion

    def append_history(self, entry: dict[str, Any]) -> None:
        """Append one JSON line to ``champion_history.jsonl``."""
        existing = self.backend.read_text(HISTORY_KEY) or ""
        line = json.dumps(entry, default=str)
        self.backend.write_text(HISTORY_KEY, existing + line + "\n")


def is_test_entry(entry: dict[str, Any]) -> bool:
    """History entries written by the rollback test (never protected, never backfilled)."""
    return entry.get("event") == "test" or str(entry.get("reason", "")).startswith("TEST")
