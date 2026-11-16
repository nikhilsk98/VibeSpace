"""CLI handlers of the registry commands: push, pull, champion, promote, rollback, list-versions, protect."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from vibespace_model.config import ModelConfig
from vibespace_model.promote import build_registry, local_registration, promote, rollback
from vibespace_model.registry import RegistryError, create_bundle, make_version
from vibespace_model.select import load_bundle
from vibespace_model.train import prepare


def default_bundle(cfg: ModelConfig, args: argparse.Namespace) -> Path:
    """``--bundle`` or the Part 3 final model."""
    path = Path(args.bundle) if args.bundle else cfg.artifacts_subdir("part3") / "final_model"
    if not path.is_dir():
        raise RegistryError(f"bundle folder {path} not found; run `bias` (Part 3) or pass --bundle")
    return path


def run(command: str, cfg: ModelConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Execute one registry command and return a JSON-serialisable result."""
    registry = build_registry(cfg, dry_run=args.dry_run)
    if command == "champion":
        return {
            "backend": registry.backend.description,
            "champion": registry.get_champion(),
            "history": registry.history(),
        }
    if command == "list-versions":
        rows = [
            {
                k: m.get(k)
                for k in (
                    "version",
                    "status",
                    "model_type",
                    "created_at",
                    "sha256",
                    "local_registry_version",
                    "test_candidate",
                )
            }
            for m in registry.list_versions()
        ]
        return {
            "backend": registry.backend.description,
            "versions": rows,
            "protected_champions": registry.protected_versions(),
        }
    if command == "pull":
        version = args.version or (registry.get_champion() or {}).get("version")
        if not version:
            raise RegistryError("no --version given and no champion set")
        dest = Path(args.dest) if args.dest else cfg.artifacts_subdir("pulled") / version
        root = registry.pull(version, dest)
        model, _, card = load_bundle(root)
        return {"version": version, "path": str(root), "model_type": model.name, "stage": card.get("stage")}
    if command == "rollback":
        return rollback(cfg, registry, args.to_version)
    if command == "protect":
        if args.version:
            copied = registry.protect(args.version)
            outcome = {args.version: "copied" if copied else "already protected"}
        else:
            outcome = registry.backfill_protected()
        return {
            "backend": registry.backend.description,
            "champion_package": registry.champion_package,
            "outcome": outcome,
            "protected_versions": registry.protected_versions(),
        }

    bundle_dir = default_bundle(cfg, args)
    extra = local_registration(cfg, args.local_alias) if args.local_alias else {}
    if command == "push":
        _, _, card = load_bundle(bundle_dir)
        version = args.version or make_version((card.get("data_version") or {}).get("git_commit"))
        with tempfile.TemporaryDirectory() as tmp:
            bundle = create_bundle(bundle_dir, Path(tmp), version, extra)
            meta = registry.push(bundle, extra)
        return {"version": version, "sha256": meta["sha256"], "backend": registry.backend.description}
    # promote
    prep = prepare(cfg, skip_hash_check=args.skip_hash_check, create_split=False)
    return promote(cfg, registry, bundle_dir, prep, extra=extra, version=args.version)
