"""`openhup-cli` - lint, simulate, and inspect without a running server.

The point of this tool is that the two most useful safety features - compile-time linting and
replay simulation - work offline. You can check a skill file in CI, or on a laptop, before it ever
touches a deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from openhup_schemas import (
    BUILTIN_DETECTORS,
    BUILTIN_METRICS,
    Anchor,
    Camera,
    Personality,
    load_skills_yaml,
)

from .skills.compile import compile_skill
from .skills.parse import describe


def _load_anchors(path: str | None) -> dict[str, Anchor]:
    """Anchors from a cameras.yaml, so lint can check anchor references and baselines."""
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return {a["id"]: Anchor.model_validate(a) for a in raw.get("anchors", [])}


def _skill_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for entry in paths:
        candidate = Path(entry)
        if candidate.is_dir():
            files.extend(sorted(candidate.glob("*.yaml")))
        else:
            files.append(candidate)
    return files


def cmd_lint(args: argparse.Namespace) -> int:
    """Compile skill files and report findings. Exit 1 on any error."""
    anchors = _load_anchors(args.cameras)
    total_errors = 0
    total_warnings = 0
    checked = 0
    skipped: list[str] = []

    for path in _skill_files(args.paths):
        raw = path.read_text()
        # A skills directory may legitimately hold other things - goals.yaml, notes. Skip anything
        # that is not a skill document rather than reporting it as a broken skill.
        if not _looks_like_skills(raw):
            skipped.append(path.name)
            continue

        try:
            skills = load_skills_yaml(raw)
        except Exception as exc:
            print(f"{path}: FAILED to parse: {exc}")
            total_errors += 1
            continue

        for skill in skills:
            checked += 1
            compiled = compile_skill(skill, anchors=anchors or None, strict=False)
            errors = [w for w in compiled.warnings if w.error]
            warnings = [w for w in compiled.warnings if not w.error]
            total_errors += len(errors)
            total_warnings += len(warnings)

            status = "ERROR" if errors else ("warn" if warnings else "ok")
            print(f"{path.name}: {skill.id} [{status}]")
            if args.verbose or errors or warnings:
                print(f"    {describe(skill)}")
            for finding in errors + warnings:
                marker = "  error" if finding.error else "  warn "
                print(f"  {marker} [{finding.code}] {finding.message}")

    print(f"\n{checked} skill(s): {total_errors} error(s), {total_warnings} warning(s)")
    if skipped:
        print(f"skipped (not skill files): {', '.join(sorted(skipped))}")
    if not anchors:
        print("note: no --cameras given, so anchor and baseline checks were skipped")
    return 1 if total_errors else 0


def _looks_like_skills(text: str) -> bool:
    """Does this YAML contain skill documents?

    A skill is a mapping with `watch` and `conditions`. Checked structurally rather than by
    filename, so `my-kitchen-rules.yaml` works and `goals.yaml` is left alone.
    """
    try:
        documents = [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    except Exception:
        return True  # let the real parser report the error properly
    return any("watch" in d and "conditions" in d for d in documents)


def cmd_simulate(args: argparse.Namespace) -> int:
    """Replay a JSON observation file against a skill file. No server, no database."""
    from openhup_schemas import Observation

    from .skills.simulate import simulate, suggest_thresholds

    anchors = _load_anchors(args.cameras)
    skills = load_skills_yaml(Path(args.skill).read_text())
    if not skills:
        print("no skill found in that file")
        return 1

    raw = json.loads(Path(args.observations).read_text())
    observations = [Observation.model_validate(entry) for entry in raw]

    for skill in skills:
        compiled = compile_skill(skill, anchors=anchors or None, strict=False)
        result = simulate(compiled, observations, anchor_id=args.anchor)
        print(f"\n{skill.id}")
        print(f"  {result.verdict_line()}")
        print(
            f"  observations={result.observations_seen} tasks={result.tasks_created} "
            f"alerts={result.alerts_raised} auto-resolved={result.tasks_auto_resolved} "
            f"suppressed={result.suppressions}"
        )
        for line in suggest_thresholds(result, compiled):
            print(f"  advice: {line}")
    return 0


def cmd_detectors(args: argparse.Namespace) -> int:
    """List detectors, their signals, and their costs."""
    for spec in BUILTIN_DETECTORS.detectors:
        flags = []
        if spec.optional:
            flags.append("opt-in")
        if spec.requires_baseline:
            flags.append("needs baseline")
        if spec.dynamic:
            flags.append(f"dynamic {spec.dynamic_kind}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{spec.name}  ({spec.cost}){suffix}")
        print(f"  {spec.description}")
        for signal in spec.signals:
            print(f"    - {signal.key}: {signal.kind}  {signal.description}")
        for param in spec.params:
            required = " (required)" if param.required else ""
            print(f"    param {param.name}: {param.type}{required}")
        print()
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    for name, description in BUILTIN_METRICS.items():
        print(f"{name:38} {description}")
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Validate config.yaml, cameras.yaml, and personalities.yaml without starting anything.

    Worth running before a restart: the alternative is finding out from a crash loop.
    """
    from .core.config import Settings

    problems: list[str] = []

    if args.vision:
        # The vision service's schema lives in the vision-service package, which the backend
        # deliberately does not depend on. Validate structurally instead: every key the service
        # requires, and none of the obvious mistakes a typo would otherwise hide until boot.
        try:
            raw = yaml.safe_load(Path(args.vision).read_text()) or {}
            if not isinstance(raw, dict):
                raise ValueError("expected a mapping")
            required = {"node_id", "backend_url", "bus", "inference", "snapshots", "sampling"}
            missing = required - raw.keys()
            if missing:
                raise ValueError(f"missing keys: {', '.join(sorted(missing))}")
            known = {
                "node_id",
                "backend_url",
                "api_token_env",
                "bus",
                "inference",
                "snapshots",
                "sampling",
                "agent",
                "mqtt",
                "cameras",
                "anchors",
                "plan_refresh",
                "log_level",
                "dry_run",
            }
            unknown = raw.keys() - known
            if unknown:
                raise ValueError(f"unknown keys: {', '.join(sorted(unknown))}")
            print(f"vision    : ok (node {raw['node_id']})")
        except Exception as exc:
            problems.append(f"vision: {exc}")

    if args.config:
        try:
            settings = Settings.load(args.config)
            print(f"config    : ok ({settings.instance_name})")
            for warning in settings.warnings():
                print(f"  note: {warning}")
        except Exception as exc:
            problems.append(f"config: {exc}")

    if args.cameras:
        try:
            raw = yaml.safe_load(Path(args.cameras).read_text()) or {}
            cameras = [Camera.model_validate(c) for c in raw.get("cameras", [])]
            anchors = [Anchor.model_validate(a) for a in raw.get("anchors", [])]
            print(f"cameras   : ok ({len(cameras)} camera(s), {len(anchors)} anchor(s))")

            known = {c.id for c in cameras}
            for anchor in anchors:
                if anchor.camera_id not in known:
                    problems.append(
                        f"anchor {anchor.id} references unknown camera {anchor.camera_id!r}"
                    )
                if not anchor.polygon:
                    print(
                        f"  note: anchor {anchor.id} has no polygon, so it watches the whole frame"
                    )
            for camera in cameras:
                if camera.kind == "rtsp" and not camera.substream_url:
                    print(
                        f"  note: camera {camera.id} has no substream_url - detection will decode "
                        f"the main stream, which is the usual cause of high CPU"
                    )
        except Exception as exc:
            problems.append(f"cameras: {exc}")

    if args.personalities:
        try:
            entries = yaml.safe_load(Path(args.personalities).read_text()) or []
            personalities = [Personality.model_validate(e) for e in entries]
            print(
                f"voices    : ok ({len(personalities)}: {', '.join(p.id for p in personalities)})"
            )
        except Exception as exc:
            problems.append(f"personalities: {exc}")

    for problem in problems:
        print(f"ERROR {problem}", file=sys.stderr)
    return 1 if problems else 0


def cmd_setup(args: argparse.Namespace) -> int:
    """First-run wizard - the whole onboarding, in one command.

    Bootstraps config/ from the shipped examples, generates deploy/env/openhup.env with real
    secrets, asks the voice (pick or gamble) and the AI provider, then guides the commands that
    need a second terminal, waiting for confirmation after each one. The personality answer is
    written to config.yaml and then never announced again (ADR-014). Re-running is safe: it
    merges over the existing config and never overwrites existing files.
    """
    from .setup import run_setup

    root = Path(args.cwd or Path(__file__).resolve().parents[2])
    run_setup(
        Path(args.config) if Path(args.config).is_absolute() else root / args.config,
        presets_path=Path(args.personalities) if args.personalities else None,
        env_path=Path(args.env) if args.env else None,
        root=root,
        confirm=input,
    )
    return 0


def cmd_export_schemas(args: argparse.Namespace) -> int:
    """Write JSON Schema for the public models, for frontend type generation."""
    from openhup_schemas import Observation, Personality, Skill, Task

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    for model in (Skill, Observation, Task, Personality, Camera, Anchor):
        target = destination / f"{model.__name__.lower()}.json"
        target.write_text(json.dumps(model.model_json_schema(), indent=2))
        print(f"wrote {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openhup-cli",
        description="Lint, simulate, and inspect OpenHup configuration offline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    lint = sub.add_parser("lint", help="compile skill files and report problems")
    lint.add_argument("paths", nargs="+", help="skill YAML files or directories")
    lint.add_argument("--cameras", help="cameras.yaml, to check anchors and baselines")
    lint.add_argument("-v", "--verbose", action="store_true")
    lint.set_defaults(func=cmd_lint)

    sim = sub.add_parser("simulate", help="replay observations against a skill")
    sim.add_argument("skill", help="skill YAML file")
    sim.add_argument("observations", help="JSON array of observations")
    sim.add_argument("--cameras")
    sim.add_argument("--anchor")
    sim.set_defaults(func=cmd_simulate)

    sub.add_parser("detectors", help="list detectors and their signals").set_defaults(
        func=cmd_detectors
    )
    sub.add_parser("metrics", help="list built-in metrics").set_defaults(func=cmd_metrics)

    check = sub.add_parser("check-config", help="validate configuration files")
    check.add_argument("--config", default="config/config.yaml")
    check.add_argument("--cameras", default="config/cameras.yaml")
    check.add_argument("--personalities", default="config/personalities.yaml")
    check.add_argument("--vision", default=None, help="vision.yaml, to check its top-level shape")
    check.set_defaults(func=cmd_validate_config)

    export = sub.add_parser("export-schemas", help="write JSON Schema for the public models")
    export.add_argument("--out", default="packages/openhup-schemas/jsonschema")
    export.set_defaults(func=cmd_export_schemas)

    setup = sub.add_parser(
        "setup",
        help="first-run wizard: bootstrap config, secrets, voice, AI provider, guided handoff",
    )
    setup.add_argument(
        "--cwd",
        default=None,
        help="repo root (default: detected from this checkout)",
    )
    setup.add_argument("--config", default="config/config.yaml")
    setup.add_argument(
        "--personalities",
        default=None,
        help="presets file to read the voice descriptions from (defaults to the shipped one)",
    )
    setup.add_argument(
        "--env",
        default=None,
        help="environment file to write API keys into (default: deploy/env/openhup.env)",
    )
    setup.set_defaults(func=cmd_setup)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["main"]
