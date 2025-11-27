from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import rich_click as click


DEFAULT_STEPS = (
    "00_ingest",
    "10_clean",
    "20_normalize",
    "30_enrich",
    "40_split",
)


@click.command()
@click.argument("target", type=click.Path(path_type=Path), required=False)
@click.option("--dataset", "dataset", type=str, default=None, help="Dataset name folder under data paths.")
@click.option("--version", "version", type=str, default=None, help="Processed version folder name (e.g., vYYYY-MM-DD).")
@click.option("--steps", "steps", multiple=True, help="Step folders for data/interim (repeatable).")
@click.option("--force", is_flag=True, help="Create missing folders even if some exist; ignore conflicts.")
@click.option("--with-readme", is_flag=True, help="Create minimal README.md placeholders.")
@click.option("--gitignore", is_flag=True, help="Append data/logs/tmp ignores to .gitignore if present or create one.")
def init_project(target: Path | None, dataset: str | None, version: str | None, steps: tuple[str, ...], force: bool, with_readme: bool, gitignore: bool) -> None:
    """Scaffold a data processing workspace in the current directory or TARGET.

    Examples:
        asrtk init-project
        asrtk init-project --dataset PARROT_v1.0 --version v2025-11-28
        asrtk init-project --steps 00_ingest --steps 10_clean --with-readme --gitignore
    """
    root = target or Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    ds = dataset
    ver = version or f"v{datetime.now().date()}"
    step_list = list(steps) if steps else list(DEFAULT_STEPS)

    def maybe_touch(p: Path) -> None:
        if with_readme:
            f = p / "README.md"
            if not f.exists():
                f.write_text(f"{p.name}\n", encoding="utf-8")

    # Top-level folders
    folders = [
        root / "sources",
        root / "data" / "interim",
        root / "data" / "processed",
        root / "data" / "external",
        root / "data" / "cache",
        root / "data" / "metadata",
        root / "pipelines",
        root / "configs",
        root / "notebooks",
        root / "reports",
        root / "tools",
        root / "logs",
        root / "tmp",
    ]

    for d in folders:
        d.mkdir(parents=True, exist_ok=True)
        maybe_touch(d)

    # Interim step folders
    for step in step_list:
        step_dir = root / "data" / "interim" / step
        if ds:
            step_dir = step_dir / ds
        step_dir.mkdir(parents=True, exist_ok=True)
        maybe_touch(step_dir)

    # Processed version folder
    proc_dir = root / "data" / "processed" / ver
    if ds:
        proc_dir = proc_dir / ds
    proc_dir.mkdir(parents=True, exist_ok=True)
    maybe_touch(proc_dir)

    # Reports and logs per version
    rep_dir = root / "reports" / (ver if ver else "current")
    if ds:
        rep_dir = rep_dir / ds
    rep_dir.mkdir(parents=True, exist_ok=True)

    log_dir = root / "logs" / (ver if ver else "current")
    if ds:
        log_dir = log_dir / ds
    log_dir.mkdir(parents=True, exist_ok=True)

    # Manifest template
    manifest_dir = root / "data" / "metadata" / "processed"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{ver}_manifest.json"
    if not manifest.exists() or force:
        data = {
            "version": ver,
            "dataset": ds or "",
            "generated_at": datetime.now().isoformat(),
            "inputs": [],
            "steps": [],
            "reports": [],
        }
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # .gitignore
    if gitignore:
        gi = root / ".gitignore"
        block = (
            "# asrtk scaffold\n"
            "data/**\n!data/metadata/**\nlogs/**\n.tmp/**\n.cache/**\n.DS_Store\n__pycache__/**\n*.log\n"
        )
        if gi.exists():
            text = gi.read_text(encoding="utf-8", errors="ignore")
            if "# asrtk scaffold" not in text:
                gi.write_text(text.rstrip()+"\n\n"+block, encoding="utf-8")
        else:
            gi.write_text(block, encoding="utf-8")

    click.echo("Project scaffold created.")
