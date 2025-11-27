"""Detect line terminators in text files.

Scans a file or a directory recursively and reports occurrences of:
- Windows CRLF (\r\n)
- Unix LF-only (\n not preceded by \r)
- Classic Mac CR-only (\r not followed by \n)
- Unicode Line Separator U+2028
- Unicode Paragraph Separator U+2029
- Unicode Next Line U+0085

By default reads files as UTF-8 text. You can override the encoding.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Dict, Iterable, Tuple

import rich_click as click
from rich.console import Console
from rich.table import Table


_CRLF_BYTES = re.compile(rb"\r\n")
_LF_ONLY_BYTES = re.compile(rb"(?<!\r)\n")
_CR_ONLY_BYTES = re.compile(rb"\r(?!\n)")
_LS_RE = re.compile(chr(0x2028))
_PS_RE = re.compile(chr(0x2029))
_NEL_RE = re.compile(chr(0x0085))


def _iter_files(path: Path, glob: str | None, exts: Tuple[str, ...] | None) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    # Directory: choose by extensions or glob
    if exts:
        for ext in exts:
            yield from path.rglob(f"*{ext}")
    else:
        pattern = glob or "**/*"
        yield from path.rglob(pattern)


def _safe_read(p: Path, encoding: str, errors: str) -> tuple[bytes, str | None]:
    """Read raw bytes and best-effort decoded text.

    Returns (data_bytes, text_or_None). Decoding failures yield None for text.
    """
    data = p.read_bytes()
    text: str | None = None
    try:
        text = data.decode(encoding, errors=errors)
    except Exception:
        try:
            text = data.decode(errors=errors)
        except Exception:
            text = None
    return data, text


def _count_terminators(data: bytes, text: str | None) -> Dict[str, int]:
    counts = {
        "CRLF": len(_CRLF_BYTES.findall(data)),
        "LF_only": len(_LF_ONLY_BYTES.findall(data)),
        "CR_only": len(_CR_ONLY_BYTES.findall(data)),
        "LS_U2028": 0,
        "PS_U2029": 0,
        "NEL_U0085": 0,
    }
    if text is not None:
        counts["LS_U2028"] = len(_LS_RE.findall(text))
        counts["PS_U2029"] = len(_PS_RE.findall(text))
        counts["NEL_U0085"] = len(_NEL_RE.findall(text))
    return counts


@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--encoding", "encoding", default="utf-8", show_default=True,
              help="Text encoding used to decode files.")
@click.option("--errors", "errors", default="replace", show_default=True,
              type=click.Choice(["strict", "ignore", "replace"]),
              help="Decoding error handling policy.")
@click.option("--glob", "glob", default=None,
              help="Glob pattern under directories (e.g., **/*.txt). Ignored if --ext is used.")
@click.option("--ext", "exts", multiple=True,
              help="File extensions filter (e.g., --ext .txt --ext .csv). Takes precedence over --glob.")
@click.option("--output", "output", type=click.Path(path_type=Path), default=None,
              help="Optional output file (text). If omitted, prints to console only.")
@click.option("--summary-only", is_flag=True, help="Print only summary totals, omit per-file rows.")
def detect_line_separators(path: Path,
                           encoding: str,
                           errors: str,
                           glob: str | None,
                           exts: Tuple[str, ...],
                           output: Path | None,
                           summary_only: bool) -> None:
    """Detect and report line terminators in PATH (file or directory).

    Examples:
        asrtk detect-line-separators data/ --ext .txt --ext .csv
        asrtk detect-line-separators corpus/ --glob "**/*.vtt"
        asrtk detect-line-separators file.csv
    """
    console = Console()

    files = list(_iter_files(path, glob, exts if exts else None))
    if not files:
        click.echo("No files matched the given criteria.")
        return

    totals = {"CRLF": 0, "LF_only": 0, "CR_only": 0, "LS_U2028": 0, "PS_U2029": 0, "NEL_U0085": 0}
    inline_totals = {"\\r\\n": 0, "\\n": 0, "\\r": 0, "\\t": 0, "<br>": 0, "&#10;": 0, "&#13;": 0}
    per_file_rows = []
    per_file_inline = []

    with click.progressbar(files, label="Scanning files") as bar:
        for p in bar:
            if p.is_dir():
                continue
            try:
                data, text = _safe_read(p, encoding=encoding, errors=errors)
            except Exception as e:
                per_file_rows.append((str(p), "<read error>", 0, 0, 0, 0, 0, 0))
                continue

            counts = _count_terminators(data, text)
            for k, v in counts.items():
                totals[k] += v
            per_file_rows.append((
                str(p),
                "ok",
                counts["CRLF"],
                counts["LF_only"],
                counts["CR_only"],
                counts["LS_U2028"],
                counts["PS_U2029"],
                counts["NEL_U0085"],
            ))

            # Inline code counts (only when we have decoded text)
            inline_counts = {"\\r\\n": 0, "\\n": 0, "\\r": 0, "\\t": 0, "<br>": 0, "&#10;": 0, "&#13;": 0}
            if text is not None:
                # Literal escape sequences
                inline_counts["\\r\\n"] = len(re.findall(r"\\r\\n", text))
                inline_counts["\\n"] = len(re.findall(r"\\n", text))
                inline_counts["\\r"] = len(re.findall(r"\\r", text))
                inline_counts["\\t"] = len(re.findall(r"\\t", text))
                # HTML <br> variants
                inline_counts["<br>"] = len(re.findall(r"(?i)<br\s*/?>", text))
                # HTML entities for LF/CR
                inline_counts["&#10;"] = len(re.findall(r"(?i)&#10;|&#x0a;", text))
                inline_counts["&#13;"] = len(re.findall(r"(?i)&#13;|&#x0d;", text))

            for k, v in inline_counts.items():
                inline_totals[k] += v
            per_file_inline.append((str(p), inline_counts))

    # Prepare output text
    lines = []
    lines.append(f"Scanned files: {len([r for r in per_file_rows if r[1] != '<decode error>'])}/{len(per_file_rows)}")
    lines.append("Totals:")
    lines.append(
        f"  CRLF={totals['CRLF']}, LF_only={totals['LF_only']}, CR_only={totals['CR_only']}, "
        f"LS_U2028={totals['LS_U2028']}, PS_U2029={totals['PS_U2029']}, NEL_U0085={totals['NEL_U0085']}"
    )

    if not summary_only:
        lines.append("")
        lines.append("Per-file:")
        for row in per_file_rows:
            fp, status, crlf, lfo, cro, ls, ps, nel = row
            if status == "<read error>":
                lines.append(f"  {fp}: READ ERROR")
            else:
                lines.append(
                    f"  {fp}: CRLF={crlf}, LF_only={lfo}, CR_only={cro}, LS_U2028={ls}, PS_U2029={ps}, NEL_U0085={nel}"
                )

    text_out = "\n".join(lines)

    # Console table summary
    table = Table(title="Line Terminators Summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for k in ("CRLF", "LF_only", "CR_only", "LS_U2028", "PS_U2029", "NEL_U0085"):
        table.add_row(k, str(totals[k]))

    console.print(table)

    # Inline codes summary table
    inline_table = Table(title="Inline Break Codes Summary")
    inline_table.add_column("Code")
    inline_table.add_column("Count", justify="right")
    for k in ("\\r\\n", "\\n", "\\r", "\\t", "<br>", "&#10;", "&#13;"):
        inline_table.add_row(k, str(inline_totals[k]))
    console.print(inline_table)

    if output:
        try:
            output.write_text(text_out, encoding="utf-8")
            console.print(f"\nSaved report to: [blue]{output}[/blue]")
        except Exception as e:
            console.print(f"[red]Failed to write report:[/red] {e}")
    else:
        # Append inline totals and per-file if not summary-only
        extra_lines = []
        extra_lines.append("\nInline code totals:")
        for k in ("\\r\\n", "\\n", "\\r", "\\t", "<br>", "&#10;", "&#13;"):
            extra_lines.append(f"  {k}={inline_totals[k]}")

        if not summary_only:
            extra_lines.append("\nPer-file inline codes:")
            for fp, ic in per_file_inline:
                rn = ic["\\r\\n"]
                n_ = ic["\\n"]
                r_ = ic["\\r"]
                t_ = ic["\\t"]
                br = ic["<br>"]
                e10 = ic["&#10;"]
                e13 = ic["&#13;"]
                extra_lines.append(
                    f"  {fp}: \\r\\n={rn}, \\n={n_}, \\r={r_}, \\t={t_}, <br>={br}, &#10;={e10}, &#13;={e13}"
                )

        console.print("\n" + text_out + "\n" + "\n".join(extra_lines))
