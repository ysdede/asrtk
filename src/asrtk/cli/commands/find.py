"""Commands for finding patterns and content in VTT files."""
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.text import Text
from typing import List, Dict, Any
import re
from collections import defaultdict
import json

from ...core.text import (
    find_sample_sentences,
    has_arabic
)

@click.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.argument("words", nargs=-1, required=True)
@click.option("--output", "-o", type=str, default="word_matches.txt", help="Output text file name")
@click.option("--context", "-c", type=int, default=50, help="Number of characters for context (default: 50)")
@click.option("--ignore-case", "-i", is_flag=True, help="Case insensitive search")
@click.option("--whole-word", "-w", is_flag=True, help="Match whole words only")
@click.option("--regex", "-r", is_flag=True, help="Treat search terms as regular expressions")
def find_words(work_dir: str,
              words: tuple[str, ...],
              output: str,
              context: int,
              ignore_case: bool,
              whole_word: bool,
              regex: bool) -> None:
    """Search for specific words or phrases in VTT files.

    Searches recursively through all VTT files in the directory for the given words
    and outputs matches with context to a file.

    Examples:
        asrtk find-words ./subtitles "hello" "world"
        asrtk find-words ./subtitles "error" -i -w
        asrtk find-words ./subtitles "h[ae]llo" -r
    """
    work_dir = Path(work_dir)

    # Find all VTT files recursively
    vtt_files = list(work_dir.rglob("*.vtt"))

    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Prepare search patterns
    patterns = []
    for word in words:
        if regex:
            pattern = word
        else:
            pattern = re.escape(word)
            if whole_word:
                pattern = fr'\b{pattern}\b'

        try:
            patterns.append(re.compile(pattern, re.IGNORECASE if ignore_case else 0))
        except re.error as e:
            click.echo(f"Invalid pattern '{word}': {e}", err=True)
            return

    # Process files
    matches = []
    total_matches = 0

    with click.progressbar(vtt_files, label='Searching files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')
                file_matches = []

                # Skip VTT header and timestamp lines
                for line in content.split('\n'):
                    if line.strip() and not line.startswith('WEBVTT') and '-->' not in line:
                        for pattern in patterns:
                            for match in pattern.finditer(line):
                                total_matches += 1
                                # Get context around match
                                start = max(0, match.start() - context)
                                end = min(len(line), match.end() + context)

                                match_info = {
                                    'file': str(vtt_file),
                                    'pattern': pattern.pattern,
                                    'matched_text': match.group(),
                                    'context': line[start:end],
                                    'start_pos': start,
                                    'match_pos': match.start() - start,
                                    'match_len': match.end() - match.start()
                                }
                                file_matches.append(match_info)

                if file_matches:
                    matches.extend(file_matches)

            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    if not matches:
        click.echo("No matches found")
        return

    # Save results
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"Found {total_matches} matches in {len(vtt_files)} files\n")
        f.write("Search parameters:\n")
        f.write(f"  Words: {', '.join(words)}\n")
        f.write(f"  Ignore case: {ignore_case}\n")
        f.write(f"  Whole word: {whole_word}\n")
        f.write(f"  Regex: {regex}\n\n")

        current_file = None
        for match in matches:
            # Print filename when it changes
            if current_file != match['file']:
                current_file = match['file']
                f.write(f"\nFile: {current_file}\n")
                f.write("-" * 80 + "\n")

            # Write match with context
            context = match['context']
            pos = match['match_pos']
            length = match['match_len']

            # Highlight the match using brackets
            highlighted = (
                context[:pos] +
                '[' + context[pos:pos+length] + ']' +
                context[pos+length:]
            )

            f.write(f"Match: {match['pattern']}\n")
            f.write(f"Context: {highlighted}\n\n")

    click.echo(f"\nFound {total_matches} matches")
    click.echo(f"Results saved to: {output_file}")

@click.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="arabic_subs.txt", help="Output text file name")
@click.option("--batch", "-b", is_flag=True, help="Create batch file for deletion")
def find_arabic(work_dir: str, output: str, batch: bool) -> None:
    """Find VTT files containing Arabic characters.

    Creates a text file listing all VTT files that contain Arabic text.
    With --batch flag, creates a batch file with del commands.
    """
    work_dir = Path(work_dir)

    # Find all VTT files recursively
    vtt_files = list(work_dir.rglob("*.vtt"))

    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Process files
    arabic_files = []

    with click.progressbar(vtt_files, label='Scanning files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')
                if has_arabic(content):
                    arabic_files.append(vtt_file)
            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    if not arabic_files:
        click.echo("No VTT files with Arabic characters found")
        return

    # Save results
    output_file = Path(output)
    batch_file = output_file.with_suffix('.bat') if batch else None

    with output_file.open('w', encoding='utf-8') as f:
        for file in arabic_files:
            f.write(f"{file}\n")

    if batch:
        with batch_file.open('w', encoding='utf-8') as f:
            f.write("@echo off\n")
            f.write("echo Deleting VTT files containing Arabic characters...\n")
            for file in arabic_files:
                # Use quotes to handle paths with spaces
                f.write(f'del "{file}"\n')
            f.write("echo Done.\n")
            f.write("pause\n")

    click.echo(f"\nFound {len(arabic_files)} files containing Arabic characters")
    click.echo(f"File list saved to: {output_file}")
    if batch:
        click.echo(f"Batch file saved to: {batch_file}")

@click.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="pattern_matches.txt", help="Output text file name")
@click.option("--context", "-c", type=int, default=50, help="Number of characters for context (default: 50)")
@click.option("--pattern", "-p", multiple=True, help="Custom regex patterns to search for")
@click.option("--numbers", "-n", is_flag=True, help="Search for potentially incorrect number formats")
def find_patterns(work_dir: str, output: str, context: int, pattern: tuple[str, ...], numbers: bool) -> None:
    """Search for specific patterns in VTT files.

    Specialized pattern matching for finding formatting issues like incorrect number formats.
    Particularly useful for finding English-style numbers in Turkish text (comma as thousands separator).

    Built-in patterns when using --numbers:
    - Numbers with comma followed by 3+ digits (likely English format)
    - Numbers with multiple commas (e.g., 1,234,567)

    Examples:
        # Find potentially incorrect number formats
        asrtk find-patterns ./subtitles --numbers

        # Search with custom pattern
        asrtk find-patterns ./subtitles -p "\d+,\d{3,}"

        # Multiple patterns
        asrtk find-patterns ./subtitles -p "\d+,\d{3,}" -p "\d+\.\d{3,}"
    """
    work_dir = Path(work_dir)

    # Predefined patterns for number formats
    number_patterns = [
        (r"\d+,\d{3,}", "Number with comma followed by 3+ digits (possible English format)"),
        (r"\d+(?:,\d+){2,}", "Number with multiple commas (e.g., 1,234,567)"),
        (r"(?<!\d)[.,]\d+", "Number starting with decimal point/comma"),
        (r"\d{1,3}(?:,\d{3})+(?!\d)", "Standard English number format (e.g., 1,234 or 1,234,567)"),
        (r"\d+,\d+,", "Number with consecutive commas")  # Catch malformed numbers
    ]

    # Compile all patterns
    compiled_patterns = []

    # Add custom patterns
    for p in pattern:
        try:
            compiled_patterns.append((re.compile(p), f"Custom pattern: {p}"))
        except re.error as e:
            click.echo(f"Invalid pattern '{p}': {e}", err=True)
            return

    # Add number patterns if requested
    if numbers:
        for p, desc in number_patterns:
            compiled_patterns.append((re.compile(p), desc))

    if not compiled_patterns:
        click.echo("No patterns specified. Use --numbers or provide patterns with -p")
        return

    # Find all VTT files
    vtt_files = list(work_dir.rglob("*.vtt"))
    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Process files
    matches = []
    total_matches = 0
    console = Console()

    with click.progressbar(vtt_files, label='Searching files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')
                file_matches = []

                # Skip VTT header and timestamp lines
                for line in content.split('\n'):
                    if line.strip() and not line.startswith('WEBVTT') and '-->' not in line:
                        for pattern, desc in compiled_patterns:
                            for match in pattern.finditer(line):
                                total_matches += 1
                                # Get context around match
                                start = max(0, match.start() - context)
                                end = min(len(line), match.end() + context)

                                match_info = {
                                    'file': str(vtt_file),
                                    'pattern': pattern.pattern,
                                    'description': desc,
                                    'matched_text': match.group(),
                                    'context': line[start:end],
                                    'start_pos': start,
                                    'match_pos': match.start() - start,
                                    'match_len': match.end() - match.start()
                                }
                                file_matches.append(match_info)

                if file_matches:
                    matches.extend(file_matches)

            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    if not matches:
        click.echo("No matches found")
        return

    # Save results
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"Found {total_matches} matches in {len(vtt_files)} files\n")
        f.write("Search patterns:\n")
        for _, desc in compiled_patterns:
            f.write(f"  - {desc}\n")
        f.write("\n")

        current_file = None
        for match in matches:
            # Print filename when it changes
            if current_file != match['file']:
                current_file = match['file']
                f.write(f"\nFile: {current_file}\n")
                f.write("-" * 80 + "\n")

            # Write match with context
            context = match['context']
            pos = match['match_pos']
            length = match['match_len']

            # Highlight the match using brackets
            highlighted = (
                context[:pos] +
                '[' + context[pos:pos+length] + ']' +
                context[pos+length:]
            )

            f.write(f"Pattern: {match['pattern']} ({match['description']})\n")
            f.write(f"Found: {match['matched_text']}\n")
            f.write(f"Context: {highlighted}\n\n")

    # Display summary with rich formatting
    console.print(f"\nFound [bold]{total_matches}[/bold] matches")
    console.print(f"Results saved to: [blue]{output_file}[/blue]")

    # Show sample of matches
    if matches:
        console.print("\nSample matches:")
        for match in matches[:5]:
            text = Text()
            context = match['context']
            pos = match['match_pos']
            length = match['match_len']

            text.append(context[:pos])
            text.append(context[pos:pos+length], style="bold red")
            text.append(context[pos+length:])

            console.print(f"  [cyan]{match['matched_text']}[/cyan] in context:")
            console.print(f"  {text}")

@click.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="brackets.json", help="Output JSON file name")
@click.option("--min-frequency", "-f", type=int, default=1, help="Minimum frequency to include (default: 1)")
@click.option("--ignore-case", "-i", is_flag=True, help="Case insensitive counting")
@click.option("--show-context", "-c", is_flag=True, help="Include sample contexts in output")
def find_brackets(work_dir: str, output: str, min_frequency: int, ignore_case: bool, show_context: bool) -> None:
    """Find and analyze text within brackets and parentheses in VTT files.

    Creates a frequency list of text found within () and [] with their occurrences.
    Useful for finding annotations, sound effects, or other bracketed content.

    Examples:
        # Basic usage
        asrtk find-brackets ./subtitles

        # Case insensitive with minimum frequency
        asrtk find-brackets ./subtitles -i -f 2

        # Include context samples
        asrtk find-brackets ./subtitles -c
    """
    work_dir = Path(work_dir)

    # Compile regex patterns
    patterns = {
        'parentheses': (r'\((.*?)\)', 'Round brackets ()'),
        'square': (r'\[(.*?)\]', 'Square brackets []'),
        'asterisk': (r'\*(.*?)\*', 'Asterisk enclosure **'),
        'slash': (r'/(.*?)/', 'Forward slash enclosure //'),
        'dash': (r'--(.+?)--', 'Double dash enclosure --')  # Using .+ to ensure at least one char between dashes
    }
    compiled_patterns = {
        name: (re.compile(pattern), desc)
        for name, (pattern, desc) in patterns.items()
    }

    # Initialize counters and context storage
    bracket_counts = defaultdict(lambda: defaultdict(int))
    bracket_contexts = defaultdict(lambda: defaultdict(list))
    total_matches = 0

    # Find all VTT files
    vtt_files = list(work_dir.rglob("*.vtt"))
    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Process files
    with click.progressbar(vtt_files, label='Processing files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')

                # Process each line
                for line in content.split('\n'):
                    if line.strip() and not line.startswith('WEBVTT') and '-->' not in line:
                        # Check each pattern
                        for name, (pattern, _) in compiled_patterns.items():
                            matches = pattern.finditer(line)
                            for match in matches:
                                text = match.group(1).strip()  # Get text inside brackets
                                if not text:
                                    continue

                                if ignore_case:
                                    text = text.lower()

                                bracket_counts[name][text] += 1
                                total_matches += 1

                                # Store context if needed (limit to 3 examples per text)
                                if show_context and len(bracket_contexts[name][text]) < 3:
                                    # Get some context around the match
                                    start = max(0, match.start() - 30)
                                    end = min(len(line), match.end() + 30)
                                    context = line[start:end].strip()
                                    if context not in bracket_contexts[name][text]:
                                        bracket_contexts[name][text].append(context)

            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    # Prepare output data
    output_data = {
        "stats": {
            "total_matches": total_matches,
            "min_frequency": min_frequency,
            "ignore_case": ignore_case,
            "files_processed": len(vtt_files)
        },
        "brackets": {}
    }

    # Process each bracket type
    for bracket_type, counts in bracket_counts.items():
        entries = []
        for text, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if count >= min_frequency:
                entry = {
                    "text": text,
                    "frequency": count,
                    "percentage": (count / total_matches) * 100
                }
                if show_context and text in bracket_contexts[bracket_type]:
                    entry["samples"] = bracket_contexts[bracket_type][text]
                entries.append(entry)

        output_data["brackets"][bracket_type] = {
            "description": compiled_patterns[bracket_type][1],
            "total": sum(counts.values()),
            "unique": len(entries),
            "entries": entries
        }

    # Save to JSON file
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Display summary
    console = Console()
    console.print(f"\nFound [bold]{total_matches}[/bold] total bracketed texts")

    for bracket_type, data in output_data["brackets"].items():
        console.print(f"\n[blue]{data['description']}[/blue]:")
        console.print(f"  Total occurrences: {data['total']}")
        console.print(f"  Unique texts: {data['unique']}")

        # Show top 5 most frequent
        console.print("\n  Most frequent:")
        for entry in data['entries'][:5]:
            console.print(f"    [cyan]{entry['text']}[/cyan]: {entry['frequency']} times")
            if show_context and "samples" in entry:
                for sample in entry['samples']:
                    text = Text(sample)
                    text.highlight_words([entry['text']], style="bold red")
                    console.print(f"      {text}")

    console.print(f"\nFull results saved to: [blue]{output_file}[/blue]")
