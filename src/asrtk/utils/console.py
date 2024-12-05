"""Console output utilities."""
from rich.console import Console
from rich.text import Text
from typing import List, Tuple, Optional

def create_highlighted_text(text: str, pattern: str, style: str = "bold red") -> Text:
    """Create a Text object with highlighted pattern.

    Args:
        text: Base text
        pattern: Pattern to highlight
        style: Rich style to apply to pattern

    Returns:
        Rich Text object with highlighted pattern
    """
    result = Text()
    parts = text.split(pattern)

    for i, part in enumerate(parts):
        if part:
            result.append(part)
        if i < len(parts) - 1:
            result.append(pattern, style=style)

    return result

def print_file_header(console: Console, file_path: str) -> None:
    """Print a file header with the filename in blue.

    Args:
        console: Rich console instance
        file_path: Path to file
    """
    console.print(f"\nIn [blue]{file_path}[/blue]:")

# original_line, new_line, old_text, new_text
def print_replacement_example(console: Console,
                            original_line: str,
                            new_line: str,
                            old_text: str,
                            new_text: str) -> None:
    """Print a replacement example with highlighting.

    Args:
        console: Rich console instance
        original_line: Original line
        new_line: New line
        old_text: Old text
        new_text: New text
    """
    indent: int = 4
    console.print(f"{' ' * indent}Original line: {repr(original_line)}")
    console.print(f"{' ' * indent}New line:  {repr(new_line)}")
    console.print(f"{' ' * indent}[dim]{repr(original_line)} → {repr(new_line)}[/dim]\n")
