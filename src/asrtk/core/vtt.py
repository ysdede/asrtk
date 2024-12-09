"""VTT file handling utilities."""
import re
from typing import List, Tuple
from pathlib import Path

# Common VTT header elements
VTT_HEADERS = {'WEBVTT', 'Kind:', 'Language:'}

# Timestamp pattern: matches "00:00:00.000 --> 00:00:00.000" format
TIMESTAMP_PATTERN = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}')

def clean_caption_text(text: str) -> str:
    """Clean caption text by removing unwanted characters and normalizing whitespace.

    Args:
        text: Raw caption text

    Returns:
        str: Cleaned text
    """
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)

    # Remove multiple spaces and normalize whitespace
    text = ' '.join(text.split())

    # Remove leading/trailing whitespace
    return text.strip()

def is_header_line(line: str) -> bool:
    """Check if line is a VTT header line.

    Args:
        line: Line to check

    Returns:
        bool: True if line is a header line
    """
    return any(line.startswith(header) for header in VTT_HEADERS)

def is_timestamp_line(line: str) -> bool:
    """Check if line contains a VTT timestamp.

    Args:
        line: Line to check

    Returns:
        bool: True if line contains a timestamp
    """
    return bool(TIMESTAMP_PATTERN.match(line))

def should_skip_line(line: str) -> bool:
    """Check if line should be skipped during text processing.

    Args:
        line: Line to check

    Returns:
        bool: True if line should be skipped
    """
    # Skip empty lines, headers, timestamps, and line numbers
    return (not line.strip() or
            is_header_line(line) or
            is_timestamp_line(line) or
            line.strip().isdigit())

def process_vtt_content(content: str) -> List[str]:
    """Process VTT content and return subtitle text lines.

    Args:
        content: Raw VTT file content

    Returns:
        List[str]: List of subtitle text lines
    """
    text_lines = []

    for line in content.split('\n'):
        if should_skip_line(line):
            continue
        if line.strip():  # Only add non-empty lines
            clean_line = clean_caption_text(line.strip())
            if clean_line:  # Only add if there's content after cleaning
                text_lines.append(clean_line)

    return text_lines

def read_vtt_file(vtt_file: Path) -> List[str]:
    """Read and process a VTT file.

    Args:
        vtt_file: Path to VTT file

    Returns:
        List[str]: List of subtitle text lines
    """
    content = vtt_file.read_text(encoding='utf-8')
    return process_vtt_content(content)

def split_vtt_into_chunks(content: str) -> List[List[str]]:
    """Split VTT content into chunks (groups of lines between timestamps).

    Args:
        content: Raw VTT file content

    Returns:
        List[List[str]]: List of chunks, where each chunk is a list of lines
    """
    chunks = []
    current_chunk = []

    for line in content.split('\n'):
        # Start new chunk on timestamp
        if is_timestamp_line(line):
            if current_chunk:  # Save previous chunk if exists
                chunks.append(current_chunk)
            current_chunk = [line]  # Start new chunk with timestamp
        elif line.strip():  # Add non-empty lines to current chunk
            current_chunk.append(line)
        elif current_chunk:  # Empty line after chunk
            chunks.append(current_chunk)
            current_chunk = []

    # Add last chunk if exists
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def combine_vtt_chunks(chunks: List[List[str]], preserve_empty_lines: bool = True) -> str:
    """Combine VTT chunks back into a single string.

    Args:
        chunks: List of chunks, where each chunk is a list of lines
        preserve_empty_lines: Whether to preserve empty lines between chunks

    Returns:
        str: Combined VTT content
    """
    combined_lines = []

    for i, chunk in enumerate(chunks):
        # Add chunk lines
        combined_lines.extend(chunk)

        # Add empty line between chunks (except after the last chunk)
        if preserve_empty_lines and i < len(chunks) - 1:
            combined_lines.append('')

    return '\n'.join(combined_lines)
