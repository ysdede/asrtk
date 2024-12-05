"""VTT file handling utilities."""
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import webvtt

def clean_caption_text(text: str) -> str:
    """Clean caption text by:
    1. Strip whitespace
    2. Remove trailing hyphens
    3. Strip again
    4. Normalize spaces

    Args:
        text: Raw caption text
    Returns:
        Cleaned text
    """
    text = text.strip()
    while text.endswith('-'):
        text = text[:-1]
    text = text.strip()
    # Replace multiple spaces with single space
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

def split_vtt_into_chunks(content: str, chunk_size: int) -> List[str]:
    """Split VTT content into chunks while preserving VTT structure and subtitle blocks.

    Args:
        content: VTT file content
        chunk_size: Maximum size of each chunk in tokens

    Returns:
        List of VTT content chunks
    """
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    subtitle_block = []
    in_subtitle = False

    # Always include VTT header in each chunk
    header = lines[0] if lines and lines[0].strip().upper() == 'WEBVTT' else 'WEBVTT\n'

    for line in lines[1:]:  # Skip WEBVTT header
        # Start of a subtitle block (timestamp line)
        if '-->' in line:
            in_subtitle = True
            if subtitle_block:
                subtitle_block.append('')  # Add blank line between subtitles
            subtitle_block = [line]
            continue

        if in_subtitle:
            if not line.strip():  # End of subtitle block
                in_subtitle = False
                block_size = sum(len(l.split()) for l in subtitle_block)

                # Check if adding this block would exceed chunk size
                if current_size + block_size > chunk_size and current_chunk:
                    # Complete current chunk
                    chunks.append('\n'.join([header] + current_chunk))
                    current_chunk = []
                    current_size = 0

                current_chunk.extend(subtitle_block)
                current_size += block_size
                subtitle_block = []
            else:
                subtitle_block.append(line)

    # Add any remaining subtitle block
    if subtitle_block:
        current_chunk.extend(subtitle_block)

    # Add any remaining content
    if current_chunk:
        chunks.append('\n'.join([header] + current_chunk))

    return chunks or [content]  # Return original content if no chunks were created

def combine_vtt_chunks(chunks: List[str]) -> str:
    """Combine VTT chunks while handling overlapping timestamps.

    Args:
        chunks: List of VTT content chunks

    Returns:
        Combined VTT content
    """
    # Remove WEBVTT header from all but first chunk
    result = chunks[0]
    for chunk in chunks[1:]:
        # Remove header and any leading blank lines
        lines = chunk.split('\n')
        while lines and (not lines[0].strip() or lines[0].strip().upper() == 'WEBVTT'):
            lines.pop(0)
        result += '\n' + '\n'.join(lines)
    return result

def read_vtt_file(vtt_file: Path) -> List[Tuple[str, str, str]]:
    """Read a VTT file and return a list of (start_time, end_time, text) tuples.

    Args:
        vtt_file: Path to VTT file

    Returns:
        List of (start_time, end_time, text) tuples
    """
    captions = webvtt.read(str(vtt_file))
    return [(c.start, c.end, clean_caption_text(c.text)) for c in captions]

def write_vtt_file(vtt_file: Path, captions: List[Tuple[str, str, str]]) -> None:
    """Write captions to a VTT file.

    Args:
        vtt_file: Path to output VTT file
        captions: List of (start_time, end_time, text) tuples
    """
    content = ["WEBVTT\n"]
    for start, end, text in captions:
        content.extend([
            f"\n{start} --> {end}",
            text,
            ""
        ])
    vtt_file.write_text('\n'.join(content), encoding='utf-8')
