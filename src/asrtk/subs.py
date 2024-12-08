import torch
import torchaudio
from pydub import AudioSegment
import gc
from typing import List, Tuple
import webvtt
from pprint import pprint

from asrtk.core.text import (
    test_punc,
    sanitize,
    format_time,
    remove_mismatched_characters
)
from asrtk.variables import blacklist

def is_blank_line(text: str) -> bool:
    """Check if a line is effectively blank (empty or just whitespace/special chars).

    Args:
        text: Text line to check

    Returns:
        bool: True if line is blank or effectively blank
    """
    # Strip whitespace and common special characters
    cleaned = text.strip(" \t\n\r\f\v-_.")
    # Consider line blank if empty or very short (1-2 chars)
    return not cleaned or len(cleaned) <= 2

def split_audio_with_subtitles(
    vtt_file,
    audio_file,
    output_folder,
    format="wav",
    tolerance=250,
    max_len=5,
    max_duration=29500,
    max_caption_length=840,
    max_time_length=30,
    period_threshold=8,
    n_samples=25,
    force_merge=False,
    forced_alignment=True,
):
    """
    Splits an audio file into chunks based on subtitles from a VTT file.

    This function processes a VTT file and its corresponding audio file, splits the audio into
    chunks based on the subtitles, and exports each chunk as a separate WAV file along with
    its corresponding subtitle in a VTT file. It includes several checks such as caption length,
    caption duration, and timestamp order to ensure the integrity of the chunks.

    Parameters:
    - vtt_file (str): Path to the VTT subtitle file.
    - audio_file (str): Path to the corresponding audio file.
    - output_folder (str): Folder path where the split audio files and VTT files will be saved.
    - format (str, optional): Output audio format (default: wav).
    - tolerance (int, optional): Additional time in milliseconds added to start and end of each chunk. Defaults to 500.
    - max_len (int, optional): Maximum number of captions to consider for merging. Defaults to 5.
    - max_duration (int, optional): Maximum duration of a chunk in milliseconds. Defaults to 10500.
    - max_caption_length (int, optional): Maximum character length of a caption. Defaults to 840.
    - max_time_length (int, optional): Maximum duration of a caption in seconds. Defaults to 30.

    The function skips over captions that are too long, have unordered timestamps, or are blank.
    It also checks for the number of periods in the first set of captions to determine whether
    to merge captions based on sentence completion.

    No return value.
    """

    import os
    import time
    import webvtt
    from pydub import AudioSegment
    import gc
    from asrtk.core.text import (
        test_punc,
        sanitize,
        format_time,
        remove_mismatched_characters
    )
    from asrtk.variables import blacklist
    from pprint import pprint
    # import numpy as np

    if forced_alignment:
        import io
        import torch
        import torchaudio
        from torchaudio.transforms import Resample
        from asrtk.align import aligner

    # torch.set_num_threads(1)

    silero_model, silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False, trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = silero_utils

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    silent_chunk_folder = os.path.join(output_folder, "silent_chunks")
    if not os.path.exists(silent_chunk_folder):
        os.makedirs(silent_chunk_folder)

    # Try loading with torchaudio, if fails convert to WAV using pydub
    audio = AudioSegment.from_file(audio_file)
    sample_rate = audio.frame_rate

    if forced_alignment:
        # Export the AudioSegment object to a byte stream
        audio_byte_stream = io.BytesIO()
        audio.export(audio_byte_stream, format="wav")
        audio_byte_stream.seek(0)  # Go to the start of the stream
        # Use torchaudio to load the audio data from the byte stream
        audio_pt, sample_rate = torchaudio.load(audio_byte_stream)

        # Check if the waveform is stereo and downmix if needed
        if audio_pt.shape[0] == 2:
            print("Stereo! Downmixing...")
            audio_pt = torch.mean(audio_pt, dim=0, keepdim=True)

        # Resampling if needed
        if sample_rate != 16000:
            print(f"Resampling to 16000 Hz...")
            resampler = Resample(orig_freq=sample_rate, new_freq=16000, lowpass_filter_width=256, rolloff=0.99)
            audio_pt = resampler(audio_pt)
            sample_rate = 16000

        # Calculate audio length
        audio_length_seconds = int(audio_pt.size(-1) / sample_rate)
        audio_len_str = time.strftime("%H:%M:%S", time.gmtime(audio_length_seconds))
        print(f"Audio len: {audio_len_str}.")

    captions = list(webvtt.read(vtt_file))

    i = 0
    i_real = 0
    avg_cps = 0
    cps_list = []
    exported_filelist = []
    merged_captions = []
    # period_threshold = 1000  # Set to a huge number to disable caption merging.
    # Test for the number of periods in the first n captions
    period_count = test_punc(captions, n_samples)
    merge_enabled = (period_count < period_threshold) or force_merge
    print(
        f"Period count: {period_count}/{n_samples}, merge enabled: {merge_enabled}, period threshold: {period_threshold}"
    )
    # time.sleep(3)
    while i < len(captions):
        current_caption = captions[i]

        current_caption.text = sanitize(current_caption.text.strip())

        # Skip if the caption text is too long
        if len(current_caption.text.strip()) > max_caption_length:
            print(f"Caption too long, skipping... [{current_caption.text.strip()}]")
            i += 1
            continue

        # Check the duration of the caption
        caption_duration = current_caption.end_in_seconds - current_caption.start_in_seconds
        if caption_duration > max_time_length:
            print(f"Caption duration too long, skipping... [{current_caption.text.strip()}]")
            i += 1
            continue

        # Check for unordered timestamps
        if i < len(captions) - 1:
            next_caption = captions[i + 1]
            if current_caption.end_in_seconds > next_caption.start_in_seconds:
                print(f"Unordered timestamps, skipping... [{current_caption.text.strip()}]")
                i += 1
                continue

        # Skip ♫
        if "♫" in current_caption.text:
            print("♫♫♫♫, skipping...")
            i += 1
            continue

        # Skip blank captions
        if current_caption.text.strip() == "":
            print("Blank caption, skipping...")
            i += 1
            continue

        if current_caption.text.strip().lower() in blacklist:  # TODO: Use Turkish lower func.
            print(f"Skipping word: {current_caption.text.strip()}")
            i += 1
            continue

        if current_caption.text.strip()[0] == "[" and current_caption.text.strip()[-1] == "]":
            print(f"Skipping {current_caption.text.strip()}")
            i += 1
            continue

        # Check for unordered timestamps
        if i < len(captions) - 1:
            next_caption = captions[i + 1]
            if current_caption.end_in_seconds > next_caption.start_in_seconds:
                print(f"Unordered timestamps, skipping... [{current_caption.text.strip()}]")
                i += 1
                continue

        # Skip blank captions
        if is_blank_line(current_caption.text):
            print(f"Blank/short caption, skipping... [{current_caption.text}]")
            i += 1
            continue

        full_text = sanitize(current_caption.text)

        start_time = current_caption.start_in_seconds * 1000  # Start time in ms
        end_time = current_caption.end_in_seconds * 1000  # End time in ms
        j = i + 1

        while (
            merge_enabled
            and (force_merge or not full_text.endswith((".", "?", "!")))
            and j < len(captions)
            and (end_time - start_time) <= max_duration
        ):
            next_caption = captions[j]

            # End merging if we hit a blank line
            if is_blank_line(next_caption.text):
                print(f"Hit blank line while merging, ending merge... [{next_caption.text}]")
                break  # Stop merging and use what we have so far

            next_text = sanitize(next_caption.text)
            potential_merge = full_text + " " + next_text
            potential_end_time = captions[j].end_in_seconds * 1000

            # Check the duration before actually merging
            if (potential_end_time - start_time) > max_duration:
                break

            if next_caption.start_in_seconds - captions[j - 1].end_in_seconds > 1:
                print('Caption gap too big, skipping...', captions[j - 1].end_in_seconds, next_caption.start_in_seconds, i)
                break

            # Check the character length of the potential merged caption
            if len(potential_merge.strip()) > max_caption_length:
                break

            full_text = sanitize(potential_merge)
            end_time = potential_end_time
            j += 1

        # Skip blank captions
        if current_caption.text.strip() == "":
            print("Blank caption, skipping...")
            i += 1
            continue

        # Add merged caption to the list
        merged_captions.append((start_time, end_time, full_text))

        # Save the merged audio chunk
        start_with_tolerance = max(0, int(start_time - tolerance))  # Ensure start time does not go below 0
        end_with_tolerance = min(len(audio), int(end_time + tolerance))  # Ensure end time does not exceed audio length

        if forced_alignment and full_text.strip() != "":
            print("Using forced alignment...")
            # Convert start and end times from milliseconds to sample indices
            start_sample = int(start_time * sample_rate / 1000)
            end_sample = int(end_time * sample_rate / 1000)

            # Convert tolerance from milliseconds to samples
            tolerance_in_samples = int(tolerance * sample_rate / 1000)

            # Apply tolerance to start and end sample indices
            start_sample_with_tolerance = max(0, start_sample - tolerance_in_samples)
            end_sample_with_tolerance = min(audio_pt.size(1), end_sample + tolerance_in_samples)

            # Extract the audio chunk with the applied tolerance
            waveform = audio_pt[:, start_sample_with_tolerance:end_sample_with_tolerance]

            # Add context manager for VAD
            with torch.no_grad():
                speech_timestamps = get_speech_timestamps(waveform, silero_model, threshold=0.6, sampling_rate=sample_rate)
            silero_model.reset_states()

            if len(speech_timestamps) == 0:
                silent_chunk_name = f"{silent_chunk_folder}/chunk_{i}.{format}"
                print(f"No speech detected, saving to {silent_chunk_name}...")
                # Convert silent chunks to 16-bit too
                waveform_16bit = convert_to_int16(waveform)
                torchaudio.save(
                    silent_chunk_name,
                    waveform_16bit,
                    sample_rate,
                    format=format,
                    encoding='PCM_S',
                    bits_per_sample=16
                )
                # Add garbage collection
                del waveform
                torch.cuda.empty_cache()
                gc.collect()
                i += 1
                continue

            # Trim the waveform to the VAD speech timestamps, we may disable this.
            pprint(f"VAD Speech timestamps: {speech_timestamps}")
            global_start = speech_timestamps[0]['start']
            global_end = speech_timestamps[-1]['end']
            backup_waveform = waveform
            waveform = waveform[:, global_start:global_end]
            # waveform = collect_chunks(speech_timestamps, waveform)

            full_text_4_alignment = full_text.replace("-", " ")
            print(f"Aligning: {full_text_4_alignment}")

            try:
                with torch.no_grad():
                    waveform, token_spans, num_frames, emission, sample_rate, transcript = aligner.do_it(waveform, full_text_4_alignment)
            except Exception as e:
                print(f"Error aligning VAD trimmed waveform. Restoring backup...")
                waveform_16bit = convert_to_int16(waveform)
                torchaudio.save(
                    f"{output_folder}/chunk_{i}_failed_alignment_vad_trim.{format}",
                    waveform_16bit,
                    sample_rate,
                    format=format,
                    encoding='PCM_S',
                    bits_per_sample=16
                )
                # Add garbage collection
                del waveform
                del backup_waveform
                torch.cuda.empty_cache()
                gc.collect()
                i += 1
                continue

            # try:
            #     waveform, token_spans, num_frames, emission, sample_rate, transcript = aligner.do_it(waveform, full_text_4_alignment)
            # except Exception as e:
            #     print(f"Error aligning: {e}")
            #     torchaudio.save(f"{output_folder}/chunk_{i}_failed_alignment.{format}", waveform, sample_rate, format=format)
            #     i += 1
            #     continue

            start_sec, end_sec = aligner.get_sentence_boundaries(waveform, token_spans, num_frames, sample_rate)

            print(f"Aligned: {start_sec:.3f} -> {end_sec:.3f} seconds")

            # Convert time in seconds to sample indices
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)

            # Crop the waveform to these sample indices
            chunk = waveform[:, start_sample:end_sample]
            num_samples = chunk.shape[1]
            duration = num_samples / sample_rate
            start_with_tolerance = start_sec * 1000
            end_with_tolerance = end_sec * 1000
            chunk_name = f"{output_folder}/chunk_{i}.{format}"

            # Convert to 16-bit PCM before saving
            chunk_16bit = convert_to_int16(chunk)
            torchaudio.save(
                chunk_name,
                chunk_16bit,
                sample_rate,
                format=format,
                encoding='PCM_S',  # Specify signed PCM encoding
                bits_per_sample=16  # Explicitly set 16 bits
            )

            # After saving the chunk, add garbage collection
            del chunk
            del waveform
            if 'backup_waveform' in locals():
                del backup_waveform
            torch.cuda.empty_cache()
            gc.collect()
        else:
            chunk_name = f"{output_folder}/chunk_{i}.{format}"
            chunk = audio[start_with_tolerance:end_with_tolerance]
            chunk.export(chunk_name, format=format)
            # Add garbage collection for non-forced alignment path
            del chunk
            gc.collect()

        print(f"Exported Audio: {chunk_name}")

        # Save the corresponding VTT
        vtt_chunk_name = f"{output_folder}/chunk_{i}.vtt"
        with open(vtt_chunk_name, "w") as vtt_chunk:
            end_with_tolerance = max(0, end_with_tolerance - start_with_tolerance)
            full_text = sanitize(full_text)
            vtt_chunk.write("WEBVTT\n\n")
            vtt_chunk.write(f"{format_time(0)} --> {format_time(end_with_tolerance)}\n")
            vtt_chunk.write(full_text + "\n")

        i_real += 1
        cps = len(full_text) / duration
        cps_list.append(cps)
        avg_cps = round(sum(cps_list) / len(cps_list), 1)
        exported_filelist.append([round(cps, 1), f"{vtt_chunk_name}", duration, len(full_text), avg_cps])
        print(f"Exported VTT: {vtt_chunk_name}, duration: {duration}, chars: {len(full_text)}, cps: {cps:.1f}, avg_cps: {avg_cps}")

        # Skip over the captions that were merged
        i = j

    # sort exported_filelist by cps
    exported_filelist.sort(key=lambda x: x[0], reverse=True)
    import csv
    output_file_path = f"{output_folder}/exported_filelist.csv"
    with open(output_file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["cps", "vtt_chunk_name", "duration", "chars", "avg_cps"]
        writer = csv.writer(csvfile, delimiter="\t")

        # Write the header
        writer.writerow(fieldnames)

        # Write the data rows
        writer.writerows(exported_filelist)

    print(f"Exported {i_real} captions to {output_folder}")

def convert_to_int16(waveform: torch.Tensor) -> torch.Tensor:
    """Convert floating point waveform to 16-bit PCM.

    Args:
        waveform: Input waveform tensor (float32)

    Returns:
        16-bit PCM waveform tensor
    """
    # Ensure the input is float32
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    # Normalize to [-1, 1]
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    # Scale to 16-bit range and convert
    waveform = (waveform * 32767).clamp(-32768, 32767).short()

    return waveform
