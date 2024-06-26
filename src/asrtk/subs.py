def split_audio_with_subtitles(
    vtt_file,
    audio_file,
    output_folder,
    tolerance=250,
    max_len=5,
    # max_duration=11000,  # Merging time limit. Captions will be merged up to this duration.
    max_duration=29500,  # Merging time limit. Captions will be merged up to this duration.
    max_caption_length=840,
    max_time_length=30,  # Hard limit, just skips longer captions.
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
    from asrtk.utils import test_punc, sanitize, format_time, remove_mismatched_characters
    from asrtk.variables import blacklist
    # import numpy as np

    if forced_alignment:
        import io
        import torch
        import torchaudio
        from torchaudio.transforms import Resample
        from asrtk.align import aligner

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

        format = "flac"
        chunk_name = f"{output_folder}/chunk_{i}.{format}"

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
            full_text_4_alignment = full_text.replace("-", " ")
            waveform, token_spans, num_frames, emission, sample_rate, transcript = aligner.do_it(waveform, full_text_4_alignment)
            start_sec, end_sec = aligner.get_sentence_boundaries(waveform, token_spans, num_frames, sample_rate)

            print(f"Aligned: {start_sec:.3f} -> {end_sec:.3f} seconds")

            # Convert time in seconds to sample indices
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)

            # Crop the waveform to these sample indices
            chunk = waveform[:, start_sample:end_sample]
            start_with_tolerance = start_sec * 1000
            end_with_tolerance = end_sec * 1000
            torchaudio.save(chunk_name, chunk, sample_rate, format=format)
        else:
            chunk = audio[start_with_tolerance:end_with_tolerance]
            chunk.export(chunk_name, format=format)

        print(f"Exported Audio: {chunk_name}")

        # Save the corresponding VTT
        vtt_chunk_name = f"{output_folder}/chunk_{i}.vtt"
        with open(vtt_chunk_name, "w") as vtt_chunk:
            end_with_tolerance = max(0, end_with_tolerance - start_with_tolerance)
            full_text = sanitize(full_text)
            vtt_chunk.write("WEBVTT\n\n")
            vtt_chunk.write(f"{format_time(0)} --> {format_time(end_with_tolerance)}\n")
            vtt_chunk.write(full_text + "\n")
        print(f"Exported VTT: {vtt_chunk_name}")

        # Skip over the captions that were merged
        i = j
