# location: asrtk/align/aligner.py
print(__name__)
import torch
import torchaudio
from torchaudio.transforms import Resample
from typing import List
import numpy as np
import IPython
import matplotlib.pyplot as plt

import IPython.display

print(torch.__version__)
print(torchaudio.__version__)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
print(f'Device forced to {device} for MMS_FA model')
# torch.set_num_threads(10)

from torchaudio.pipelines import MMS_FA as bundle
from ..core.text import romanize_turkish
from asrtk.normalizer import Normalizer
normalizer = Normalizer()

model = bundle.get_model()
model.to(device)

tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans

# Compute average score weighted by the span length
def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start, t_spans[-1].end
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    fig.tight_layout()


def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    # print(f"{waveform.size(1)=} / {num_frames=} = {ratio=}")
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    # print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)

# Define a function to read VTT files
def read_vtt_as_text(vtt_file_path):
    import webvtt
    captions = webvtt.read(vtt_file_path)
    return ' '.join(caption.text.strip() for caption in captions)

def get_sentence_boundaries(waveform, token_spans, num_frames, sample_rate):
    # Calculate the ratio to convert frame indices to time (seconds)
    ratio = waveform.size(1) / num_frames

    # Use the first token span of the first word to get the start time
    start_frame_index = token_spans[0][0].start
    # Use the last token span of the last word to get the end time
    end_frame_index = token_spans[-1][-1].end

    # Convert the start and end frame indices to time in seconds
    start_time_sec = start_frame_index * ratio / sample_rate
    end_time_sec = end_frame_index * ratio / sample_rate

    return start_time_sec, end_time_sec


def do_it(audio, text, sample_rate=bundle.sample_rate):
    if not text:
        print('No text')
        return

    text_normalized = normalizer.convert_numbers_to_words(text)  # TODO We have a new normalizer
    text_normalized = romanize_turkish(text_normalized)

    if torch.is_tensor(audio):
        waveform = audio
    elif isinstance(audio, list) or isinstance(audio, np.ndarray):
        # Convert list or numpy array to PyTorch tensor
        waveform = torch.tensor(audio, dtype=torch.float32)
    elif isinstance(audio, str):
            waveform, sample_rate = torchaudio.load(audio)
    elif not torch.is_tensor(audio):
        raise TypeError("Audio must be a PyTorch tensor, a list, or a numpy array")

    if waveform.shape[0] == 2:  # Check if the waveform is stereo
        print('Stereo! down-mixing to mono...')
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != bundle.sample_rate:
        resampler = Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)
        sample_rate = bundle.sample_rate  # Güncellenmiş örneklenme hızını kullanın

    # print(f'{waveform.shape=}')  # Debug: Print waveform shape

    waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform  # Add a batch dimension if it's missing

    # Ensure sample rate matches expected rate
    assert sample_rate == bundle.sample_rate

    # Tokenize transcript
    transcript = text_normalized.split()
    # print('transcript:', transcript)
    tokens = tokenizer(transcript)

    try:
        emission, token_spans = compute_alignments(waveform, transcript)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Transcript: {transcript}")
        raise

    num_frames = emission.size(1)

    return waveform, token_spans, num_frames, emission, sample_rate, transcript


def save_clip(cropped_file_path, cropped_waveform, sample_rate):
    torchaudio.save(cropped_file_path, cropped_waveform, sample_rate)

def find_word_timings(word, waveform, transcript, token_spans, sample_rate, num_frames):
    """
    Calculate the ratio to convert frame indices to time (seconds)

    # Example Usage
    word = "sonucu"
    start_time, end_time = find_word_timings(word, transcript, token_spans, sample_rate, num_frames)

    if start_time is not None and end_time is not None:
        print(f"Word '{word}' timings: {start_time:.3f} - {end_time:.3f} sec")
    else:
        print(f"Word '{word}' not found in the transcript.")

    """
    ratio = waveform.size(1) / num_frames

    # Reverse iterate through the transcript to find the word
    for i in reversed(range(len(transcript))):
        if transcript[i] == word:
            # Found the word, now get its spans
            spans = token_spans[i]
            x0 = int(ratio * spans[0].start)
            x1 = int(ratio * spans[-1].end)
            return x0, x1

    # Word not found
    return None, None

def trim_last_word(word, text, audio_clip):

    word = romanize_turkish(word)

    if text:
        waveform, token_spans, num_frames, emission, sample_rate, transcript = do_it(audio_clip, text)

    start_frame, end_frame = find_word_timings(word, waveform, transcript, token_spans, sample_rate, num_frames)

    if start_frame is not None and end_frame is not None:
        start_time = start_frame / sample_rate
        end_time = end_frame / sample_rate

        print(f"Word '{word}' timings: {start_time:.3f} - {end_time:.3f} sec. {start_frame} - {end_frame} frames.")

        # Trim the original audio_clip instead of the processed waveform
        # Ensure audio_clip is a 1D numpy array
        if isinstance(audio_clip, np.ndarray) and audio_clip.ndim == 1:
            trimmed_audio = audio_clip[:start_frame - 400]
            return trimmed_audio
        else:
            print("trim_last_word: Invalid audio_clip format. Expected a 1D numpy array.")
            return None
    else:
        print(f'trim_last_word: Word "{word}" not found in the transcript.')
        return None

def trim_last_word_cuda(word, text, audio_clip_tensor):
    """
    Trims the last occurrence of the specified word from the audio_clip_tensor.
    Assumes audio_clip_tensor is a CUDA tensor.

    Parameters:
    - word: The word to trim from the end of the audio.
    - text: Full text corresponding to the audio clip for forced alignment.
    - audio_clip_tensor: The audio clip as a CUDA tensor.

    Returns:
    - Trimmed audio as a CUDA tensor.
    """

    word = romanize_turkish(word)

    if text:
        waveform, token_spans, num_frames, emission, sample_rate, transcript = do_it(audio_clip_tensor.cpu().numpy(), text)
        start_frame, end_frame = find_word_timings(word, waveform, transcript, token_spans, sample_rate, num_frames)

        if start_frame is not None and end_frame is not None:
            start_time = start_frame / sample_rate
            end_time = end_frame / sample_rate

            # print(f"Word '{word}' timings: {start_time:.3f} - {end_time:.3f} sec. {start_frame} - {end_frame} frames.")

            # Calculate the number of frames to trim using CUDA tensor
            num_frames_to_trim = start_frame - 400
            if num_frames_to_trim > 0:
                # Trim the audio_clip_tensor directly
                trimmed_audio_tensor = audio_clip_tensor[:num_frames_to_trim]
                return trimmed_audio_tensor
            else:
                # print("trim_last_word_cuda: The calculated start_frame is less than the trimming offset.")
                return audio_clip_tensor
        else:
            print(f'trim_last_word_cuda: Word "{word}" not found in the transcript.')
            return audio_clip_tensor
    else:
        print("trim_last_word_cuda: No text provided.")
        return audio_clip_tensor


def trim_end_silence(word, text, audio_clip):
    word = romanize_turkish(word)

    if text:
        waveform, token_spans, num_frames, emission, sample_rate, transcript = do_it(audio_clip, text)

    start_frame, end_frame = find_word_timings(word, waveform, transcript, token_spans, sample_rate, num_frames)

    if start_frame is not None and end_frame is not None:
        start_time = start_frame / sample_rate
        end_time = end_frame / sample_rate

        print(f"Word '{word}' timings: {start_time:.3f} - {end_time:.3f} sec. {start_frame} - {end_frame} frames.")

        # Trim the original audio_clip instead of the processed waveform
        # Ensure audio_clip is a 1D numpy array
        if isinstance(audio_clip, np.ndarray) and audio_clip.ndim == 1:
            trimmed_audio = audio_clip[:end_frame]
            return trimmed_audio
        else:
            print("trim_last_word: Invalid audio_clip format. Expected a 1D numpy array.")
            return audio_clip
    else:
        print(f'trim_last_word: Word "{word}" not found in the transcript.')
        return audio_clip

def trim_end_silence_cuda(word, text, audio_clip):
    word = romanize_turkish(word)

    if text:
        waveform, token_spans, num_frames, emission, sample_rate, transcript = do_it(audio_clip, text)

    start_frame, end_frame = find_word_timings(word, waveform, transcript, token_spans, sample_rate, num_frames)

    if start_frame is not None and end_frame is not None:
        # Convert frame to sample index if needed
        start_time = start_frame / sample_rate
        end_time = end_frame / sample_rate

        # print(f"Word '{word}' timings: {start_time:.3f} - {end_time:.3f} sec. {start_frame} - {end_frame} frames.")

        # Trim using tensor slicing
        trimmed_audio = audio_clip[:end_frame]
        return trimmed_audio
    else:
        # print(f'trim_end_silence: Word "{word}" not found in the transcript.')
        return audio_clip



def crop_end_silence(audio, sample_rate, silence_threshold=0.01, min_silence_length=0.5, crop_length=0.4):
    """
    Check the end of the audio for silence longer than min_silence_length
    and crop crop_length from it if detected.

    :param audio: NumPy array containing the audio data
    :param sample_rate: Sample rate of the audio
    :param silence_threshold: Threshold for considering a sample as silence
    :param min_silence_length: Duration to check for silence at the end (in seconds)
    :param crop_length: Length of silence to be cropped (in seconds)
    :return: Audio with end silence cropped if necessary
    """
    silence_samples = int(sample_rate * min_silence_length)
    crop_samples = int(sample_rate * crop_length)
    end_audio = audio[-2*silence_samples:]  # Checking twice the min_silence_length

    if np.all(np.abs(end_audio) <= silence_threshold):
        # If the end is silent, crop the specified length
        cropped_audio = audio[:-crop_samples] if crop_samples < len(audio) else []
        return cropped_audio
    else:
        return audio

"""
Usage
"""
if __name__ == 'main':
    # Read the VTT file and normalize the text
    audio_path = "raw_audio/chunk_301.wav"
    vtt_file_path = "raw_audio/chunk_301.vtt"  # Replace with your VTT file path
    input_text = read_vtt_as_text(vtt_file_path).split()
    print(input_text)

    if input_text:
        waveform, token_spans, num_frames, emission, sample_rate, transcript = do_it(audio_path, input_text)
