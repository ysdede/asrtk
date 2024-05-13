import tkinter as tk
from pygame import mixer
import webvtt
import os
from datetime import datetime
from asrtk.utils import natural_sort_key


class AudioPlayer:
    def __init__(self, master, audio_folder, audio_type=".wav"):
        self.master = master
        self.audio_folder = audio_folder
        self.audio_type = audio_type

        # Get the list of files and sort them naturally
        self.files = sorted([f for f in os.listdir(audio_folder) if f.endswith(audio_type)], key=natural_sort_key)

        self.current_index = 0

        mixer.init()

        # GUI Elements
        self.subtitle_text = tk.Text(master, height=4, width=100)
        self.subtitle_text.pack()

        self.info_label = tk.Label(master, text="", height=4, width=50)
        self.info_label.pack()

        self.details_label = tk.Label(master, text="", height=2, width=100)
        self.details_label.pack()

        play_button = tk.Button(master, text="Play", command=self.play)
        play_button.pack()

        prev_button = tk.Button(master, text="Previous", command=self.prev_track)
        prev_button.pack()

        next_button = tk.Button(master, text="Next", command=self.next_track)
        next_button.pack()

        # Additional GUI Elements for sorting and jumping to a specific chunk
        sort_asc_button = tk.Button(master, text="Sort Asc", command=self.sort_files_asc)
        sort_asc_button.pack()

        sort_desc_button = tk.Button(master, text="Sort Desc", command=self.sort_files_desc)
        sort_desc_button.pack()

        self.chunk_entry = tk.Entry(master)
        self.chunk_entry.pack()

        jump_button = tk.Button(master, text="Jump to", command=self.jump_to_chunk)
        jump_button.pack()

        self.update_text()

    def prev_track(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_text()
            self.play()

    def sort_files_asc(self):
        self.files.sort(key=lambda f: self.get_file_duration(f))
        self.current_index = 0
        self.update_text()

    def sort_files_desc(self):
        self.files.sort(key=lambda f: self.get_file_duration(f), reverse=True)
        self.current_index = 0
        self.update_text()

    def jump_to_chunk(self):
        chunk_index = int(self.chunk_entry.get())
        if 0 <= chunk_index < len(self.files):
            self.current_index = chunk_index
            self.update_text()
            self.play()

    def get_file_duration(self, file):
        vtt_path = os.path.join(self.audio_folder, file.replace(self.audio_type, ".vtt"))
        captions = webvtt.read(vtt_path)
        if captions:
            start_time = datetime.strptime(captions[0].start, "%H:%M:%S.%f")
            end_time = datetime.strptime(captions[-1].end, "%H:%M:%S.%f")
            return (end_time - start_time).total_seconds()
        return 0

    def play(self):
        audio_path = os.path.join(self.audio_folder, self.files[self.current_index])
        mixer.music.load(audio_path)
        mixer.music.play()

    def next_track(self):
        if self.current_index < len(self.files) - 1:
            self.current_index += 1
            self.update_text()
            self.play()

    def update_text(self):
        # Update subtitles
        vtt_file = self.files[self.current_index].replace(self.audio_type, ".vtt")
        vtt_path = os.path.join(self.audio_folder, vtt_file)
        captions = webvtt.read(vtt_path)
        subtitle_text = "\n".join([caption.text for caption in captions])

        self.subtitle_text.delete(1.0, tk.END)
        self.subtitle_text.insert(tk.END, subtitle_text)

        # Update details info (index, file name, start, end times, and duration)
        if captions:
            start_time = datetime.strptime(captions[0].start, "%H:%M:%S.%f")
            end_time = datetime.strptime(captions[-1].end, "%H:%M:%S.%f")
            duration = end_time - start_time
            duration_in_seconds = duration.total_seconds()
        else:
            start_time = end_time = duration_in_seconds = "N/A"

        details_text = f"Index: {self.current_index}, File: {vtt_file}, Start: {captions[0].start}, End: {captions[-1].end}, Duration: {duration_in_seconds:.3f} seconds"
        self.details_label.config(text=details_text)
