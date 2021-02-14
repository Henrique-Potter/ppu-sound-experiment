from utils.audio_experiment_util import *
import os
import numpy as np


# Calculates the average size of audio files.
def calculate_audio_file_statistics(data_set):

    total_file_size = 0

    file_count = 0

    for id, same_person_audio_list in data_set.items():
        for audio_info in same_person_audio_list:

            file_count += 1
            raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
            print("\nFile nr: {} at: {}".format(file_count, raw_wav_path))

            total_file_size += os.path.getsize(raw_wav_path)

    print("\n\n Nr Files: {} Total Size: {} Avg Size: {}".format(file_count, total_file_size, total_file_size/file_count))


def main():
    voice_vectors = np.load('d_vect_timit.npy', allow_pickle=True).item()
    voice_samples = np.load('data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    data_folder = "f:\\timit"

    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    calculate_audio_file_statistics(data_set)


if __name__ == '__main__':
    main()


