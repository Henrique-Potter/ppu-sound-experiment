import numpy as np
import soundfile as sf
from utils.audio_experiment_util import *


# Play the effects
def play_all_priv_sound_files(data_set):
    fxs = generate_effects()

    file_count = 0

    for id, same_person_audio_list in data_set.items():
        for audio_info in same_person_audio_list:
            file_count += 1

            raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
            print("\nFile nr: {} at: {}".format(file_count, raw_wav_path))

            # Loading files for the speaker id
            raw_audio_spid, fs = sf.read(raw_wav_path)
            # print("Now playing raw sound.")
            # sample_sound(raw_audio_spid, fs)
            for fx in fxs:
                if fx[1] is not None:
                    priv_audio = fx[1](raw_audio_spid)

                    print("Now playing sound with {}".format(fx[0]))
                    play_audio(priv_audio, fs)


def main():
    # Loading voice metadata
    voice_vectors = np.load('../models/frozen_graphs/d_vect_timit.npy', allow_pickle=True).item()
    # Loading voice transcript
    voice_samples = np.load('../data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    # Path to timit dataset
    data_folder = "f:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    play_all_priv_sound_files(data_set)


if __name__ == '__main__':
    main()
