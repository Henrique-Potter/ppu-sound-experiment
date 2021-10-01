import soundfile as sf
import time as t
from utils.audio_experiment_util import *
import numpy as np
import pandas as pd
from datetime import date


def bench_sound_effect(data_set):
    fxs = generate_effects()

    file_count = 0

    effects_time = []

    for id, same_person_audio_list in data_set.items():
        for audio_info in same_person_audio_list:
            file_count += 1

            raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
            print("\nFile nr: {} at: {}".format(file_count, raw_wav_path))

            # Loading files for the speaker id
            raw_audio, fs = sf.read(raw_wav_path)

            for fx in fxs:
                if 'Blur' in fx[0]:

                    print("Starting {} sound effect benchmark.".format(fx[0]))
                    #input("Press any key to continue...")
                    counter = 0
                    window = fx[1]
                    start_time = t.time()

                    for i in range(500):
                        priv_audio = np.rint(sound_blur_with_numpy(raw_audio, window)).astype(np.int16)
                        print("fx {} round {}".format(fx[0], i))
                        counter += 1

                    total_time = t.time()-start_time
                    effects_time.append((fx[0], total_time/counter))
                else:
                    print("Starting {} sound effect benchmark.".format(fx[0]))
                    #input("Press any key to continue...")
                    start_time = t.time()
                    counter = 0

                    for i in range(500):
                        priv_audio = fx[1](raw_audio)
                        print("fx {} round {}".format(fx[0], i))
                        counter += 1
                    total_time = t.time()-start_time

                    effects_time.append((fx[0], total_time/counter))

            all_data_df = pd.DataFrame(data=effects_time, columns=['Effect', 'Time to apply 1000x'])
            today = date.today()
            # dd/mm/YY
            d1 = today.strftime("%d/%m/%Y")

            all_data_df.to_excel("effects_time_output{}.xlsx".format(d1))
            break
        break


def main():
    # Loading voice metadata
    voice_vectors = np.load('../models/frozen_graphs/d_vect_timit.npy', allow_pickle=True).item()
    # Loading voice transcript
    voice_samples = np.load('../data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    # Path to timit dataset
    data_folder = "d:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    bench_sound_effect(data_set)


if __name__ == '__main__':
    main()
