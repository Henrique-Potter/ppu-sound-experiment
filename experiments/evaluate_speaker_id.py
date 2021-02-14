import time
import pandas as pd
import numpy as np

from utils.audio_experiment_util import *
from models.speaker_identification import SpeakerIdentification


def process_speaker_id_experiment(data_set):

    si = SpeakerIdentification()

    fxs = generate_effects()

    all_data = []
    ml_engine_reset = 0

    for fx in fxs:

        total_cosine_diff = 0
        file_count = 0

        for idx, same_person_audio_list in data_set.items():
            for audio_info in same_person_audio_list:
                file_count += 1
                ml_engine_reset += 1

                raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
                print("\nFile nr: {} at: {}".format(file_count, raw_wav_path))

                # Load the file for the Speech Recog
                raw_audio_sid, frame_rate = si.load_audio_file(raw_wav_path)

                # if fx[1] is None:
                #     priv_audio = raw_audio_sid
                if 'Blur' in fx[0]:
                    print("Applying sound fx {}".format(fx[0]))
                    # input("Press any key to continue...")
                    window = fx[1]
                    priv_audio = sound_blur_with_numpy(raw_audio_sid, window)
                else:
                    print("Applying sound fx {}".format(fx[0]))
                    priv_audio = fx[1](raw_audio_sid)

                # Extracting d-vectors
                priv_d_vector = si.generate_d_vector(priv_audio)

                #Measure cosine similarity decay against original vector
                dist = cosine_similarity(priv_d_vector, audio_info[1])
                total_cosine_diff += dist

            if file_count >= 100:
                # Mitigating gpu memory leak
                print("Cleaning GPU cache!")
                si.empty_gpu_cache()
                break
            elif ml_engine_reset >= 300:
                si.empty_gpu_cache()
                print("Releasing ml engine object to free GPU memory.")
                si = si.SpeakerIdentification()
                time.sleep(5)
                ml_engine_reset = 0

        avg_cos_priv = total_cosine_diff / file_count

        all_data.append([fx[0], avg_cos_priv])

    all_data_np = np.asarray(all_data).transpose()

    all_data_df = pd.DataFrame(data=all_data, columns=['Privatizer', 'Mean Cosine diff'])
    all_data_df.to_excel("sid_output.xlsx")

    si.empty_gpu_cache()


def main():
    # Loading voice metadata
    voice_vectors = np.load('../models/frozen_graphs/d_vect_timit.npy', allow_pickle=True).item()
    # Loading voice transcript
    voice_samples = np.load('../data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    # Path to timit dataset
    data_folder = "f:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    process_speaker_id_experiment(data_set)

    #visualize_sound_experiments(data_set)


if __name__ == '__main__':
    main()


