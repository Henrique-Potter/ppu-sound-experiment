import time

from utils.audio_experiment_util import *
from models.speaker_identification import SpeakerIdentification
from models.deep_speech import *
import pandas as pd
from jiwer import wer


def process_speech2text_experiment(data_set):

    ml_engine_s2t = Speech2Text()
    fxs = generate_effects()

    all_data = []

    for fx in fxs:

        total_wer_priv_text = 0
        file_count = 0

        for id, same_person_audio_list in data_set.items():
            for audio_info in same_person_audio_list:

                file_count += 1
                raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
                print("\nFile nr: {} at: {}".format(file_count, raw_wav_path))

                # Load the file for the Speech Recog
                raw_audio, nframes, frame_rate = ml_engine_s2t.load_audio_file(raw_wav_path)

                # if fx[1] is None:
                #     priv_audio = raw_audio_sid
                if 'Blur' in fx[0]:
                    print("Applying sound fx {}".format(fx[0]))
                    # input("Press any key to continue...")
                    window = fx[1]
                    priv_audio = np.rint(sound_blur_with_numpy(raw_audio, window)).astype(np.int16)
                else:
                    print("Applying sound fx {}".format(fx[0]))
                    priv_audio = fx[1](raw_audio)

                # Applying the model
                priv_text = ml_engine_s2t.convert_to_text(priv_audio, nframes, frame_rate)

                # Getting a performance metrics for this particular audio effect
                priv_error = wer(audio_info[2], priv_text)

                total_wer_priv_text += priv_error

            if file_count >= 100:
                break

        avg_wer_priv_text = total_wer_priv_text / file_count

        all_data.append([fx[0], avg_wer_priv_text])

    all_data_df = pd.DataFrame(data=all_data, columns=['Privatizer', 'AVG WER'])
    all_data_df.to_excel("s2t_output.xlsx")


def sample_sound_after_effect(data_set):

    ml_engine_s2t = Speech2Text()

    fxs = generate_effects()

    all_data = []
    ml_engine_reset = 0
    for window in range(3, 18, 2):

        total_wer_priv_text = 0
        file_count = 0

        for id, same_person_audio_list in data_set.items():
            for audio_info in same_person_audio_list:
                file_count += 1
                ml_engine_reset += 1

                raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
                print("\nFile nr: {} at: {}".format(file_count, raw_wav_path))

                # Load the file for the Speech Recog
                raw_audio, nframes, frame_rate = ml_engine_s2t.load_audio_file(raw_wav_path)

                start = time.time()

                priv_audio = sound_blur(raw_audio, window)
                priv_audio2 = np.rint(sound_blur_with_numpy(raw_audio, window)).astype(np.int16)

                total_time = time.time() - start
                print("Time to blur {}".format(total_time))
                print("Frame speed {}".format(nframes/total_time))

                play_audio(raw_audio, frame_rate)
                play_audio(priv_audio, frame_rate)

            if file_count >= 50:
                break

        avg_wer_priv_text = total_wer_priv_text / file_count

        all_data.append(["Blurred window {}".format(window), avg_wer_priv_text])

    all_data_np = np.asarray(all_data).transpose()

    all_data_df = pd.DataFrame(data=all_data, columns=['Privatizer', 'WER'])
    all_data_df.to_excel("sid_output.xlsx")


def main():
    # Loading voice metadata
    voice_vectors = np.load('../models/frozen_graphs/d_vect_timit.npy', allow_pickle=True).item()
    # Loading voice transcript
    voice_samples = np.load('../data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    # Path to timit dataset
    data_folder = "f:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    process_speech2text_experiment(data_set)

    #sample_sound_after_effect(data_set)

    #visualize_sound_experiments(data_set)


if __name__ == '__main__':
    main()


