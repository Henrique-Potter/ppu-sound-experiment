from utils.audio_experiment_util import *
import wave
import sys
from models.deep_speech import Speech2Text
import numpy as np


def load_audio_file(wav_audio_path):

    sample_rate = 16000

    fin = wave.open(wav_audio_path, 'rb')
    frame_rate = fin.getframerate()

    if frame_rate != sample_rate:
        print(
            'Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(
                frame_rate, sample_rate), file=sys.stderr)
        fs, audio = Speech2Text.convert_samplerate(wav_audio_path, sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    nframes = fin.getnframes()
    fin.close()

    return audio, nframes, frame_rate


def plot_spectrogram(sound_vector, frame_rate, file, effect_text):

    import matplotlib.pyplot as plot

    plot.subplot(211)
    plot.title('Vocal spectrogram with the effect {}'.format(effect_text))
    plot.plot(sound_vector)
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')
    plot.subplot(212)
    plot.specgram(sound_vector, Fs=frame_rate)
    plot.xlabel('Time (s)')
    plot.ylabel('Frequency (Hz)')
    plot.show()


def visualize_sound_experiments(data_set):

    fxs = generate_effects(True)


    ml_engine_reset = 0
    for window in range(15, 18, 2):

        total_wer_priv_text = 0
        file_count = 0

        for id, same_person_audio_list in data_set.items():
            for audio_info in same_person_audio_list:
                file_count += 1


                raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
                print("\nFile nr: {} at: {}".format(file_count, raw_wav_path))

                # Load the file for the Speech Recog
                raw_audio, nframes, frame_rate = load_audio_file(raw_wav_path)

                for fx in fxs:
                    if fx[1] is None:
                        plot_spectrogram(raw_audio, frame_rate, audio_info[0], 'RAW')
                        play_audio(raw_audio, frame_rate)
                    else:
                        priv_audio = fx[1](raw_audio)
                        plot_spectrogram(priv_audio, frame_rate, audio_info[0], fx[0])
                        play_audio(priv_audio, frame_rate)

                priv_audio2 = np.rint(sound_blur_with_numpy(raw_audio, window)).astype(np.int16)
                plot_spectrogram(priv_audio2, frame_rate, audio_info[0], 'AUDIO_BLUR')
                play_audio(priv_audio2, frame_rate)

                break

            break


def main():
    # Loading voice metadata
    voice_vectors = np.load('../models/frozen_graphs/d_vect_timit.npy', allow_pickle=True).item()
    # Loading voice transcript
    voice_samples = np.load('../data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    # Path to timit dataset
    data_folder = "f:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    visualize_sound_experiments(data_set)


if __name__ == '__main__':
    main()


