import numpy as np
import soundfile as sf
import sounddevice as sd
import os
import pandas as pd
import time as t
from pysndfx import AudioEffectsChain


# Creates a new list that only contains file paths which has a pre-calculated d-vector
def parse_audio_files_path(voice_samples, voice_vectors, data_folder):
    found_ids = {}

    for file, id in voice_samples.items():
        # Remove unnecessary path information
        file_temp = "/".join(file.split("/")[2:4])
        voice_vector = voice_vectors.get(file_temp, [])

        if len(voice_vector) != 0 and not found_ids.get(id):
            full_path, transcript_text = load_transcript(data_folder, file)
            new_vector_list = [(full_path, voice_vector, transcript_text)]
            found_ids[id] = new_vector_list
        elif len(voice_vector) != 0:
            full_path, transcript_text = load_transcript(data_folder, file)
            found_ids[id].append((full_path, voice_vector, transcript_text))

    return found_ids


def load_transcript(data_folder, file):

    full_path = os.path.abspath(os.path.join(data_folder, file))
    head, tail = os.path.split(full_path)
    tail_comps = tail.split('.')
    tail_comps[1] = 'txt'
    wav_tail = ".".join(tail_comps)
    audio_transcript_path = os.path.join(head, wav_tail)
    txt_file = open(audio_transcript_path, "r+")
    # Removing the dot at the last word and then removing the first two words
    transcript_text_clean = txt_file.readline() \
                                .replace('.', '') \
                                .replace(':', '') \
                                .replace('?', '') \
                                .replace('!', '') \
                                .replace(',', '').split()[2:]
    transcript_text = ' '.join(transcript_text_clean)
    return full_path, transcript_text


def generate_effects():

    fxs = []

    # fxs.append(['Raw', None])
    for pitch in range(-40, -440, -40):
        # Shift in semitones (12 semitones = 1 octave)
        fx = (
            AudioEffectsChain().pitch(pitch)
        )

        fxs.append(['Pitch {}'.format(pitch), fx])

    for pitch in range(40, 440, 40):
        # Shift in semitones (12 semitones = 1 octave)
        fx = (
            AudioEffectsChain().pitch(pitch)
        )

        fxs.append(['Pitch {}'.format(pitch), fx])

    for depth in range(10, -1, -1):
        # Tremolo depth
        var_depth = 100 - depth * 10

        if var_depth == 0:
            var_depth = 1

        fx1 = (
            AudioEffectsChain().tremolo(500, depth=var_depth)
        )
        fxs.append(['Tremolo {}'.format(var_depth), fx1])

    for var in range(0, 110, 10):
        # Reverb power
        var_room_scale = var
        var_reverb = var
        fx1 = (
            AudioEffectsChain().reverb(reverberance=var_reverb,
                                       hf_damping=90,
                                       room_scale=var_room_scale,
                                       stereo_depth=100,
                                       pre_delay=20,
                                       wet_gain=0,
                                       wet_only=False)
        )
        fxs.append(['Reverberance rs {} rverb {}'.format(var, var), fx1])

    for tempo_scale in range(75, 25, -5):
        # Tempo scale
        var_tempo_scale = tempo_scale / 100
        # if var_tempo_scale == 0:
        #     var_tempo_scale = .1

        fx1 = (
            AudioEffectsChain().tempo(var_tempo_scale,
                                      use_tree=False,
                                      opt_flag=None,
                                      segment=10,
                                      search=30,
                                      overlap=30)
        )
        fxs.append(['Tempo scale {}'.format(var_tempo_scale), fx1])

    tempo_var = 75
    for var in range(10, -1, -1):

        var_pitch = (400 - (var * 40)) * (-1)
        var_room_rev = (100 - var * 10)

        var_depth = 100 - var * 10
        if var_depth == 0:
            var_depth = 1

        var_tempo_scale = tempo_var/100
        tempo_var -= 5

        fx1 = (
            AudioEffectsChain().pitch(var_pitch)
                .tempo(var_tempo_scale,
                       use_tree=False,
                       opt_flag=None,
                       segment=10,
                       search=30,
                       overlap=30)
                .reverb(reverberance=var_room_rev,
                        hf_damping=90,
                        room_scale=var_room_rev,
                        stereo_depth=100,
                        pre_delay=20,
                        wet_gain=0,
                        wet_only=False)
                .tremolo(500, depth=var_depth)
        )
        fxs.append(['Mixed level {}'.format(10 - var), fx1])

    return fxs


def play_priv_sound_files(data_set):
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
                    sample_sound(priv_audio, fs)


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
            raw_audio_spid, fs = sf.read(raw_wav_path)
            # print("Now playing raw sound.")
            # sample_sound(raw_audio_spid, fs)
            for fx in fxs:
                if fx[1] is not None:
                    print("Starting {} sound effect benchmark.".format(fx[0]))
                    input("Press any key to continue...")
                    start_time = t.time()
                    for i in range(10):
                        priv_audio = fx[1](raw_audio_spid)
                    total_time = start_time-t.time()

                    effects_time.append((fx[0], total_time))
                    # print("Now playing sound with {}".format(fx[0]))
                    # sample_sound(priv_audio, fs)

            all_data_df = pd.DataFrame(data=effects_time, columns=['Effect', 'Time to apply 1000x'])
            all_data_df.to_excel("effects_time_output.xlsx")
            break
        break


def sample_sound(priv_sound, frame_rate):
    sd.play(priv_sound, frame_rate)
    status = sd.wait()


def main():
    voice_vectors = np.load('d_vect_timit.npy', allow_pickle=True).item()
    voice_samples = np.load('data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    data_folder = "f:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    #play_priv_sound_files(data_set)

    bench_sound_effect(data_set)


if __name__ == '__main__':
    main()
