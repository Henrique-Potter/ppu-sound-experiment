import numpy as np
import speaker_identification as si
import sounddevice as sd
from deep_speech import Speech2Text
import pandas as pd
from pysndfx import AudioEffectsChain
from jiwer import wer
import os


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


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

    #fxs.append(['Raw', None])

    for pitch in range(-40, -440, -40):
        # Shift in semitones (12 semitones = 1 octave)
        fx = (
            AudioEffectsChain().pitch(pitch)
        )

        fxs.append(['Pitch {}'.format(pitch), fx])
    #
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

    for tempo_scale in range(10, -1, -1):
        # Tempo scale
        if tempo_scale == 0:
            var_tempo_scale = .09
        else:
            var_tempo_scale = tempo_scale/10

        fx1 = (
            AudioEffectsChain().tempo(var_tempo_scale,
                                      use_tree=False,
                                      opt_flag=None,
                                      segment=10,
                                      search=30,
                                      overlap=30)
        )
        fxs.append(['Tempo scale {}'.format(var_tempo_scale), fx1])

    for var in range(10, -1, -1):

        var_pitch = (400-(var*40))*(-1)
        var_room_rev = (100 - var*10)

        var_depth = 100 - var * 10
        if var_depth == 0:
            var_depth = 1

        var_tempo_scale = var / 10
        if var_tempo_scale == 0:
            var_tempo_scale = .09

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
        fxs.append(['Mixed level {}'.format(10-var), fx1])

    return fxs


def process_speak_id_experiment(data_set, play_sample=False):

    ml_engine = si.SpeakerIdentification()

    fxs = generate_effects()

    all_data = []

    for fx in fxs:

        total_cosine_diff = 0
        file_count = 0

        for id, same_person_audio_list in data_set.items():
            for audio_info in same_person_audio_list:
                file_count += 1

                raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
                print("\nFile nr: {} at: {}".format(file_count, raw_wav_path))

                # Load the file for the Speech Recog
                raw_audio_s2t, frame_rate = ml_engine.load_audio_file(raw_wav_path)

                if fx[1] != None:
                    print("Applying sound fx {}".format(fx[0]))
                    priv_audio = fx[1](raw_audio_s2t)
                    if play_sample:
                        sample_sound(raw_audio_s2t, frame_rate)
                        sample_sound(priv_audio, frame_rate)
                else:
                    priv_audio = raw_audio_s2t

                # Extracting d-vectors
                priv_d_vector = ml_engine.generate_d_vector(priv_audio)

                #Measure cosine similarity decay against original vector
                dist = cosine_similarity(priv_d_vector, audio_info[1])
                total_cosine_diff += dist

            if file_count >= 100:
                break

        avg_cos_priv = total_cosine_diff / file_count

        all_data.append([fx[0], avg_cos_priv])

    all_data_df = pd.DataFrame(data=all_data, columns=['Pitch', 'AVG Cosine diff'])
    all_data_df.to_excel("sid_output.xlsx")


def process_speech2text_experiment(data_set, play_sample=False):

    ml_engine = Speech2Text()

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
                raw_audio, nframes, frame_rate = ml_engine.load_audio_file(raw_wav_path)

                if fx[1] != None:
                    print("Applying sound fx {}".format(fx[0]))
                    priv_audio = fx[1](raw_audio)
                    if play_sample:
                        #sample_sound(raw_audio, frame_rate)
                        sample_sound(priv_audio, frame_rate)
                else:
                    priv_audio = raw_audio

                priv_text = ml_engine.convert_to_text(priv_audio, nframes, frame_rate)

                priv_error = wer(audio_info[2], priv_text)

                total_wer_priv_text += priv_error

            if file_count >= 100:
                break

        avg_wer_priv_text = total_wer_priv_text / file_count

        all_data.append([fx[0], avg_wer_priv_text])

    all_data_df = pd.DataFrame(data=all_data, columns=['Privatizer', 'AVG WER'])
    all_data_df.to_excel("s2t_output.xlsx")


def sample_sound(priv_sound, frame_rate):
    sd.play(priv_sound, frame_rate)
    status = sd.wait()


def main():
    voice_vectors = np.load('d_vect_timit.npy', allow_pickle=True).item()
    voice_samples = np.load('data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    data_folder = "d:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    process_speech2text_experiment(data_set, True)
    #process_speak_id_experiment(data_set)


if __name__ == '__main__':
    main()
