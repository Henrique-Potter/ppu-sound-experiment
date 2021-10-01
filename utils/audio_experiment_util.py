
def cosine_similarity(x, y):
    import numpy as np
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


# Creates a new list that only contains file paths, transcripts, and a pre-calculated d-vector
# Facilitates models performance evaluation
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
    import os

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


def sound_blur(sound_vector, w_size):
    import numpy as np
    half_w = int(w_size/2)
    sound_vector_size = len(sound_vector)

    blurred_sound = np.copy(sound_vector)

    for i in range(sound_vector_size):
        if i-half_w < 0:
            blurred_sound[i] = round(np.sum(sound_vector[0:(i + half_w + 1)]) / len(sound_vector[0:(i + half_w + 1)]))
        elif i+half_w+1 > sound_vector_size:
            blurred_sound[i] = round(np.sum(sound_vector[i-half_w:i])/len(sound_vector[i-half_w:i]))
        else:
            blurred_sound[i] = round(np.sum(sound_vector[(i - half_w):(i+half_w+1)])/w_size)
    return blurred_sound


# Averaging sound amplitude with a sliding window (Fastest)
# w_size is the windows size
def sound_blur_with_numpy(sound_vector, w_size):
    import numpy as np
    return np.convolve(sound_vector, np.ones((w_size,))/w_size, mode='same')


# Averaging sound amplitude with a sliding window
# w_size is the windows size
def sound_blur_with_pandas(sound_vector, w_size):
    import pandas as pd

    return pd.Series(sound_vector).rolling(window=w_size).mean().iloc[w_size-1:].values


# Effects list
def generate_effects(max_effects=False):
    from pysndfx import AudioEffectsChain

    # fxs = [['Raw', None]]
    fxs = []

    for pitch in range(-40, -440, -40):
        # Shift in semitones (12 semitones = 1 octave)
        if max_effects:
            pitch = -440

        fx = (
            AudioEffectsChain().pitch(pitch)
        )

        fxs.append(['Pitch {}'.format(pitch), fx])

        if max_effects:
            break

    for pitch in range(40, 440, 40):
        # Shift in semitones (12 semitones = 1 octave)

        if max_effects:
            pitch = 440

        fx = (
            AudioEffectsChain().pitch(pitch)
        )

        fxs.append(['Pitch {}'.format(pitch), fx])

        if max_effects:
            break

    for depth in range(10, -1, -1):
        # Tremolo depth

        if max_effects:
            depth = 1

        var_depth = 100 - depth * 10

        if var_depth == 0:
            var_depth = 1

        fx1 = (
            AudioEffectsChain().tremolo(500, depth=var_depth)
        )
        fxs.append(['Tremolo {}'.format(var_depth), fx1])

        if max_effects:
            break

    for var in range(0, 110, 10):
        # Reverb power

        if max_effects:
            var = 90

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

        if max_effects:
            break

    for tempo_scale in range(75, 25, -5):
        # Tempo scale

        if max_effects:
            tempo_scale = 50

        var_tempo_scale = tempo_scale / 100
        if var_tempo_scale == 0:
            var_tempo_scale = .1

        fx1 = (
            AudioEffectsChain().tempo(var_tempo_scale,
                                      use_tree=False,
                                      opt_flag=None,
                                      segment=10,
                                      search=30,
                                      overlap=30)
        )
        fxs.append(['Tempo scale {}'.format(var_tempo_scale), fx1])

        if max_effects:
            break


    temp = 100
    for var in range(10, -1, -1):

        if max_effects:
            var = 4

        var_pitch = (400 - (var * 40)) * (-1)
        var_room_rev = (100 - var * 10)

        var_depth = 100 - var * 10
        if var_depth == 0:
            var_depth = 1

        fx1 = (
            AudioEffectsChain().pitch(var_pitch)
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

        if max_effects:
            break

    return fxs

# Creates a new list that only contains file paths which has a pre-calculated d-vector
# def parse_audio_files_path(voice_samples, voice_vectors, data_folder):
#     found_ids = {}
#
#     for file, id in voice_samples.items():
#         # Remove unnecessary path information
#         file_temp = "/".join(file.split("/")[2:4])
#         voice_vector = voice_vectors.get(file_temp, [])
#
#         if len(voice_vector) != 0 and not found_ids.get(id):
#             full_path, transcript_text = load_transcript(data_folder, file)
#             new_vector_list = [(full_path, voice_vector, transcript_text)]
#             found_ids[id] = new_vector_list
#         elif len(voice_vector) != 0:
#             full_path, transcript_text = load_transcript(data_folder, file)
#             found_ids[id].append((full_path, voice_vector, transcript_text))
#
#     return found_ids
#
#
# def load_transcript(data_folder, file):
#
#     full_path = os.path.abspath(os.path.join(data_folder, file))
#     head, tail = os.path.split(full_path)
#     tail_comps = tail.split('.')
#     tail_comps[1] = 'txt'
#     wav_tail = ".".join(tail_comps)
#     audio_transcript_path = os.path.join(head, wav_tail)
#     txt_file = open(audio_transcript_path, "r+")
#     # Removing the dot at the last word and then removing the first two words
#     transcript_text_clean = txt_file.readline() \
#                                 .replace('.', '') \
#                                 .replace(':', '') \
#                                 .replace('?', '') \
#                                 .replace('!', '') \
#                                 .replace(',', '').split()[2:]
#     transcript_text = ' '.join(transcript_text_clean)
#     return full_path, transcript_text


def generate_effects_v2():
    from pysndfx import AudioEffectsChain
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


def play_audio(priv_sound, frame_rate):
    import sounddevice as sd
    sd.play(priv_sound, frame_rate)
    status = sd.wait()

