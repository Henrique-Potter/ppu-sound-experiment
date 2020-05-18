import numpy as np
import pandas as pd

# Averaging sound amplitude with a sliding window
# w_size is the windows size
from pysndfx import AudioEffectsChain


def sound_blur(sound_vector, w_size):

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

    mv = np.convolve(sound_vector, np.ones((w_size,))/w_size, mode='same')
    return mv


# Averaging sound amplitude with a sliding window
# w_size is the windows size
def sound_blur_with_pandas(sound_vector, w_size):

    return pd.Series(sound_vector).rolling(window=w_size).mean().iloc[w_size-1:].values


# Effects list
def generate_effects():

    fxs = []
    #
    # fxs.append(['Raw', None])

    for window_size in range(3, 18, 2):
        # Sliding window avg
        fxs.append(('Blur window {}'.format(window_size), window_size))

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

    # for tempo_scale in range(75, 25, -5):
    #     # Tempo scale
    #     var_tempo_scale = tempo_scale / 100
    #     # if var_tempo_scale == 0:
    #     #     var_tempo_scale = .1
    #
    #     fx1 = (
    #         AudioEffectsChain().tempo(var_tempo_scale,
    #                                   use_tree=False,
    #                                   opt_flag=None,
    #                                   segment=10,
    #                                   search=30,
    #                                   overlap=30)
    #     )
    #     fxs.append(['Tempo scale {}'.format(var_tempo_scale), fx1])

    temp = 100
    for var in range(10, -1, -1):

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

    return fxs

