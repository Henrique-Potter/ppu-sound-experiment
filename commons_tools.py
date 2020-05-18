import numpy as np
import pandas as pd

# Averaging sound amplitude with a sliding window
# w_size is the windows size
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

