import numpy as np
import speaker_identification as si
import soundfile as sf
import sounddevice as sd
import os
from deep_speech import Speech2Text
import sox
import pandas as pd
from pysndfx import AudioEffectsChain

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
    audio_trasncript_path = os.path.join(head, wav_tail)
    txt_file = open(audio_trasncript_path, "r+")
    # Removing the dot at the last word and then removing the first two words
    transcript_text_clean = txt_file.readline() \
                                .replace('.', '') \
                                .replace(':', '') \
                                .replace('?', '') \
                                .replace('!', '') \
                                .replace(',', '').split()[2:]
    transcript_text = ' '.join(transcript_text_clean)
    return full_path, transcript_text


def st2_acc(raw_text, priv_text, true_text_script):

    from jiwer import wer

    priv_error = wer(true_text_script, priv_text)
    raw_error = wer(true_text_script, raw_text)

    return priv_error, raw_error


def main():

    voice_vectors = np.load('d_vect_timit.npy', allow_pickle=True).item()
    voice_samples = np.load('data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    data_folder = "d:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    #speaker_id_engine = si.SpeakerIdentification()
    s2t = Speech2Text()

    all_data = []

    total_cosine_distance = 0
    total_wer_priv_text = 0
    total_wer_raw_text = 0

    file_count = 0

    for id, same_person_audio_list in data_set.items():
        for audio_info in same_person_audio_list:
            file_count += 1

            # Rewriting the wav files to fix the no riff header
            t_privatizer = sox.Transformer()
            t_raw = sox.Transformer()

            fx = (
                AudioEffectsChain().pitch(120.0)
            )

            raw_wav_path = audio_info[0].replace('.wav', '_raw.wav')
            #priv_wav_path = audio_info[0].replace('.wav', '_priv.wav')

            #t_privatizer.pitch(-3.0)

            #t_privatizer.build(audio_info[0], priv_wav_path)


            # Loading files for the speaker id
            #[priv_audio_spid, fs] = sf.read(priv_wav_path)
            #[raw_audio_spid, fs2] = sf.read(raw_wav_path)

            # Extract d-vectors
            # priv_d_vector = speaker_id_engine.generate_d_vector(priv_audio_spid)
            # raw_d_vector = speaker_id_engine.generate_d_vector(raw_audio_spid)
            # Measure cosine similarity decay against original vector
            # dist = cosine_similarity(priv_d_vector, audio_info[1])
            # total_cosine_distance += dist

            raw_audio_s2t, nframes, frame_rate = s2t.load_audio_file(raw_wav_path)

            priv_audio = fx(raw_audio_s2t)

            sd.play(raw_audio_s2t, frame_rate)
            status = sd.wait()

            sd.play(priv_audio, frame_rate)
            status = sd.wait()

            priv_text = s2t.convert_to_text(priv_audio, nframes, frame_rate)
            raw_text = s2t.convert_to_text(raw_audio_s2t, nframes, frame_rate)

            #priv_error, raw_error = st2_acc(raw_text, priv_text, audio_info[2])

            #total_wer_priv_text += priv_error
            #total_wer_raw_text += raw_error

        # if file_count >= 10:
        #     break

    avg_wer_priv_text = total_wer_priv_text / file_count
    avg_wer_raw_text = total_wer_raw_text / file_count
    total_cosine_distance = total_cosine_distance / file_count

    all_data.append([3, avg_wer_raw_text, avg_wer_priv_text, total_cosine_distance])



    #pd.DataFrame(data=all_data, columns=['Pitch', 'AVG raw text', 'AVG cosine distance'])


if __name__ == '__main__':
    main()
