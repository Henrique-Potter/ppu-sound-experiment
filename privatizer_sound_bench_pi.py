import soundfile as sf
import os
import time as t
from commons_tools import *


def parse_audio_files_path(voice_samples, voice_vectors, data_folder):
    found_ids = {}

    for file, id in voice_samples.items():
        # Remove unnecessary path information
        file_temp = "/".join(file.split("/")[2:4])
        voice_vector = voice_vectors.get(file_temp, [])

        if len(voice_vector) != 0 and not found_ids.get(id):
            full_path, transcript_text = load_transcript(data_folder, file.upper())
            new_vector_list = [(full_path, voice_vector, transcript_text)]
            found_ids[id] = new_vector_list
        elif len(voice_vector) != 0:
            full_path, transcript_text = load_transcript(data_folder, file.upper())
            found_ids[id].append((full_path, voice_vector, transcript_text))

    return found_ids


def load_transcript(data_folder, file):

    full_path = os.path.abspath(os.path.join(data_folder, file))
    head, tail = os.path.split(full_path)
    tail_comps = tail.split('.')
    tail_comps[1] = 'TXT'
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
            all_data_df.to_excel("effects_time_output.xlsx")
            break
        break


def main():
    voice_vectors = np.load('d_vect_timit.npy', allow_pickle=True).item()
    voice_samples = np.load('data_lists/TIMIT_labels.npy', allow_pickle=True).item()

    data_folder = "/home/pi/projects/TIMIT"
    data_folder = "f:\\timit"
    data_set = parse_audio_files_path(voice_samples, voice_vectors, data_folder)

    bench_sound_effect(data_set)


if __name__ == '__main__':
    main()
