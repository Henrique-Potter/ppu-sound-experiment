import numpy as np
import shlex
import subprocess
import sys
import wave

from deepspeech import Model, printVersions
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote


class Speech2Text:

    def __init__(self, model_path='..\\models\\frozen_graphs\\output_graph.pbmm', beam_width=500):

        self.model_path = model_path
        self.beam_width = beam_width

        print('Loading model from file {}'.format(model_path), file=sys.stderr)
        model_load_start = timer()
        self.speech_recog_engine = Model(model_path, beam_width)
        model_load_end = timer() - model_load_start
        print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

        self.desired_sample_rate = self.speech_recog_engine.sampleRate()

    @staticmethod
    def convert_samplerate(audio_path, desired_sample_rate):
        sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(
            quote(audio_path), desired_sample_rate)
        try:
            output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
        except OSError as e:
            raise OSError(e.errno,
                          'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))
        return desired_sample_rate, np.frombuffer(output, np.int16)

    @staticmethod
    def metadata_to_string(metadata):
        return ''.join(item.character for item in metadata.items)

    def words_from_metadata(self, metadata):
        word = ""
        word_list = []
        word_start_time = 0
        # Loop through each character
        for i in range(0, metadata.num_items):
            item = metadata.items[i]
            # Append character to word if it's not a space
            if item.character != " ":
                word = word + item.character
            # Word boundary is either a space or the last character in the array
            if item.character == " " or i == metadata.num_items - 1:
                word_duration = item.start_time - word_start_time

                if word_duration < 0:
                    word_duration = 0

                each_word = dict()
                each_word["word"] = word
                each_word["start_time "] = round(word_start_time, 4)
                each_word["duration"] = round(word_duration, 4)

                word_list.append(each_word)
                # Reset
                word = ""
                word_start_time = 0
            else:
                if len(word) == 1:
                    # Log the start time of the new word
                    word_start_time = item.start_time

        return word_list

    def load_audio_file(self, wav_audio_path):

        fin = wave.open(wav_audio_path, 'rb')
        frame_rate = fin.getframerate()

        if frame_rate != self.desired_sample_rate:
            print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(
                    frame_rate, self.desired_sample_rate), file=sys.stderr)
            fs, audio = Speech2Text.convert_samplerate(wav_audio_path, self.desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

        nframes = fin.getnframes()
        fin.close()

        return audio, nframes, frame_rate

    @staticmethod
    def load_audio_file_static(wav_audio_path, desired_sample_rate):

        fin = wave.open(wav_audio_path, 'rb')
        frame_rate = fin.getframerate()

        if frame_rate != desired_sample_rate:
            print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(
                    frame_rate, desired_sample_rate), file=sys.stderr)
            fs, audio = Speech2Text.convert_samplerate(wav_audio_path, desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

        nframes = fin.getnframes()
        fin.close()

        return audio, nframes, frame_rate

    def convert_to_text(self, audio, nframes, frame_rate):

        audio_length = nframes * (1 / frame_rate)

        print('Running inference.', file=sys.stderr)
        inference_start = timer()

        text = self.speech_recog_engine.stt(audio)

        inference_end = timer() - inference_start
        print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

        return text


