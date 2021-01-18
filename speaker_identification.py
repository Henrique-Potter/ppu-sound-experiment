import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from dnn_models import MLP
from dnn_models import SincNet as cnn
from data_io import read_conf_inp, str_to_bool
import sys
from timeit import default_timer as timer
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# self.device = torch.device("cpu")


class SpeakerIdentification:

    def __init__(self, model_file = 'models/model_raw.pkl'):
        # Model to use for computing the d-vectors
        #model_file = 'models/model_raw.pkl'  # This is the model to use for computing the d-vectors (it should be pre-trained using the speaker-id DNN)
        cfg_file = 'cfg/SincNet_TIMIT.cfg'  # Config file of the speaker-id experiment used to generate the model

        self.avoid_small_en_fr = True
        self.energy_th = 0.1  # Avoid frames with an energy that is 1/10 over the average energy


        # Reading cfg file
        options = read_conf_inp(cfg_file)

        self.Batch_dev, self.CNN_net, self.DNN1_net, self.DNN2_net, self.fc_lay, self.wlen, self.wshift = self.build_model(options)

        checkpoint_load = torch.load(model_file)
        self.CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
        self.DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
        self.DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

        self.CNN_net.eval()
        self.DNN1_net.eval()
        self.DNN2_net.eval()

    @staticmethod
    def empty_gpu_cache():
        torch.cuda.empty_cache()

    def generate_d_vector(self, signal):

        print('Running inference.', file=sys.stderr)
        inference_start = timer()

        d_vector_dim = self.fc_lay[-1]

        with torch.no_grad():

            # Amplitude normalization
            signal = signal / np.max(np.abs(signal))

            # Creates a new tensor in the device (GPU or CPU)
            signal = torch.from_numpy(signal).float().to(device).contiguous()

            # Small fr
            batch_dev, en_arr_bin = self.fix_small_fr(signal)

            # split signals into chunks
            beg_samp = 0
            end_samp = self.wlen

            N_fr = int((signal.shape[0] - self.wlen) / (self.wshift))

            sig_arr = torch.zeros([batch_dev, self.wlen]).float().to(device).contiguous()
            dvects = Variable(torch.zeros(N_fr, d_vector_dim).float().to(device).contiguous())
            count_fr = 0
            count_fr_tot = 0
            while end_samp < signal.shape[0]:
                sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                beg_samp = beg_samp + self.wshift
                end_samp = beg_samp + self.wlen
                count_fr = count_fr + 1
                count_fr_tot = count_fr_tot + 1
                if count_fr == batch_dev:
                    inp = Variable(sig_arr)
                    dvects[count_fr_tot - batch_dev:count_fr_tot, :] = self.DNN1_net(self.CNN_net(inp))
                    count_fr = 0
                    sig_arr = torch.zeros([batch_dev, self.wlen]).float().to(device).contiguous()

            if count_fr > 0:
                inp = Variable(sig_arr[0:count_fr])
                dvects[count_fr_tot - count_fr:count_fr_tot, :] = self.DNN1_net(self.CNN_net(inp))

            if self.avoid_small_en_fr:
                dvects = dvects.index_select(0, (en_arr_bin == 1).nonzero().view(-1))

            # averaging and normalizing all the d-vectors
            d_vect_out = torch.mean(dvects / dvects.norm(p=2, dim=1).view(-1, 1), dim=0)

            # checks for nan
            nan_sum = torch.sum(torch.isnan(d_vect_out))

            if nan_sum > 0:
                #print(self.wav_lst_te[i])
                sys.exit(0)

            # saving the d-vector in a numpy dictionary
            #dict_key = self.wav_lst_te[i].split('/')[-2] + '/' + self.wav_lst_te[i].split('/')[-1]
            d_vector = d_vect_out.cpu().numpy()
            d_vector = np.copy(d_vector)
            #d_vect_dict[dict_key] = d_vect_out.cpu().numpy()

            inference_end = timer() - inference_start
            print('Inference took %0.3fs' % inference_end, file=sys.stderr)

        return np.copy(d_vector)

    def fix_small_fr(self, signal):

        if self.avoid_small_en_fr:
            # computing energy on each frame:
            beg_samp = 0
            end_samp = self.wlen

            N_fr = int((signal.shape[0] - self.wlen) / self.wshift)
            batch_dev = N_fr
            en_arr = torch.zeros(N_fr).float().contiguous().to(device)
            count_fr = 0
            count_fr_tot = 0
            while end_samp < signal.shape[0]:
                en_arr[count_fr] = torch.sum(signal[beg_samp:end_samp].pow(2))
                beg_samp = beg_samp + self.wshift
                end_samp = beg_samp + self.wlen
                count_fr = count_fr + 1
                count_fr_tot = count_fr_tot + 1
                if count_fr == N_fr:
                    break

            en_arr_bin = en_arr > torch.mean(en_arr) * 0.1
            en_arr_bin.to(device)
            n_vect_elem = torch.sum(en_arr_bin)

            if n_vect_elem < 10:
                print('only few elements used to compute d-vectors')
                sys.exit(0)

        return batch_dev, en_arr_bin

    @staticmethod
    def load_audio_file(raw_wav_path):

        # Loading files for the speaker id
        raw_audio_spid, fs = sf.read(raw_wav_path)
        return raw_audio_spid, fs

    def build_model(self, options):

        # [windowing]
        fs = int(options.fs)
        cw_len = int(options.cw_len)
        cw_shift = int(options.cw_shift)

        # [cnn]
        cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
        cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
        cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
        cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
        cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
        cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
        cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
        cnn_act = list(map(str, options.cnn_act.split(',')))
        cnn_drop = list(map(float, options.cnn_drop.split(',')))

        # [dnn]
        fc_lay = list(map(int, options.fc_lay.split(',')))
        fc_drop = list(map(float, options.fc_drop.split(',')))
        fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
        fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
        fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
        fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
        fc_act = list(map(str, options.fc_act.split(',')))

        # [class]
        class_lay = list(map(int, options.class_lay.split(',')))
        class_drop = list(map(float, options.class_drop.split(',')))
        class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
        class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
        class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
        class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
        class_act = list(map(str, options.class_act.split(',')))

        # loss function
        cost = nn.NLLLoss()

        # Converting context and shift in samples
        wlen = int(fs * cw_len / 1000.00)
        wshift = int(fs * cw_shift / 1000.00)

        # Batch_dev
        batch_dev = 128

        # Feature extractor CNN
        CNN_arch = {'input_dim': wlen,
                    'fs': fs,
                    'cnn_N_filt': cnn_N_filt,
                    'cnn_len_filt': cnn_len_filt,
                    'cnn_max_pool_len': cnn_max_pool_len,
                    'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
                    'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
                    'cnn_use_laynorm': cnn_use_laynorm,
                    'cnn_use_batchnorm': cnn_use_batchnorm,
                    'cnn_act': cnn_act,
                    'cnn_drop': cnn_drop,
                    }
        CNN_net = cnn(CNN_arch)
        CNN_net.to(device)

        DNN1_arch = {'input_dim': CNN_net.out_dim,
                     'fc_lay': fc_lay,
                     'fc_drop': fc_drop,
                     'fc_use_batchnorm': fc_use_batchnorm,
                     'fc_use_laynorm': fc_use_laynorm,
                     'fc_use_laynorm_inp': fc_use_laynorm_inp,
                     'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
                     'fc_act': fc_act,
                     }
        DNN1_net = MLP(DNN1_arch)
        DNN1_net.to(device)

        DNN2_arch = {'input_dim': fc_lay[-1],
                     'fc_lay': class_lay,
                     'fc_drop': class_drop,
                     'fc_use_batchnorm': class_use_batchnorm,
                     'fc_use_laynorm': class_use_laynorm,
                     'fc_use_laynorm_inp': class_use_laynorm_inp,
                     'fc_use_batchnorm_inp': class_use_batchnorm_inp,
                     'fc_act': class_act,
                     }
        DNN2_net = MLP(DNN2_arch)
        DNN2_net.to(device)

        return batch_dev, CNN_net, DNN1_net, DNN2_net, fc_lay, wlen, wshift

