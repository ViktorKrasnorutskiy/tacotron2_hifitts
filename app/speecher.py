# IMPORTS FOR MODELS
import sys
sys.path.append('tacotron2/')
sys.path.append('tacotron2/waveglow/')
import numpy as np
import torch
import tensorflow
from tacotron2.hparams import create_hparams
from multispeaker_modification import Tacotron2, load_model
from text import text_to_sequence
from tacotron2.waveglow import glow
from io import BytesIO
from scipy.io.wavfile import write
from config import conf


device = 'cuda' if torch.cuda.is_available() else 'cpu'
hp = create_hparams()


# PATHS
t2_state_path = conf['tacotron2_checkpoint']
wg_state_path = conf['waveglow_statedict_path']


class Speecher:
    def __init__(self):

        with torch.no_grad():

            # INIT ACOUSTIC MODEL
            self.t2 = load_model(hp)
            self.t2.load_state_dict(torch.load(t2_state_path, map_location=torch.device('cpu'))['state_dict'])
            _ = self.t2.to(device).eval()#.half()

            # INIT VOCODER MODEL
            self.wg = torch.load(wg_state_path)['model']
            self.wg = self.wg.remove_weightnorm(self.wg)
            _ = self.wg.to(device).eval()

    def synthesize(self, text, speaker_id):

        # TEXT PREPROCESSING
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()

        with torch.no_grad():

            # GET MELSPEC FROM TACOTRON INFERENCE BY SEQUENCE
            _, m, _, _ = self.t2.inference(sequence, speaker_id)

            # GET AUDIO FROM WAVEGLOW INFERENCE BY MELSPEC
            a = self.wg.infer(m)
            a = a[0].numpy()

        sr = hp.sampling_rate

        # SAVE AUDIO AS WAV IN BUFFER AND RETURN IT
        buf = BytesIO()
        write(buf, sr, a)
        audio = buf.getvalue()
        buf.close()

        return audio
