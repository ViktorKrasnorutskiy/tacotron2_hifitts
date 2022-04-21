import matplotlib
#%matplotlib inline
import matplotlib.pylab as plt

import IPython.display as ipd
import torchaudio


import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
#

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')
#

hparams = create_hparams()
hparams.sampling_rate = 44100 #22050
#

checkpoint_path = "outdir/checkpoint_3500"
#checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
model.eval()#_ = model.cuda().eval().half()
#

text = "Waveglow is really awesome!"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
#sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
sequence = torch.from_numpy(sequence).to(device='cpu', dtype=torch.int64)
#

speaker_id = 0
speaker_id = torch.IntTensor([speaker_id]).long() # .cuda().long()
#
with torch.no_grad():
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, speaker_id)

plot_data((mel_outputs.float().data.numpy()[0],#.cpu().numpy()[0],
           mel_outputs_postnet.float().data.numpy()[0],#.cpu().numpy()[0],
           alignments.float().data.numpy()[0].T))#.cpu().numpy()[0].T))
#
waveglow_path = 'waveglow/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.eval()#.half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)
#

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

#ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
torchaudio.save('foo_save.wav', audio[0], hparams.sampling_rate)
