import os
project_path = os.getcwd().replace('\\', '/')
taco_path = project_path + '/tacotron2'
print(project_path)
print(taco_path)

import json
import pandas as pd
hifitts_path = 'C:/Users/vikto/Documents/Projects/final_project/data/hi_fi_tts_v0'
def read_json(json_path):
    dataset_type = json_path.split('_')[-1].replace('.json', '')
    with open(json_path, encoding='utf-8') as f:
        cond = "[" + f.read().replace("}\n{", "},\n{") + "]"
        json_data = json.loads(cond)
        for item in json_data:
            item['dataset_type'] = dataset_type
    return json_data

manifests = [manifest for manifest in os.listdir(hifitts_path) if 'manifest' in manifest]
manifest_paths = [f'{hifitts_path}/{manifest}' for manifest in manifests]
manifest_jsons = [read_json(manifest_path) for manifest_path in manifest_paths]
manifest_dfs = [pd.DataFrame(manifest_json) for manifest_json in manifest_jsons]
manifests_df = pd.concat(manifest_dfs, axis=0)
print('manifest_df', manifests_df.shape)

df = manifests_df.reset_index(drop=True).copy()
df['reader_id'] = df['audio_filepath'].apply(lambda x: x.split('/')[1].split('_')[0])
df['data_quality'] = df['audio_filepath'].apply(lambda x: x.split('/')[1].split('_')[1])
df['book_id'] = df['audio_filepath'].apply(lambda x: x.split('/')[2])
df['book_chapter'] = df['audio_filepath'].apply(lambda x: x.split('/')[3].replace('.flac', ''))
df['mel_path'] = 'mels/' + df.index.astype('string') + '_' + df['dataset_type'] + '_' + df['reader_id']
readers_list = [reader_id for reader_id in df.reader_id.unique()]
readers_dict = {reader_id: str(readers_list.index(reader_id)) for reader_id in readers_list}
df['reader_id_norm'] = df['reader_id'].apply(lambda x: readers_dict[x])
df['txt_line'] = df['mel_path'] + '|' + df['text'] + '|' + df['reader_id_norm'] + '\n'
print('df', df.shape)

import sys
import numpy as np
import soundfile as sf
import torch
sys.path.append(taco_path)
from tacotron2.utils import load_flac_to_torch
from tacotron2 import layers
from tacotron2.hparams import create_hparams

def flac_to_mel(line_for_create_mel):
    audio_path = line_for_create_mel.split('&')[0]
    mel_path = line_for_create_mel.split('&')[1]
    _hparams = create_hparams()
    _stft = layers.TacotronSTFT(
            _hparams.filter_length, _hparams.hop_length, _hparams.win_length,
            _hparams.n_mel_channels, _hparams.sampling_rate, _hparams.mel_fmin,
            _hparams.mel_fmax)
    def _load_flac_to_torch(audio_path):
        data, sampling_rate = sf.read(audio_path)
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate
    def _get_mel(audio_path):
        audio, sampling_rate = _load_flac_to_torch(audio_path)
        if sampling_rate != _stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, _stft.sampling_rate))
        audio_norm = audio / _hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = _stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec
    if 'hifitts' not in os.listdir(taco_path):
        os.mkdir(taco_path + '/hifitts')
        os.mkdir(taco_path + '/hifitts/mels')
    load_audio_path = hifitts_path + '/' + audio_path
    save_mel_path = taco_path + '/hifitts/' + mel_path
    melspec = _get_mel(load_audio_path)
    np.save(save_mel_path, melspec)

df['line_for_create_mel'] = df['audio_filepath'] + '&' + df['mel_path']
df['line_for_create_mel'].apply(lambda x: flac_to_mel(x))
