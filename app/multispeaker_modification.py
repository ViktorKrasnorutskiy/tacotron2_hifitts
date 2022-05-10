import os
import time
import math
import random
from math import sqrt
from text import text_to_sequence, symbols

from numpy import finfo
import numpy as np

import tensorflow as tf

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tacotron2.model import LocationLayer, Attention, Prenet, Postnet, Encoder
from tacotron2.hparams import create_hparams
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.logger import Tacotron2Logger
from tacotron2.layers import ConvNorm, LinearNorm


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(hparams):
    model = Tacotron2(hparams).to(device)
    return model


class Decoder(nn.Module):
    def __init__(self, hparams, encoder_out_embedding_dim):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        ''' ADD SPEAKERS EMB DIM TO ENCODER OUT DIM '''
        self.speakers_embedding_dim = hparams.speakers_embedding_dim
        self.encoder_embedding_dim = encoder_out_embedding_dim
        ''' ADD SPEAKERS EMB DIM TO ENCODER OUT DIM '''
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])
        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + encoder_out_embedding_dim, # EDITED
            hparams.attention_rnn_dim)
        self.attention_layer = Attention(
            hparams.attention_rnn_dim, encoder_out_embedding_dim, # EDITED
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)
        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + encoder_out_embedding_dim, # EDITED
            hparams.decoder_rnn_dim, 1)
        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + encoder_out_embedding_dim, # EDITED
            hparams.n_mel_channels * hparams.n_frames_per_step)
        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + encoder_out_embedding_dim, # EDITED
            1, bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B,
            self.encoder_embedding_dim
        ).zero_())
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)
        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def inference(self, memory):
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory, mask=None)
        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break
            decoder_input = mel_output
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        ''' ADD SPEAKER '''
        self.speakers_embedding = nn.Embedding(
            hparams.n_speakers,
            hparams.speakers_embedding_dim)
        encoder_out_embedding_dim = hparams.encoder_embedding_dim +  hparams.speakers_embedding_dim
        ''' ADD SPEAKER '''
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams, encoder_out_embedding_dim)
        self.postnet = Postnet(hparams)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
        return outputs

    def inference(self, inputs, speaker_id):
        outputs = []
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        outputs.append(encoder_outputs)
        ''' ADD SPEAKER '''
        speaker_id = torch.IntTensor([speaker_id])
        speaker_id = speaker_id.unsqueeze(1)
        embedded_speaker = self.speakers_embedding(speaker_id)
        embedded_speaker = embedded_speaker.expand(-1, encoder_outputs.shape[1], -1)
        outputs.append(embedded_speaker)
        ''' ADD SPEAKER '''
        merged_outputs = torch.cat(outputs, -1)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            merged_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        return outputs
