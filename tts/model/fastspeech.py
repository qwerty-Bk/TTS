import torch.nn as nn
import config

from tts.model.block import Encoder, LengthRegulator, Decoder


class FastSpeech(nn.Module):
    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()
        self.len_reg = LengthRegulator()
        self.decoder = Decoder()
        self.fc = nn.Linear(config.linear_input, config.linear_mel)

    def forward(self, sequence, durations=None, seq_length=None):
        output = self.encoder(sequence)

        if self.training:
            output, pred_dur = self.len_reg(output, target=durations)
            output = self.decoder(output)
            output = self.fc(output).transpose(1, 2)
            return output, pred_dur.squeeze(-1)

        output, pred_dur = self.len_reg(output, seq_length=seq_length)
        output = self.decoder(output)
        output = self.fc(output).transpose(1, 2)
        return output
