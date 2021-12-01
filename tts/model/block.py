from torch import nn
from tts.model.attention import MultiHeadAttention
import torch
import config
import math

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(config.conv1_input_size, config.conv1_output_size, kernel_size=config.conv_kernel,
                      padding=config.conv_kernel // 2),
            nn.ReLU(),
            nn.Conv1d(config.conv2_input_size, config.conv2_output_size, kernel_size=config.conv_kernel,
                      padding=config.conv_kernel // 2),
            nn.Dropout(config.dropout)
        )
        self.layer_norm = nn.LayerNorm(config.conv2_output_size)

    def forward(self, input):
        residual = input.clone()
        output = input.transpose(1, 2)
        output = self.convs(output)
        output = output.transpose(1, 2)
        output = self.layer_norm(output + residual)
        return output


class FFTBlock(nn.Module):
    def __init__(self):
        super(FFTBlock, self).__init__()
        self.att = MultiHeadAttention(config.att_heads, config.fft_emb, dropout=config.dropout)
        self.conv = ConvNet()

    def forward(self, input, mask_att=None):
        output, att = self.att(input, input, input, mask_att)

        output = self.conv(output)

        return output, att


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, input):
        output = input.transpose(1, 2)
        output = self.conv(output)
        return output.transpose(1, 2)


class DurationPredictor(nn.Module):
    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.conv = nn.Sequential(
            Conv(config.dp_input_size, config.dp_hidden_size, kernel_size=config.dp_kernel_size,
                 padding=config.dp_kernel_size // 2),
            nn.LayerNorm(config.dp_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            Conv(config.dp_hidden_size, config.dp_output_size, kernel_size=config.dp_kernel_size,
                 padding=config.dp_kernel_size // 2),
            nn.LayerNorm(config.dp_output_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(config.dp_output_size, 1),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.conv(input)
        output.squeeze()
        if not self.training:
            output.unsqueeze(0)
        return output


def LR_function(x, _durations):
    batch_size, leng, feats = x.shape
    max_len = torch.max(torch.sum(_durations, -1), -1)[0]
    durations = torch.round(_durations).int()
    # print(_durations.shape)
    # print(max_len)
    output = torch.zeros((batch_size, max_len.cpu().int().item(), feats))

    for i in range(batch_size):
        count = 0
        for j in range(leng):
            output[i][count:count + durations[i][j]] = x[i][j]
            count = count + durations[i][j]

    return output


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_pred = DurationPredictor()

    def forward(self, input, target=None):
        durations = self.duration_pred(input).exp()

        if target is None:
            output = LR_function(input, durations.squeeze(-1))
            return output, durations

        output = LR_function(input, target)
        return output, durations


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_attention_mask(seq):
    mask = seq.eq(0)
    mask = mask.unsqueeze(1).expand(-1, seq.shape[1], -1)
    return mask


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sequential_encoding = nn.Embedding(config.vocab_size, config.encoder_input_size)
        self.positional_encoding = PositionalEncoding(config.encoder_input_size)
        blocks = [FFTBlock() for i in range(config.encoder_layers)]
        self.layers = nn.Sequential(*blocks)

    def forward(self, sequence):
        output = self.sequential_encoding(sequence)
        output = self.positional_encoding(output)

        mask_att = generate_attention_mask(sequence)

        for layer in self.layers:
            output, _ = layer(output, mask_att=mask_att)

        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.positional_encoding = PositionalEncoding(config.encoder_input_size)
        blocks = [FFTBlock() for i in range(config.decoder_layers)]
        self.layers = nn.Sequential(*blocks)

    def forward(self, sequence):
        output = sequence.to(device)
        output = self.positional_encoding(output)

        for layer in self.layers:
            output, _ = layer(output, mask_att=None)

        return output
