from torch import nn
from tts.model.attention import MultiHeadAttention
import torch
import config

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
    durations = torch.round(_durations).int()
    batch_size, leng, feats = x.shape
    max_len = torch.max(torch.sum(durations, -1), -1)[0]
    output = torch.zeros((batch_size, max_len.cpu().int().item(), feats))

    for i in range(batch_size):
        count = 0
        for j in range(leng):
            for k in range(durations[i][j]):
                output[i][count + k] = x[i][j]
            count = count + durations[i][j]

    return output


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_pred = DurationPredictor()

    def forward(self, input, target=None):
        durations = self.duration_pred(input).exp()

        if target is None:
            output = LR_function(input, durations)
            return output, durations

        output = LR_function(input, target)
        return output, durations


def generate_attention_mask(seq):
    mask = seq.eq(0)
    mask = mask.unsqueeze(1).expand(-1, seq.shape[1], -1)
    return mask


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sequential_encoding = nn.Embedding(config.vocab_size, config.encoder_input_size)
        blocks = [FFTBlock() for i in range(config.encoder_layers)]
        self.layers = nn.Sequential(*blocks)

    def forward(self, sequence):
        output = self.sequential_encoding(sequence)

        mask_att = generate_attention_mask(sequence)

        for layer in self.layers:
            output, _ = layer(output, mask_att=mask_att)

        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        blocks = [FFTBlock() for i in range(config.decoder_layers)]
        self.layers = nn.Sequential(*blocks)

    def forward(self, sequence):
        output = sequence.to(device)

        for layer in self.layers:
            output, _ = layer(output, mask_att=None)

        return output
