from torch import nn
import torch.nn.functional as F


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super(FastSpeechLoss, self).__init__()
        self.mel_loss = nn.MSELoss()
        self.duration_loss = nn.MSELoss()

    def forward(self, real_mel, pred_mel, real_dur, pred_dur):
        duration_loss = self.duration_loss(real_dur, pred_dur)

        if real_mel.shape[-1] < pred_mel.shape[-1]:
            mel_loss = self.mel_loss(real_mel, pred_mel[..., :real_mel.shape[-1]])
        else:
            mel_loss = self.mel_loss(real_mel, F.pad(pred_mel, (0, real_mel.shape[-1] - pred_mel.shape[-1]),
                                                     "constant", -11.5129251))

        return mel_loss, duration_loss