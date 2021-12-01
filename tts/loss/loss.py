from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super(FastSpeechLoss, self).__init__()
        self.mel_loss = nn.MSELoss()
        self.duration_loss = nn.MSELoss()

    def forward(self, real_mel, pred_mel, real_dur, pred_dur):
        # print(real_dur.shape, pred_dur.shape)
        duration_loss = self.duration_loss(real_dur, pred_dur)

        # print(real_mel.shape, pred_mel.shape)
        # if real_mel.shape[-1] < pred_mel.shape[-1]:
        #     mel_loss = self.mel_loss(real_mel, pred_mel[..., :real_mel.shape[-1]])
        # else:
        #     mel_loss = self.mel_loss(real_mel, F.pad(pred_mel, (0, real_mel.shape[-1] - pred_mel.shape[-1]),
        #                                              "constant", -11.5129251))
        min_shape = min(real_mel.shape[-1], pred_mel.shape[-1])
        mel_loss = self.mel_loss(real_mel[..., :min_shape], pred_mel[..., :min_shape])

        return mel_loss, duration_loss
