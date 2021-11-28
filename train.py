from tts.dataloader.dataloader import get_dataloader
from tts.spect.melspec import get_featurizer, calc_mel_len
from tts.vocoder.vocoder import Vocoder
from tts.aligner.aligner import GraphemeAligner
from tts.model.fastspeech import FastSpeech
from tts.optimizer.optimizer import NoamOpt
from tts.loss.loss import FastSpeechLoss
import config

import torch
from torch import nn
import wandb
import os
from scipy.io import wavfile
from random import randint

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def tbd():
    train_dataloader = get_dataloader(batch_size=2)

    featurizer = get_featurizer()
    vocoder = Vocoder().to(device).eval()
    aligner = GraphemeAligner().to(device)
    # wandb.init(project='dla3')
    i = 0
    for batch in train_dataloader:
        batch.durations = aligner(
            batch.waveform.to(device), batch.waveform_length, batch.transcript
        )
        ml = calc_mel_len(batch)
        sound_part = torch.sum(batch.durations, dim=1)
        batch.durations /= sound_part.repeat(batch.durations.shape[-1], 1).transpose(0, 1)
        print('should be 1:', torch.sum(batch.durations, dim=1))
        print(batch.durations.shape, ml.repeat(batch.durations.shape[-1], 1).transpose(0, 1).shape)
        batch.durations *= ml.repeat(batch.durations.shape[-1], 1).transpose(0, 1)
        print("wf shape:", batch.waveform.shape)
        mels = featurizer(batch.waveform)
        print('mels:', mels.shape)
        sound_part = torch.sum(batch.durations, dim=1)
        print('dur shape:', batch.durations.shape)
        print('durs:', sound_part)
        # trunc = math.ceil(mels.shape[-1] * sound_part) * 0 + 832
        # print(trunc)
        # short_wav = vocoder.inference(mels[:, :, :trunc].to('cuda:0'))
        # log_audio(short_wav, "test")
        print('wf l:', batch.waveform_length)

        # print('sw s:', short_wav.shape)

        print('mel len:', ml)
        # new_durations = aligner(
        #     short_wav.to(device), [trunc], batch.transcript
        # )
        # print('new:', sum(new_durations[0]))
        break


def log_audio(wav, prefix):
    tmp_path = "tmp.wav"
    wavfile.write(tmp_path, 22050, wav.cpu().numpy())
    wandb.log({prefix + "audio": wandb.Audio(tmp_path, sample_rate=22050)})
    os.remove(tmp_path)


if __name__ == '__main__':
    # tbd()
    # 1 / 0
    model = nn.DataParallel(FastSpeech()).to(device)
    # print(model)
    print(sum(param.numel() for param in model.parameters()))
    adam_opt = torch.optim.Adam(model.parameters(),
                                betas=(0.9, 0.98),
                                eps=1e-9)
    optimizer = NoamOpt(adam_opt)
    featurizer = get_featurizer()
    vocoder = Vocoder().to(device).eval()
    aligner = GraphemeAligner().to(device)
    criterion = FastSpeechLoss()

    train_dataloader = get_dataloader(batch_size=3, limit=1)

    model.train()

    wandb.init(project='dla3')

    for epoch in range(config.epochs):
        mel_running_loss, dur_running_loss = 0, 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.no_grad():
                batch.durations = aligner(
                    batch.waveform.to(device), batch.waveform_length, batch.transcript
                )
            # sound_part = torch.sum(batch.durations, dim=1)
            # batch.durations /= sound_part.repeat(batch.durations.shape[-1], 1).transpose(0, 1)
            mel_len = calc_mel_len(batch)
            batch.durations *= mel_len.repeat(batch.durations.shape[-1], 1).transpose(0, 1)
            mels = featurizer(batch.waveform)
            mels = mels.to(device)
            # print('mels:', mels.shape)
            batch.to(device)
            output, durations = model(batch.tokens, batch.durations)
            # print('final:', output.shape)
            # print('durations:', batch.durations.shape, durations.shape)
            # 1 / 0

            mel_loss, dur_loss = criterion(mels, output, batch.durations, durations)
            loss = mel_loss + dur_loss
            loss.backward()
            mel_running_loss += mel_loss.item()
            dur_running_loss += dur_loss.item()

            optimizer.step()
            if (i + 1) % 1 == 0:
                wandb.log({'mel_loss': mel_running_loss, 'dur_loss': dur_running_loss, 'lr': optimizer.lr,
                           'loss': dur_running_loss + mel_running_loss})
                print({'mel_loss': mel_running_loss, 'dur_loss': dur_running_loss, 'lr': optimizer.lr,
                       'loss': dur_running_loss + mel_running_loss})
                mel_running_loss, dur_running_loss = 0, 0
                real_wav = vocoder.inference(mels)
                pred_wav = vocoder.inference(output)
                wav_i = randint(0, pred_wav.shape[0] - 1)
                log_audio(pred_wav[wav_i], "pred")
                log_audio(real_wav[wav_i], "real")
            # if epoch == config.epochs - 1:
            #     print(mels)
            #     print(output)
            #     print('----')
            #     print(batch.durations)
            #     print(durations)

        # break
