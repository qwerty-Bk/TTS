import numpy as np

from tts.dataloader.dataloader import get_dataloader
from tts.spect.melspec import get_featurizer, calc_mel_len
from tts.vocoder.vocoder import Vocoder
from tts.aligner.aligner import GraphemeAligner
from tts.model.fastspeech import FastSpeech
from tts.optimizer.optimizer import NoamOpt
from tts.loss.loss import FastSpeechLoss
from tts.dataloader.ljspeech import TestDataset
from tts.dataloader.collator import TestCollator
import config

import torch
from torch import nn
import wandb
import os
from scipy.io import wavfile
from random import randint
from torch.optim.lr_scheduler import OneCycleLR

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def log_audio(wav, prefix):
    tmp_path = "tmp.wav"
    wavfile.write(tmp_path, config.sr, wav.cpu().numpy())
    wandb.log({prefix + "audio": wandb.Audio(tmp_path, sample_rate=config.sr)})
    os.remove(tmp_path)


def validation(model, dataloader, log_audio=log_audio, vocoder=None):
    for i, batch in enumerate(dataloader):
        transcript, tokens, tokens_length = batch
        tokens.to(device)
        tokens_length.to(device)

        output = model(tokens)

        pred_wav = vocoder.inference(output)
        for j in range(len(transcript)):
            log_audio(pred_wav[j], transcript[j].split()[0])


if __name__ == '__main__':
    model = nn.DataParallel(FastSpeech()).to(device)
    # print(model)
    print(sum(param.numel() for param in model.parameters()))
    featurizer = get_featurizer()
    vocoder = Vocoder().to(device).eval()
    aligner = GraphemeAligner().to(device)
    criterion = FastSpeechLoss()

    train_dataloader = get_dataloader(batch_size=2, limit=1)
    test_dataloader = get_dataloader(TestDataset, "test.txt", batch_size=3, collate_fn=TestCollator)

    adam_opt = torch.optim.AdamW(model.parameters(),
                                 betas=(0.9, 0.98),
                                 eps=1e-9)
    if config.opt == "noam":
        optimizer = NoamOpt(adam_opt)
        scheduler = None
    elif config.opt == "oc":
        optimizer = adam_opt
        scheduler = OneCycleLR(optimizer, config.max_lr, config.epochs * len(train_dataloader))
    else:
        raise ValueError()

    model.train()

    wandb.init(project='dla3')

    best_loss = np.inf

    for epoch in range(config.epochs):
        mel_running_loss, dur_running_loss = 0, 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.no_grad():
                batch.durations = aligner(
                    batch.waveform.to(device), batch.waveform_length, batch.transcript
                )
            mel_len = calc_mel_len(batch)
            batch.durations *= mel_len.repeat(batch.durations.shape[-1], 1).transpose(0, 1)
            mels = featurizer(batch.waveform)
            mels = mels.to(device)
            batch.to(device)
            output, durations = model(batch.tokens, batch.durations)

            mel_loss, dur_loss = criterion(mels, output, batch.durations, durations)
            loss = mel_loss + dur_loss
            loss.backward()
            mel_running_loss += mel_loss.item()
            dur_running_loss += dur_loss.item()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if (i + 1) % 1 == 0:
                wandb.log({'mel_loss': mel_running_loss, 'dur_loss': dur_running_loss,
                           'loss': dur_running_loss + mel_running_loss})
                if config.opt == "noam":
                    wandb.log({'lr': optimizer.lr})
                else:
                    wandb.log({'lr': scheduler.get_last_lr()})
                print({'mel_loss': mel_running_loss, 'dur_loss': dur_running_loss,
                       'loss': dur_running_loss + mel_running_loss})

                if mel_running_loss + dur_running_loss < best_loss:
                    best_loss = mel_running_loss + dur_running_loss
                    print('updating model')
                    torch.save(model.state_dict(), "best_model")

                mel_running_loss, dur_running_loss = 0, 0
                pred_wav = vocoder.inference(output)
                wav_i = randint(0, pred_wav.shape[0] - 1)
                log_audio(pred_wav[wav_i], "pred")
                log_audio(batch.waveform[wav_i], "real")

                model.eval()
                validation(model, test_dataloader, vocoder=vocoder)
                model.train()
