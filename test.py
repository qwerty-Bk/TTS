from tts.dataloader.dataloader import get_dataloader
from tts.spect.melspec import get_featurizer, calc_mel_len
from tts.vocoder.vocoder import Vocoder
from tts.aligner.aligner import GraphemeAligner
from tts.model.fastspeech import FastSpeech
from tts.dataloader.ljspeech import TestDataset
from tts.dataloader.collator import TestCollator
from train import validation

import torch
from torch import nn
from scipy.io import wavfile

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def log_audio(wav, prefix):
    tmp_path = prefix + ".wav"
    wavfile.write(tmp_path, 22050, wav.cpu().numpy())


if __name__ == '__main__':
    model = nn.DataParallel(FastSpeech()).to(device)
    model.load_state_dict(torch.load("best_model", map_location=device))
    featurizer = get_featurizer()
    vocoder = Vocoder().to(device).eval()
    aligner = GraphemeAligner().to(device)

    test_dataloader = get_dataloader(TestDataset, "test.txt", batch_size=1, collate_fn=TestCollator)

    model.eval()
    validation(model, test_dataloader, log_audio, vocoder=vocoder, log_all=True)
