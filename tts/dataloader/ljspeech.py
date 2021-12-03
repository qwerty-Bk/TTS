import torchaudio
import torch
import wget
import tarfile
from pathlib import Path


def text_clean(text):
    bad = list('"üêàéâè“”’[]')
    good = [''] + list('ueaeae') + ['', '', "'", '', '']
    replace_dict = dict(zip(bad, good))
    for key in replace_dict.keys():
        text = text.replace(key, replace_dict[key])
    return text


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        self.load()
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        transcript = text_clean(transcript)
        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveform_length, transcript, tokens, token_lengths

    def load(self):
        url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        filename = "LJSpeech-1.1.tar.bz2"
        if not Path(filename).is_file():
            print('Downloading dataset')
            filename = wget.download(url)
        if not Path("./LJSpeech-1.1").exists():
            print("Unzipping dataset")
            my_tar = tarfile.open(filename)
            my_tar.extractall('.')
            my_tar.close()

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


class TestDataset:
    def __init__(self, file):
        super().__init__()
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        with open(file, 'r') as f:
            self.sentences = f.readlines()

    def __getitem__(self, index: int):
        transcript = self.sentences[index]
        transcript = text_clean(transcript)
        tokens, token_lengths = self._tokenizer(transcript)

        return transcript, tokens, token_lengths

    def __len__(self):
        return len(self.sentences)

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
