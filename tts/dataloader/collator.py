from typing import Tuple, Optional, List
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
import config


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)
        self.tokens = self.tokens.to(device)
        self.token_lengths = self.token_lengths.to(device)
        self.durations = self.durations.to(device)


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Batch:
        if config.aligner != "pretrained":
            waveform, waveform_length, transcript, tokens, token_lengths = list(
                zip(*instances)
            )
        else:
            waveform, waveform_length, transcript, tokens, token_lengths, durations = list(
                zip(*instances)
            )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        if config.aligner != "pretrained":
            return Batch(waveform, waveform_length, transcript, tokens, token_lengths)

        # print(durations)
        durations = pad_sequence(durations).transpose(0, 1)
        return Batch(waveform, waveform_length, transcript, tokens, token_lengths, durations)


class TestCollator:

    def __call__(self, instances: List[Tuple]):
        transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return transcript, tokens, token_lengths
