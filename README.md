# TTS

Pretrained aligner taken from https://github.com/xcmyz/FastSpeech.

## Training

```angular2html
!git clone https://github.com/qwerty-Bk/TTS.git
%cd TTS
!pip install -r requirements.txt
!python3 train.py
!python3 train.py
```
(when you run ```!python3 train.py``` for the first time, it will fail, don't mind it)

Note that for training on Datasphere you should install packages with 

```angular2html
%pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f 
%pip install wandb
```

instead of ```!pip install -r requirements.txt```.

## Testing

```angular2html
!git clone https://github.com/qwerty-Bk/TTS.git
%cd TTS
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13SRBPIuFOAhIwUaYxOkHRp5W4L4Oa6K8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13SRBPIuFOAhIwUaYxOkHRp5W4L4Oa6K8" -O best_model && rm -rf /tmp/cookies.txt
!pip install -r requirements.txt
!python3 test.py
!python3 test.py
```

Change test.txt as you want (one line = one sentence).
