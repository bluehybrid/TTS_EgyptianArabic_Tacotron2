# TTS-EgyptianArabic-Tacotron2

TTS models (Tacotron2), trained on EGYARA dataset from MASRY TTS paper including the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) for direct TTS inference.


Papers:

Tacotron2 | Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions ([arXiv](https://arxiv.org/abs/1712.05884))

MASRY TTS | Masry: A Text-to-Speech System for the Egyptian Arabic ([SCITEPRESS](https://www.scitepress.org/Documents/2023/122443/))

HiFi-GAN  | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis ([arXiv](https://arxiv.org/abs/2010.05646))


## Quick Setup
Required packages:
`torch torchaudio pyyaml`

~ for training: `librosa matplotlib tensorboard`




Download the pretrained weights for the Tacotron2 model for Egyptian Arabic (https://drive.google.com/file/d/1etruUB2hNsYfvn5_zsDrQM6uVJW62u8u/view?usp=drive_link) then put it in pretrained folder

This repo includes the diacritization model Camel Tools (https://github.com/CAMeL-Lab/camel_tools)

Download the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) weights ([link](https://drive.google.com/u/0/uc?id=1zSYYnJFS-gQox-IeI71hVY-fdPysxuFK&export=download)). Either put them into `pretrained/hifigan-asc-v1` or edit the following lines in `configs/basic.yaml`.

```yaml
# vocoder
vocoder_state_path: pretrained/hifigan-asc-v1/hifigan-asc.pth
vocoder_config_path: pretrained/hifigan-asc-v1/config.json
```

## Using the models

The `Tacotron2` from `models.tacotron2` are wrappers that simplify text-to-mel inference. The `Tacotron2Wave` models includes the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) for direct text-to-speech inference.

## Inferring the Mel spectrogram

```python
from models.tacotron2 import Tacotron2
model = Tacotron2('pretrained/tacotron2_ar_adv.pth')
model = model.cuda()
mel_spec = model.ttmel("ازيك عامل ايه")
```



## End-to-end Text-to-Speech

```python
from models.tacotron2 import Tacotron2Wave
model = Tacotron2Wave('pretrained/tacotron2_ar_adv.pth')
model = model.cuda()
wave = model.tts("اَزيك عامل ايه")

```



By default, Arabic letters are converted using the [Buckwalter transliteration](https://en.wikipedia.org/wiki/Buckwalter_transliteration). The transliteration can also be used directly. If no Arabic script is expected to be used you can set `arabic_in=False`.



### Inference from text file
```bash
python inference.py
# default parameters:
python inference.py --list data/infer_text.txt --out_dir samples/results --model tacotron2 --checkpoint pretrained/tacotron2_ar_adv.pth --batch_size 2 --denoise 0
```

## Testing the model
To test the model run:
```bash
python test.py
# default parameters:
python test.py --model tacotron2 --checkpoint pretrained/tacotron2_ar_adv.pth --out_dir samples/test
```




## Training the model
Before training, the audio files must be resampled. The model was trained after preprocessing the files using `scripts/preprocess_audio.py`.

To train the model with options specified in the config file run:
```bash
python train.py
# default parameters:
python train.py --config configs/EGYARA.yaml
```



