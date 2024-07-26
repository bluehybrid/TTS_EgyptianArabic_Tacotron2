import argparse
import os
import torch
import torchaudio
from evaluate import load
import text
import utils.make_html as html
from utils.plotting import get_spectrogram_figure
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser
from utils import get_basic_config
from pydub import AudioSegment
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator

import sys
sys.path.append("klaammain")
from klaam import SpeechRecognition

#default:
#python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --out_dir samples/test

# Examples:
#python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --out_dir samples/test_fp_adv
#python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --denoise 0.01 --out_dir samples/test_fp_adv_d
#python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_mse.pth --out_dir samples/test_fp_mse

#python test.py --model tacotron2 --checkpoint pretrained/tacotron2_ar_adv.pth --out_dir samples/test_tc2_adv
#python test.py --model tacotron2 --checkpoint pretrained/tacotron2_ar_adv.pth --denoise 0.01 --out_dir samples/test_tc2_adv_d
#python test.py --model tacotron2 --checkpoint pretrained/tacotron2_ar_mse.pth --out_dir samples/test_tc2_mse


#python test.py --model tacotron2 --checkpoint checkpoints/exp1/states_10000.pth --out_dir samples/test_tc2_adv

def test(args, text_arabic):

    config = get_basic_config()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    out_dir = args.out_dir
    sample_rate = 22_050

    # Load model
    if args.model == 'fastpitch':
        from models.fastpitch import FastPitch
        model = FastPitch(args.checkpoint)
    elif args.model == 'tacotron2':
        from models.tacotron2 import Tacotron2
        model = Tacotron2(args.checkpoint)
    else:
        raise "model type not supported"

    print(f'Loaded {args.model} from: {args.checkpoint}')
    model.eval()

    # Load vocoder model
    vocoder = load_hifigan(
        state_dict_path=config.vocoder_state_path,
        config_file=config.vocoder_config_path)
    print(f'Loaded vocoder from: {config.vocoder_state_path}')

    model, vocoder = model.to(device), vocoder.to(device)
    denoiser = Denoiser(vocoder)

    # Infer spectrogram and wave
    with torch.inference_mode():
        mel_spec = model.ttmel(text_arabic)
        wave = vocoder(mel_spec[None])
        if args.denoise > 0:
            wave = denoiser(wave, args.denoise)            

    # Save wave and images
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created folder: {out_dir}")

    torchaudio.save(f'{out_dir}/wave.wav', wave[0].cpu(), sample_rate)
    # Load the audio file
    audio = AudioSegment.from_file(f'{out_dir}/wave.wav', format='wav')

# Slow down the audio by 25% (you can adjust the speed as needed)
    slowed_audio = audio.speedup(playback_speed=0.9)

# Export the slowed down audio to a new file
    slowed_audio.export('output.wav', format='wav')

    get_spectrogram_figure(mel_spec.cpu()).savefig(
        f'{out_dir}/mel_spec.png')

    t_phon = text.arabic_to_phonemes(text_arabic)
    t_phon = text.simplify_phonemes(t_phon.replace(' ', '').replace('+', ' '))

    with open(f'{out_dir}/index.html', 'w', encoding='utf-8') as f:
        f.write(html.make_html_start())
        f.write(html.make_h_tag("Test sample", n=1))
        f.write(html.make_sample_entry2(f"./wave.wav", text_arabic, t_phon))
        f.write(html.make_h_tag("Spectrogram"))
        f.write(html.make_img_tag('./mel_spec.png'))
        f.write(html.make_volume_script(0.42))
        f.write(html.make_html_end())

    print(f"Saved test sample to: {out_dir}")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    x="فين شرشيره"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='fastpitch')
    parser.add_argument(
        '--checkpoint', default='pretrained/fastpitch_ar_adv.pth')  
    parser.add_argument('--denoise', type=float, default=0)  
    parser.add_argument('--out_dir', default='samples/test')
    args = parser.parse_args()
    mle = MLEDisambiguator.pretrained(model_name='calima-egy-r13', analyzer=None, top=1, cache_size=100000)
    sentence = simple_word_tokenize(x)
    disambig = mle.disambiguate(sentence)
    diacritized = [d.analyses[0].analysis['diac'] for d in disambig]
    a=' '.join(diacritized)

    text_arabic =a 

    test(args, text_arabic)
    out_dir = args.out_dir
    model = SpeechRecognition(lang = 'egy')
    p=model.transcribe(f'{out_dir}/wave.wav')
    print(a)
    print(p)
    cer = load("cer")
    predictions = [p]
    references = [x]
    cer_score = cer.compute(predictions=predictions, references=references)
    print("CER:")
    print(cer_score)
    wer = load("wer")
    predictions1 = [p]
    references1 = [x]
    wer_score = wer.compute(predictions=predictions1, references=references1)
    print("WER:")
    print(wer_score)



if __name__ == '__main__':
    main()
