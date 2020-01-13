# coding: utf-8
"""
python preprocess.py --num_workers 10 --name son --in_dir D:\hccho\multi-speaker-tacotron-tensorflow-master\datasets\son --out_dir .\data\son
python preprocess.py --num_workers 10 --name moon --in_dir D:\hccho\multi-speaker-tacotron-tensorflow-master\datasets\moon --out_dir .\data\moon
 ==> out_dir에  'audio', 'mel', 'linear', 'time_steps', 'mel_frames', 'text', 'tokens', 'loss_coeff'를 묶은 npz파일이 생성된다.
"""
import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from hparams import hparams, hparams_debug_string
import warnings
import importlib

warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess(mod, in_dir, out_dir, test_speakers=None, num_workers=1):
    os.makedirs(out_dir, exist_ok=True)
    metadata, speakers = mod.build_from_path(in_dir, out_dir, test_speakers=test_speakers, num_workers=num_workers, tqdm=tqdm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='VCTK')
    parser.add_argument('--in_dir', type=str, default='/hd0/dataset/VCTK/VCTK-Corpus/wav48')
    #parser.add_argument('--in_dir', type=str, default='/hd0/dataset/NIKL/wavs')
    parser.add_argument('--out_dir', type=str, default='/hd0/autovc/preprocessed/VCTK')
    parser.add_argument('--test_speakers', type=str, default=None) # select test speaker (e.g. "p333,p334")
    parser.add_argument('--num_workers', type=str, default=8)
    parser.add_argument('--hparams', type=str, default=None)
    args = parser.parse_args()

    if args.hparams is not None:
        hparams.parse(args.hparams)
    print(hparams_debug_string())

    in_dir = args.in_dir
    out_dir = args.out_dir
    num_workers = args.num_workers
    num_workers = cpu_count() if num_workers is None else int(num_workers)  # cpu_count() = process 갯수

    if args.test_speakers:
        test_speakers = args.test_speakers.split(',')
        test_speakers = [spk.strip() for spk in test_speakers]
    else:
        test_speakers = None

    assert args.name in ['autovc', 'speaker_encoder', 'VCTK', 'NIKL']
    print("Sampling frequency: {}".format(hparams.sample_rate))
    mod = importlib.import_module("datasets." + args.name)
    preprocess(mod, in_dir, out_dir, test_speakers, num_workers)