# coding: utf-8
"""
python preprocess.py --num_workers 10 --name son --in_dir D:\hccho\multi-speaker-tacotron-tensorflow-master\datasets\son --out_dir .\data\son
python preprocess.py --num_workers 10 --name moon --in_dir D:\hccho\multi-speaker-tacotron-tensorflow-master\datasets\moon --out_dir .\data\moon
 ==> out_dir에  'audio', 'mel', 'linear', 'time_steps', 'mel_frames', 'text', 'tokens', 'loss_coeff'를 묶은 npz파일이 생성된다.
"""
import argparse
import os
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import glob
from hparams import hparams, hparams_debug_string
import warnings
import librosa
import numpy as np
from audio import trim_silence, melspectrogram

warnings.simplefilter(action='ignore', category=FutureWarning)

def build_from_path(hparams, in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess
        - out_dir: output directory of npz files
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txtX

    """
    speakers = []

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    print(os.path.join(in_dir, "*"))
    speaker_paths = glob.glob(os.path.join(in_dir, "*"))
    # 전처리 할 data가 없는 경우
    if not speaker_paths:
        print("dataset is empty!")
        exit(-1)

    print("There are {} speakers".format(len(speaker_paths)))

    for path in speaker_paths:
        speaker_name = path.split('/')[-1]

        speakers.append(speaker_name)
        data_out_dir = os.path.join(out_dir, speaker_name)
        if not os.path.exists(data_out_dir):
            try:
                os.mkdir(data_out_dir)
            except FileExistsError:
                print("speaker {} exists".format(data_out_dir.split("/")[-1]))

        print("speaker %s processing..." % speaker_name)
        futures.append(executor.submit(partial(_process_utterance, data_out_dir, path, speaker_name, hparams)))
        index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None], speakers

def _process_utterance(out_dir, in_dir, speaker, hparams):
    wav_paths = glob.glob(os.path.join(in_dir, "*.wav"))
    if not wav_paths:
        return None

    num_samples = len(wav_paths)

    for idx, wav_path in enumerate(wav_paths):
        wav_name, ext = os.path.splitext(os.path.basename(wav_path))
        if ext == ".wav":
            wav, sr = librosa.load(wav_path, sr=hparams.sample_rate)

            # rescale wav
            if hparams.rescaling:  # hparams.rescale = True
                wav = wav / np.abs(wav).max() * hparams.rescaling_max

            # M-AILABS extra silence specific
            if hparams.trim_silence:  # hparams.trim_silence = True
                wav = trim_silence(wav, hparams)  # Trim leading and trailing silence

            mel = melspectrogram(wav, hparams)
            seq_len = wav.shape[0]
            frame_len = mel.shape[1]

            file_name = wav_name
            np.savez(os.path.join(out_dir, file_name), mel=mel.T, seq_len=seq_len, frame_len=frame_len)

    return num_samples

def preprocess(in_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata, speakers = build_from_path(hparams, in_dir, out_dir, num_workers=num_workers, tqdm=tqdm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='/hd0/dataset/VCTK/VCTK-Corpus/wav48')
    parser.add_argument('--out_dir', type=str, default='/hd0/autovc/preprocessed')
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

    print("Sampling frequency: {}".format(hparams.sample_rate))
    preprocess(in_dir, out_dir, num_workers)