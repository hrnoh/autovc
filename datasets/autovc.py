from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from audio import melspectrogram, trim_silence

from backup.hparams import hparams
import librosa
import glob
from os.path import join

def build_from_path(in_dir, out_dir, test_speaker=None, num_workers=1, tqdm=lambda x: x):
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
    # Train & Test path 설정
    train_path = join(out_dir, "train")
    test_path = join(out_dir, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # speaker 저장 변수
    speakers = {}

    # for multiprocessing
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    print(os.path.join(in_dir, "*"))
    speaker_paths = glob.glob(os.path.join(in_dir, "*"))
    # 전처리 할 data가 없는 경우
    if not speaker_paths:
        print("dataset is empty!")
        exit(-1)

    # train & test split
    total_speaker_num = len(speaker_paths)
    train_speaker_num = (total_speaker_num // 10) * 9
    print("Total speaker number : %d" % total_speaker_num)
    print("train : %d, test : %d" % (train_speaker_num, total_speaker_num - train_speaker_num))

    for i, path in enumerate(speaker_paths):
        # extract speaker name
        speaker_name = path.split('/')[-1]
        speakers[speaker_name] = i

        # data output dir
        if i < train_speaker_num:
            data_out_dir = os.path.join(train_path, speaker_name)
        else:
            data_out_dir = os.path.join(test_path, speaker_name)

        print("speaker %s processing..." % speaker_name)
        futures.append(executor.submit(partial(_process_utterance, data_out_dir, path, i, speaker_name, hparams)))
        index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None], speakers

def _process_utterance(out_dir, in_dir, label, speaker_name, hparams):
    wav_paths = glob.glob(os.path.join(in_dir, "*.wav"))
    if not wav_paths:
        return None

    num_samples = len(wav_paths)
    npz_dir = os.path.join(out_dir, speaker_name)
    os.makedirs(npz_dir, exist_ok=True)

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
            np.savez(os.path.join(out_dir, file_name), mel=mel.T, speaker=label, seq_len=seq_len, frame_len=frame_len)


    return num_samples