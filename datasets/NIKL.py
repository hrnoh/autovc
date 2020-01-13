from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from audio import melspectrogram, trim_silence

from hparams import hparams
import librosa
import glob


def build_from_path(in_dir, out_dir, test_speakers=None, num_workers=1, tqdm=lambda x: x):
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

    # speaker 저장 변수
    speakers = {}

    # for multiprocessing
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    # read speaker list
    spk_list_name = os.path.join(in_dir, 'spk_list.txt')
    with open(spk_list_name, 'r') as f:
        speaker_paths = [os.path.join(in_dir, path).strip() for path in f.read().split('\n')]

    # 전처리 할 data가 없는 경우
    if not speaker_paths:
        print("dataset is empty!")
        exit(-1)

    # print total speakers
    total_speaker_num = len(speaker_paths)
    print("Total speaker number : %d" % total_speaker_num)

    for i, path in enumerate(speaker_paths):
        # extract speaker name
        speaker_name = path.split('/')[-1]
        speakers[speaker_name] = i

        print("speaker %s processing..." % speaker_name)
        futures.append(executor.submit(partial(_process_utterance, out_dir, path, i, speaker_name, hparams)))
        index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None], speakers

def _process_utterance(out_dir, in_dir, label, speaker_name, hparams):
    wav_paths = glob.glob(os.path.join(in_dir, "*.wav"))
    if not wav_paths:
        return None

    total_utter_num = len(wav_paths)
    train_utter_num = (total_utter_num // 10) * 9
    print("[%s] train : %d, test : %d" % (speaker_name, train_utter_num, total_utter_num - train_utter_num))

    num_samples = len(wav_paths)
    npz_dir = os.path.join(out_dir, speaker_name)
    os.makedirs(npz_dir, exist_ok=True)

    # Train & Test path 설정
    train_path = os.path.join(npz_dir, "train")
    test_path = os.path.join(npz_dir, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

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

            # data output dir
            if idx < train_utter_num:
                data_out_dir = train_path
            else:
                data_out_dir = test_path
            file_name = wav_name
            np.savez(os.path.join(data_out_dir, file_name), mel=mel.T, speaker=label, seq_len=seq_len, frame_len=frame_len)


    return num_samples