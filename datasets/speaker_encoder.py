from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio

from hparams import hparams
from os.path import exists, basename, splitext
import librosa
from glob import glob
from os.path import join

from sklearn.model_selection import train_test_split

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
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
    speakers = []

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
        speakers.append(speaker_name)

        # data output dir
        if i < train_speaker_num:
            data_out_dir = os.join(train_path, speaker_name)
        else:
            data_out_dir = os.join(test_path, speaker_name)

        print("speaker %s processing..." % speaker_name)
        futures.append(executor.submit(partial(_process_utterance, data_out_dir, path, speaker_name, hparams)))
        index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None], speakers

def _process_utterance(out_dir, in_dir, speaker, hparams):
    wav_paths = glob.glob(os.path.join(in_dir, "*.wav"))
    if not wav_paths:
        return None

    num_samples = len(wav_paths)

    utter_min_len = (hparams.sv_frame * hparams.hop + hparams.window) * hparams.sr
    for idx, wav_path in enumerate(wav_paths):
        wav_name, ext = os.path.splitext(os.path.basename(wav_path))
        utterances_spec = []
        if ext == ".wav":
            utter, sr = librosa.load(wav_path, sr=hparams.sample_rate)

            # rescale wav
            if hparams.rescaling:  # hparams.rescale = True
                utter = utter / np.abs(utter).max() * hparams.rescaling_max

            # M-AILABS extra silence specific
            #if hparams.trim_silence:  # hparams.trim_silence = True
            #    wav = trim_silence(wav, hparams)  # Trim leading and trailing silence
            intervals = librosa.effects.split(utter, top_db=30)
            for interval in intervals:
                if (interval[1] - interval[0]) > utter_min_len:
                    utter_part = utter[interval[0]:interval[1]]
                    S = librosa.core.srft(y=utter_part, n_fft=hparams.nfft,
                                          win_length=int(hparams.window * hparams.sr), hop_length=int(hparams.hop * hparams.sr))
                    S = np.abs(S) ** hparams.power
                    mel_basis = librosa.filters.mel(sr=hparams.sr, n_fft=hparams.nfft, n_mels=hparams.nmels)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)

                    utterances_spec.append(S[:, :hparams.sv_frame])
                    utterances_spec.append(S[:, -hparams.sv_frame:])

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        np.save(os.path.join(out_dir, speaker+"%d.npy" % idx), utterances_spec)

        #mel = melspectrogram(wav, hparams)
        #seq_len = wav.shape[0]
        #frame_len = mel.shape[1]

        #file_name = wav_name
        #np.savez(os.path.join(out_dir, file_name), mel=mel.T, seq_len=seq_len, frame_len=frame_len)

    return num_samples