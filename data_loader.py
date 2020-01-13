from torch.utils import data
import torch
import os
from utils import to_categorical
from math import ceil
import glob
import random
import numpy as np
from hparams import hparams

speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']

class BaseDataset(data.Dataset):
    """ Dataset for base dataset """
    def __init__(self):
        pass

class AutoVCDataset(data.Dataset):
    """ Dataset for Mel-spectrogram features"""
    def __init__(self, data_dir, train = True, speakers = None):
        spk_path = glob.glob(os.path.join(data_dir, '*'))
        if speakers:
            spk_path = [spk for spk in spk_path if os.path.basename(spk)[:4] in speakers]

        # 일단 speaker 구분 없이 모든 sample을 가지도록 구현
        self.all_files = self.read_all_samples(spk_path, train)
        self.num_files = len(self.all_files)

        if train:
            print("\t Number of training samples: ", self.num_files)
        else:
            print("\t Number of test samples: ", self.num_files)

    # 모든 sample을 read
    def read_all_samples(self, spk_path, train=True):
        all_samples = []

        if train:
            spk_path = [os.path.join(path, 'train') for path in spk_path]
        else:
            spk_path = [os.path.join(path, 'test') for path in spk_path]

        for path in spk_path:
            samples = glob.glob(os.path.join(path, "*.npz"))
            all_samples += samples
        return all_samples

    # speaker별 sample을 read
    def read_all_samples_per_speakers(self, spk_path):
        spk_name = os.path.basename(spk_path)
        all_samples_per_spk = {}
        for path in spk_path:
            samples = glob.glob(os.path.join(path, "*.npz"))
            all_samples_per_spk[spk_name] = samples
        return all_samples_per_spk

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.all_files[index]
        npz_file = np.load(filename)

        # mel, sequence length, frame length 추출
        mel = npz_file['mel']
        speaker = npz_file['speaker'].item()
        seq_len = npz_file['seq_len'].item()
        frame_len = npz_file['frame_len'].item()

        return mel, speaker, seq_len, frame_len

def mel_collate(batch):
    """ Zero-pads model inputs and targets based on number of frames per step """
    len_out = int(hparams.freq * ceil(float(hparams.seq_len / hparams.freq)))

    mels = []
    labels = []
    labels_onehot = []
    for mel, speaker, sample_len, frame_len in batch:
        if frame_len < len_out:
            len_pad = len_out - frame_len
            x = np.pad(mel, ((0, len_pad), (0, 0)), 'constant')
        else:
            start = np.random.randint(frame_len - len_out + 1)
            x = mel[start:start+len_out]

        mels.append(x)

        label = to_categorical(speaker, hparams.speaker_num)
        labels.append(speaker)
        labels_onehot.append(label)

    mels = torch.FloatTensor(mels)
    labels = torch.LongTensor(labels)
    labels_onehot = torch.FloatTensor(labels_onehot)
    return mels, labels, labels_onehot

def mel_collate2(batch):
    """ Zero-pads model inputs and targets based on number of frames per step """
    frame_lens = [frame_len for mel, speaker, seq_len, frame_len in batch]
    max_len = max(frame_lens)
    freq = hparams.freq
    len_out = int(freq * ceil(float(max_len/freq)))

    mels = []
    labels = []
    labels_onehot = []
    for mel, speaker, seq_len, frame_len in batch:
        print(mel.shape[0])
        len_pad = len_out - frame_len
        padded_x = np.pad(mel, ((0, len_pad), (0, 0)), 'constant')
        mels.append(padded_x)

        label = to_categorical(speaker, hparams.speaker_num)
        labels.append(speaker)
        labels_onehot.append(label)

    mels = torch.FloatTensor(mels)
    labels = torch.LongTensor(labels)
    labels_onehot = torch.FloatTensor(labels_onehot)
    return mels, labels, labels_onehot


def get_loader(data_dir, batch_size=32, speakers=None, collate_fn=mel_collate, train=True, num_workers=1):
    dataset = AutoVCDataset(data_dir, train)
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    return data_loader

if __name__ == "__main__":
    data_dir = "/hd0/autovc/preprocessed/NIKL"
    train_loader = get_loader(data_dir=data_dir,
                        batch_size=hparams.batch_size,
                        speakers=None, # speaker를 직접 지정 가능 ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
                        train=True,
                        collate_fn=mel_collate,
                        num_workers=1)

    test_loader = get_loader(data_dir=data_dir,
                              batch_size=hparams.batch_size,
                              speakers=None,
                              # speaker를 직접 지정 가능 ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
                              train=False,
                              collate_fn=mel_collate,
                              num_workers=1)

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    print("======== Train loader ========")
    for i in range(5):
        mel, labels, labels_onehot = next(train_iter)
        print('-'*50)
        print(mel.size(), labels.size(), labels_onehot.size())

    print("======== Test loader ========")
    for i in range(5):
        mel, labels, labels_onehot = next(test_iter)
        print('-' * 50)
        print(mel.size(), labels.size(), labels_onehot.size())