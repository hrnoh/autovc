from torch.utils import data
import torch
import os
import random
from math import ceil
import glob
import numpy as np
from hparams import hparams

speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']

class AutoVCDataset(data.Dataset):
    """ Dataset for Mel-spectrogram features"""
    def __init__(self, data_dir, speakers = None):
        spk_path = glob.glob(os.path.join(data_dir, '*'))
        if speakers:
            spk_path = [spk for spk in spk_path if os.path.basename(spk)[:4] in speakers]

        # 일단 speaker 구분 없이 모든 sample을 가지도록 구현
        self.all_files = self.read_all_samples(spk_path)
        self.num_files = len(self.all_files)
        print("\t Number of training samples: ", self.num_files)

    # 모든 sample을 read
    def read_all_samples(self, spk_path):
        all_samples = []
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
        seq_len = npz_file['seq_len'].item()
        frame_len = npz_file['frame_len'].item()

        return mel, seq_len, frame_len

def mel_collate(batch):
    """ Zero-pads model inputs and targets based on number of frames per step """
    frame_lens = [frame_len for mel, seq_len, frame_len in batch]
    max_len = max(frame_lens)
    freq = hparams.freq_axis_kernel_size
    len_out = int(freq * ceil(float(max_len/freq)))

    results = []
    for mel, seq_len, frame_len in batch:
        len_pad = len_out - frame_len
        padded_x = np.pad(mel, ((0, len_pad), (0, 0)), 'constant')
        results.append(padded_x)

    out = torch.FloatTensor(results)
    return out


def get_loader(data_dir, batch_size=32, speakers=None, collate_fn=mel_collate, mode='train', num_workers=1):
    dataset = AutoVCDataset(data_dir)
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    return data_loader

if __name__ == "__main__":
    data_dir = "/hd0/autovc/preprocessed"
    loader = get_loader(data_dir=data_dir,
                        batch_size=hparams.batch_size,
                        speakers=None, # speaker를 직접 지정 가능 ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
                        mode='train',
                        collate_fn=mel_collate,
                        num_workers=1)

    data_iter = iter(loader)
    for i in range(10):
        mel = next(data_iter)
        print('-'*50)
        print(mel.size())