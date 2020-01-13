from utils import prepare_dirs
from hparams import hparams
import argparse
import importlib
from data_loader import get_loader, mel_collate

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=hparams.exp_name)
    parser.add_argument('--data_dir', type=str, default='/hd0/autovc/preprocessed/VCTK')
    parser.add_argument('--log_dir', type=str, default='/hd0/autovc/VCTK_emb256_fixed_len')
    parser.add_argument('--num_workers', type=str, default=8)
    parser.add_argument('--hparams', type=str, default=None)
    config = parser.parse_args()

    if config.exp_name == "autovc":
        solver_name = "solver"
    elif config.exp_name == "autovc_onehot_emb":
        solver_name = "solver_one_hot_emb"
    elif config.exp_name == "autovc_onehot":
        solver_name = "solver_one_hot"
    else:
        print("Invalid exp name")
        exit(-1)

    # create log & model directory
    prepare_dirs(config)

    train_loader = get_loader(data_dir=config.data_dir,
                        batch_size=hparams.batch_size,
                        speakers=None,
                        # speaker를 직접 지정 가능 ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
                        train=True,
                        collate_fn=mel_collate,
                        num_workers=1)

    test_loader = get_loader(data_dir=config.data_dir,
                              batch_size=hparams.batch_size,
                              speakers=None,
                              # speaker를 직접 지정 가능 ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
                              train=False,
                              collate_fn=mel_collate,
                              num_workers=1)

    mod = importlib.import_module("solvers." + solver_name)

    solver = mod.Solver(train_loader, test_loader, config)
    solver.train()