import numpy as np
import os
from shutil import copyfile

def to_categorical(label, class_num):
    one_hot = np.zeros(class_num)
    one_hot[label] = 1

    return one_hot

def prepare_dirs(config):
    if hasattr(config, 'log_dir'):
        log_dir = config.log_dir
        os.makedirs(log_dir, exist_ok=True)

        if hasattr(config, 'exp_name'):
            exp_name = config.exp_name
            root_path = os.path.join(log_dir, exp_name)
            log_path = os.path.join(root_path, 'log')
            model_path = os.path.join(root_path, 'model')

            os.makedirs(log_path, exist_ok=True)
            os.makedirs(model_path, exist_ok=True)

            copyfile("hparams.py", os.path.join(root_path, "hparams.py"))

if __name__ == "__main__":
    print(to_categorical(3, 5))