import os
import time
import datetime

from model import Generator, SpeakerEncoderOnehot
from hparams import hparams
import lrschedule
import torch
import torch.backends.cudnn as cudnn

class Solver(object):
    """ Solver for training and testing AutoVC """
    def __init__(self, train_loader, test_loader, config=None):
        """ Initialize configurations. """
        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Training configurations
        self.batch_size = hparams.batch_size
        self.beta1 = hparams.adam_beta1
        self.beta2 = hparams.adam_beta2
        self.adam_eps = hparams.adam_eps
        self.current_lr = hparams.initial_learning_rate
        self.global_step = 0
        self.global_epoch = 0

        # Miscellaneous
        self.use_tensorboard = hparams.use_tensorboard

        # CUDA settings
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            cudnn.benchmark = False
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Build the model.
        self.build_model()

        # Log settings
        self.log_step = hparams.log_step

        if config.exp_name:
            self.log_dir = os.path.join(config.log_dir, config.exp_name, 'log')
            self.model_dir = os.path.join(config.log_dir, config.exp_name, 'model')
        else:
            print('Please set exp_name & log_dir')
            exit(-1)

        # Save settings
        self.checkpoint_interval = hparams.checkpoint_interval

        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """ Create a content encoder and a decoder """
        self.speaker_enc = SpeakerEncoderOnehot().to(self.device)
        self.G = Generator(hparams.dim_neck, hparams.dim_emb, hparams.dim_pre, hparams.freq).to(self.device)
        self.optimizer = torch.optim.Adam(self.G.parameters(),
                                          lr=hparams.initial_learning_rate,
                                          betas=(hparams.adam_beta1, hparams.adam_beta2),
                                          eps=hparams.adam_eps,
                                          weight_decay=hparams.weight_decay,
                                          amsgrad=hparams.amsgrad)

        self.print_network(self.G, 'G')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def reset_grad(self):
        self.optimizer.zero_grad()

    def train(self):
        """ Train AutoVC. """
        # Set data loader.
        train_loader = self.train_loader

        while self.global_epoch < hparams.nepochs:
            print(len(train_loader))
            for mels, labels, labels_onehot in train_loader:
                start_time = time.time()

                self.speaker_enc.train()
                self.G.train()

                if hparams.lr_schedule is not None:
                    lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                    self.current_lr = lr_schedule_f(hparams.initial_learning_rate, self.global_step, **hparams.lr_schedule_kwargs)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.current_lr
                self.optimizer.zero_grad()

                mels = mels.to(self.device)
                labels = labels.to(self.device)

                spk_embed = self.speaker_enc(labels)
                mel_outputs, mel_outputs_postnet, code = self.G(mels, spk_embed, spk_embed)

                # Calculate loss
                loss_init_recon = torch.nn.functional.mse_loss(mel_outputs, mels)
                loss_recon = torch.nn.functional.mse_loss(mel_outputs_postnet, mels)

                recon_code = self.G(mel_outputs_postnet, spk_embed, None)

                loss_content = torch.nn.functional.l1_loss(recon_code, code)

                loss_total = loss_init_recon + loss_recon + loss_content

                # Back prop
                self.reset_grad()
                loss_total.backward()
                self.optimizer.step()

                # Logging
                loss = {}
                loss['loss_init_recon'] = loss_init_recon.item()
                loss['loss_recon'] = loss_recon.item()
                loss['loss_content'] = loss_content.item()
                loss['loss_total'] = loss_total.item()

                if (self.global_step+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}]".format(et, self.global_step+1)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, self.global_step + 1)

                if (self.global_step+1) % self.checkpoint_interval == 0:
                    G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(self.global_step + 1))
                    speaker_enc_path = os.path.join(self.model_dir, '{}-spk-enc.ckpt'.format(self.global_step + 1))
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.speaker_enc.state_dict(), speaker_enc_path)
                    print('Saved model checkpoints into {}...'.format(self.model_dir))

                self.global_step = self.global_step + 1
            self.global_epoch = self.global_epoch + 1