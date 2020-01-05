import tensorflow as tf
import numpy as np

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    name="wavenet_vocoder",

    # Convenient model builder
    builder="wavenet",

    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw [-1, 1]
    # 3. mulaw-quantize [0, mu]
    # If input_type is raw or mulaw, network assumes scalar input and
    # discretized mixture of logistic distributions output, otherwise one-hot
    # input and softmax output are assumed.
    # **NOTE**: if you change the one of the two parameters below, you need to
    # re-run preprocessing before training.
    input_type="raw",
    quantize_channels=65536,  # 65536 or 256

    # Audio:
    sample_rate=16000,
    # this is only valid for mulaw is True
    silence_threshold=2,
    num_mels=80,
    fmin=125,
    fmax=7600,
    fft_size=1024,
    win_size=1024,

    # Audio for speaker encoder
    sr = 16000,
    nfft = 1024,
    window = 0.025, # (s) (25ms)
    hop = 0.01, # (s) (10ms)
    nmels=40,
    sv_frame = 180, # Max number of time steps in input after preprocess

    preemphasize = False, #whether to apply filter
    preemphasis = 0.97,

    # shift can be specified by either hop_size or frame_shift_ms
    hop_size=256,
    frame_shift_ms=None,
    min_level_db=-100,
    ref_level_db=20,
    # whether to rescale waveform or not.
    # Let x is an input waveform, rescaled waveform y is given by:
    # y = x / np.abs(x).max() * rescaling_max
    rescaling=True,
    rescaling_max=0.999,
    # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
    # happen depends on min_level_db and ref_level_db, causing clipping noise.
    # If False, assertion is added to ensure no clipping happens.o0
    signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,
    symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast convergence)

    # Mixture of logistic distributions:
    log_scale_min=float(-32.23619130191664),

    trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    trim_fft_size = 1024,
    trim_hop_size = 256,
    trim_top_db = 25,

    power = 1.0,

    # Model:
    # This should equal to `quantize_channels` if mu-law quantize enabled
    # otherwise num_mixture * 3 (pi, mean, log_scale)
    out_channels=10 * 3,
    layers=24,
    stacks=4,
    residual_channels=512,
    gate_channels=512,  # split into 2 gropus internally for gated activation
    skip_out_channels=256,
    dropout=1 - 0.95,
    kernel_size=3,
    # If True, apply weight normalization as same as DeepVoice3
    weight_normalization=True,
    # Use legacy code or not. Default is True since we already provided a model
    # based on the legacy code that can generate high-quality audio.
    # Ref: https://github.com/r9y9/wavenet_vocoder/pull/73
    legacy=True,

    # Speaker encoder parameters
    hidden=768,
    num_layer=2,
    proj=256,

    # Local conditioning (set negative value to disable))
    cin_channels=80,
    # If True, use transposed convolutions to upsample conditional features,
    # otherwise repeat features to adjust time resolution
    upsample_conditional_features=True,
    # should np.prod(upsample_scales) == hop_size
    upsample_scales=[4, 4, 4, 4],
    # Freq axis kernel size for upsampling network
    freq_axis_kernel_size=3,

    # Global conditioning (set negative value to disable)
    # currently limited for speaker embedding
    # this should only be enabled for multi-speaker datasets
    gin_channels=-1,  # i.e., speaker embedding dim
    n_speakers=-1,  

    # Data loader
    pin_memory=True,
    num_workers=2,

    # train/test
    # test size can be specified as portion or num samples
    test_size=0.0441,  # 50 for CMU ARCTIC single speaker
    test_num_samples=None,
    random_state=1234,

    # Loss

    # Training:
    batch_size=2,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    amsgrad=False,
    initial_learning_rate=1e-3,
    # see lrschedule.py for available lr_schedule
    lr_schedule="noam_learning_rate_decay",
    lr_schedule_kwargs={},  # {"anneal_rate": 0.5, "anneal_interval": 50000},
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=-1,
    # max time steps can either be specified as sec or steps
    # if both are None, then full audio samples are used in a batch
    max_time_sec=None,
    max_time_steps=8000,
    # Hold moving averaged parameters and use them for evaluation
    exponential_moving_average=True,
    # averaged = decay * averaged + (1 - decay) * x
    ema_decay=0.9999,

    # Save
    # per-step intervals
    checkpoint_interval=10000,
    train_eval_interval=10000,
    # per-epoch interval
    test_eval_epoch_interval=5,
    save_optimizer_state=True,

    # Eval:
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
