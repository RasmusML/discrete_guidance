import ml_collections
import os


def get_cifar10_train_config(parent_dir):
    """
    Config for training denoising model (same as Campbell)
    """
    # Directory where model outputs will be saved
    save_dir = os.path.join(parent_dir, "outputs")
    # Directory where the CIFAR10 dataset is stored
    data_dir = os.path.join(parent_dir, "data")

    config = ml_collections.ConfigDict()
    config.experiment_name = "cifar10"
    config.save_location = save_dir
    config.state = "train"

    config.init_model_path = None

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 1
    config.eps_ratio = 1e-9

    # Configs for loss computation
    config.loss = loss = ml_collections.ConfigDict()
    loss.nll_weight = 0.001
    loss.min_time = 0.01

    # Configs for training setup
    config.training = training = ml_collections.ConfigDict()
    training.n_iters = 2000000
    training.clip_grad = True
    training.warmup = 5000

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4

    # Configs for training data
    config.data = data = ml_collections.ConfigDict()
    data.name = "DiscreteCIFAR10"
    data.root = data_dir
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 128  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [3, 32, 32]
    data.random_flips = True

    # Define denoising model
    config.model = model = ml_collections.ConfigDict()
    # model.ema_decay = 0.9999 #0.9999

    model.num_classes = 10
    model.ch = 128
    model.num_res_blocks = 2
    model.num_scales = 4
    model.ch_mult = [1, 2, 2, 2]
    model.input_channels = 3
    model.scale_count_to_put_attn = 1
    model.data_min_max = [0, 255]
    model.dropout = 0.1
    model.skip_rescale = True
    model.time_embed_dim = model.ch
    model.time_scale_factor = 1000
    model.fix_logistic = False

    # Configs forward process
    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exponential = 100.0
    model.time_base = 3.0

    # Configs for logging
    config.saving = saving = ml_collections.ConfigDict()

    saving.enable_preemption_recovery = False
    saving.preemption_start_day_YYYYhyphenMMhyphenDD = None
    saving.checkpoint_freq = 1000
    saving.num_checkpoints_to_keep = 2
    saving.checkpoint_archive_freq = 200000
    saving.log_low_freq = 1000
    saving.low_freq_loggers = ["denoisingImages"]
    saving.prepare_to_resume_after_timeout = False

    return config


def get_cifar10_noisy_classifier_train_config(parent_dir):
    """
    Configs for training noisy classifier
    """
    # Directory where model outputs will be saved
    save_dir = os.path.join(parent_dir, "outputs")
    # Directory where the CIFAR10 dataset is stored
    data_dir = os.path.join(parent_dir, "data")

    config = ml_collections.ConfigDict()
    config.experiment_name = "cifar10"
    config.save_location = save_dir
    config.state = "train"

    config.init_model_path = None

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 1
    config.eps_ratio = 1e-9

    # Configs for loss computation
    config.loss = loss = ml_collections.ConfigDict()
    loss.min_time = 0.01

    # Configs for training setup
    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"
    training.n_iters = 500000
    training.clip_grad = True
    training.warmup = 5000

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4

    # Configs for training data
    config.data = data = ml_collections.ConfigDict()
    data.name = "DiscreteCIFAR10"
    data.root = data_dir
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 128  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [3, 32, 32]
    data.random_flips = True
    data.categorical = False

    # Define classifier model
    config.model = model = ml_collections.ConfigDict()

    model.num_classes = 10
    model.ch = 128
    model.num_res_blocks = 2
    model.num_scales = 4
    model.ch_mult = [1, 2, 2, 2]
    model.input_channels = 3
    model.scale_count_to_put_attn = 1
    model.data_min_max = [0, 255]
    model.dropout = 0.1
    model.skip_rescale = True
    model.time_embed_dim = model.ch
    model.time_scale_factor = 1000

    # Configs forward process
    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exponential = 100.0
    model.time_base = 3.0

    # Configs for logging
    config.saving = saving = ml_collections.ConfigDict()

    saving.enable_preemption_recovery = False
    saving.preemption_start_day_YYYYhyphenMMhyphenDD = None
    saving.checkpoint_freq = 5000
    saving.num_checkpoints_to_keep = 1
    saving.checkpoint_archive_freq = 10000

    saving.prepare_to_resume_after_timeout = False

    return config


def get_cifar10_test_config(
    parent_dir,
    save_samples_dir=None,
    sampler_name="tau_leaping",
):
    """
    Config for sampling from trained denoising model and noisy classifier

    Args:
        parent_dir (str): Path to the parent directory for the cifar10 experiment
        save_samples_dir (str): If not None, the directory where image samples
            will be saved
    """
    # Directory where model checkpoints are be saved
    output_dir = os.path.join(parent_dir, "outputs")
    # Directory where the CIFAR10 dataset is stored
    data_dir = os.path.join(parent_dir, "data")

    config = ml_collections.ConfigDict()

    if save_samples_dir:
        config.save_samples_dir = save_samples_dir

    model_weights_dir = os.path.join(parent_dir, "model_weights")

    # Denoising model locations
    denoising_model_location = os.path.join(
        model_weights_dir, "denoising_model", "model_ckpt.pt"
    )
    denoising_model_config_location = os.path.join(
        model_weights_dir, "denoising_model", "config.yaml"
    )
    config.denoising_model_train_config_path = denoising_model_config_location
    config.denoising_model_checkpoint_path = denoising_model_location

    # Predictive model locations
    property_model_location = os.path.join(
        model_weights_dir, "noisy_classifier", "model_ckpt.pt"
    )
    property_model_config_location = os.path.join(
        model_weights_dir, "noisy_classifier", "config.yaml"
    )
    config.property_model_train_config_path = property_model_config_location
    config.property_model_checkpoint_path = property_model_location

    config.eval_name = "CIFAR10"
    config.train_config_overrides = [
        [["device"], "cuda"],
        [["data", "root"], data_dir],
        [["distributed"], False],
    ]

    config.state = "eval"
    config.device = "cuda"
    config.eps_ratio = 1e-9

    config.data = data = ml_collections.ConfigDict()
    data.name = "DiscreteCIFAR10"
    data.root = data_dir
    data.train = False
    data.download = True
    data.S = 256
    data.batch_size = 128
    data.shuffle = False
    data.shape = [3, 32, 32]
    data.random_flips = False
    data.categorical = False

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = sampler_name
    sampler.num_steps = 500
    sampler.min_t = 0.01
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.1

    return config
