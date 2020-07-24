class Config(object):
    c_dim = 5
    c2_dim = 8
    celeba_crop_size = 178
    rafd_crop_size = 256
    image_size = 128
    g_conv_dim = 64
    d_conv_dim = 64
    g_repeat_num = 6
    d_repeat_num = 6
    lambda_cls = 1
    lambda_rec = 10
    lambda_gp = 10

    dataset = 'CelebA'
    batch_size = 16
    num_iters = 200000
    num_iters_decay = 100000
    g_lr = 0.0001
    d_lr = 0.0001
    n_critic = 5
    beta1 = 0.5
    beta2 = 0.999
    resume_iters = None
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    
    # Test configuration.
    test_iters = 200000

    # Miscellaneous.
    num_workers = 1
    mode = 'train'
    use_tensorboard = True

    # Directories.
    celeba_image_dir = 'data/CelebA_nocrop/images'
    attr_path = 'data/list_attr_celeba.txt'
    rafd_image_di = 'data/RaFD/train'
    log_dir = 'stargan/logs'
    model_save_dir = 'stargan/models'
    sample_dir = 'stargan/samples'
    result_dir = 'stargan/results'

    # Step size.
    log_step = 10
    sample_step = 1000
    model_save_step = 10000
    lr_update_step = 1000

config = Config()