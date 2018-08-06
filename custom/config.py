train_params = dict(gpu=1,
                   width = 512,
                   height = 512,
                   init_iter = 0,
                   interval = 400000,
                   max_stage=14,
                   inception_interval = 2500,
                   inception = False,
                   separate = True,                # Genの学習のzを切り替えるかどうか
                   minibatch_repeats = 4,          # Minibatchリピート回数
                   batches = (16,16,16,16,16,16,8,2),
                   smoothing = 0.999,
                   test_interval = 100,
                   save_interval = 5000,
                   image_mirror = False,
                   senga = False,
                   )

loss_params = dict(iwass = 1.0
                   )

network_params = dict(latent_in = 512,
                      G_filters = (512,512,512,512,256,128,64,32),
                      D_filters = (32,64,128,256,512,512,512,512),
                      image_ch = 3,
                      slope=0.2
                      )
