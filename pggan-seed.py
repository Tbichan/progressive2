
# coding: utf-8

# In[1]:


import os
import math
import random
import numpy as np
import chainer
from chainer import cuda, Chain, optimizers, Variable, serializers
import chainer.functions as F
import chainer.links as L

import cv2
import argparse

from custom.net import Generator
import custom.config

# シード値設定
#np.random.seed(seed=20180101)
def vs(v):
    str = ""
    for i in range(100):
        n = int((v[i] + 1.0)*100)
        h = hex(n)
        tmp = h[2:]
        if len(tmp) == 1:
            tmp = "0" + tmp
        tmp = tmp.upper()
        str = str + tmp
    return str


# In[9]:

def main():
    init = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gpu', '-g', type=int, default=0,help='GPU ID')
    parser.add_argument('--create', '-c', type=int, default=100,help='create num')
    parser.add_argument('--load_gen_g_model', default='generator_smooth.npz',help='load generator model')
    parser.add_argument('--level', '-l', type=float, default=12.0,help='GPU ID')
    parser.add_argument('--senga', type=int, default=0,help='senga enable?')
    args = parser.parse_args()
    print(args)

    seed=args.seed
    max_stage = custom.config.train_params['max_stage']
    n_hidden = custom.config.network_params['latent_in']
    gen = Generator(n_hidden=n_hidden, max_stage=max_stage)
    serializers.load_npz(args.load_gen_g_model, gen)

    
    gpu = args.gpu
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        gen.to_gpu(gpu) # GPUを使うための処理
        import cupy
        xp = cuda.cupy
    else:
        xp = np
    
    create_num = args.create
    level = args.level
    senga = args.senga

    for i in range(create_num):
        #zInput = np.random.uniform(-1.0, 1.0, size=[1, 100]).astype(np.float32)
    
        xp.random.seed(seed=i+seed)
        zInput = gen.make_hidden(1)
    
        # Chainer変数に変換
        z = Variable(xp.asarray(zInput, dtype=np.float32))
    
        # テストモードに
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            # 偽物用
            x = gen(z, level)
        if gpu >= 0:
            imgs = cuda.to_cpu(x.data)
        else:
            imgs = x.data
            
        imgs = np.transpose(imgs, (0, 3, 2, 1))
    
        gene = vs(zInput[0])

        #cv2.imwrite(''+gene+'.jpg', 255 * 0.5 * (imgs[0]+1.0))
        if senga == 1:
            cv2.imwrite('seed/'+str(i+seed)+'.jpg', 255 * 0.5 * (imgs[0,:,:,3]+1.0))
        elif senga == -1:
            cv2.imwrite('seed/'+str(i+seed)+'.jpg', 255 - 255 * 0.5 * (imgs[0,:,:,3]+1.0))
        else:
            cv2.imwrite('seed/'+str(i+seed)+'.jpg', 255 * 0.5 * (imgs[0,:,:,:3]+1.0))
        print(str(i+1))
        #break
    
if __name__ == '__main__':
    main()
