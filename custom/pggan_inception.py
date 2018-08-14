
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
import cupy
import cv2

import argparse

from custom.inception_score import Inception
from custom.inception_score import inception_score

from custom.net import Generator
#import config

# In[2]:


xp = cuda.cupy

# In[9]:
def calc_inception(gen, samples, stage, gpu):
    
    create_num = samples
    level = stage
    
    ims = []
    sum = 0
    seed = 0

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
        
        imgs = cuda.to_cpu(x.data)
        imgs = np.transpose(imgs, (0, 3, 2, 1))
        img = 255 * 0.5 * (imgs[0]+1.0)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.transpose(img, (2, 0, 1))
        ims.append(img)
        
        sum+=1
            
    ims = np.asarray(ims, dtype=np.float32)
    #train, test = datasets.get_cifar10(ndim=3, withlabel=False, scale=255.0)
    #del gen

    # Use all 60000 images, unless the number of samples are specified
    #ims = np.concatenate((train, test))
    if samples > 0:
        ims = ims[:samples]
    print(ims.shape)
    
    model = Inception()
    serializers.load_hdf5("custom/inception_score.model", model)

    if gpu >= 0:
        #cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        mean, std = inception_score(model, ims)
        
    del model

    #print('Inception score mean:', mean)
    #print('Inception score std:', std)
    
    return mean, std

def main():
    init = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gpu', '-g', type=int, default=0,help='GPU ID')
    #parser.add_argument('--create', '-c', type=int, default=100,help='create num')
    parser.add_argument('--level', '-l', type=float, default=12.0,help='level num')
    parser.add_argument('--load_gen_g_model', default='generator.npz',help='load generator model')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--inception_model', type=str, default='inception_score.model')

    args = parser.parse_args()
    print(args)

    seed=args.seed
    
    gen = Generator(128)
    serializers.load_npz(args.load_gen_g_model, gen)

    gpu = args.gpu
    chainer.cuda.get_device(gpu).use()

    gen.to_gpu(gpu) # GPUを使うための処理
    
    create_num = args.samples
    level = args.level
    
    ims = []
    sum = 0

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
        
        imgs = cuda.to_cpu(x.data)
        imgs = np.transpose(imgs, (0, 3, 2, 1))
        img = 255 * 0.5 * (imgs[0]+1.0)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        ims.append(img)
        
        sum+=1
            
    ims = np.asarray(ims, dtype=np.float32)
    #train, test = datasets.get_cifar10(ndim=3, withlabel=False, scale=255.0)
    del gen

    # Use all 60000 images, unless the number of samples are specified
    #ims = np.concatenate((train, test))
    if args.samples > 0:
        ims = ims[:args.samples]
    print(ims.shape)
    
    model = Inception()
    serializers.load_hdf5(args.inception_model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        mean, std = inception_score(model, ims)

    print('Inception score mean:', mean)
    print('Inception score std:', std)
    
    
if __name__ == '__main__':
    main()
