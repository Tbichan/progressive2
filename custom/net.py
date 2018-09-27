import numpy as np

import math
import chainer
import chainer.functions as F
from chainer.functions.activation.leaky_relu import leaky_relu
import chainer.links as L
from custom.custom_layer import EqualizedConvolution2d, EqualizedLinear
    
class GeneratorBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        super(GeneratorBlock, self).__init__()
        self.out_ch = out_ch
        irelu = chainer.initializers.HeNormal(scale=1.0)
        
        with self.init_scope():
            self.c0 = EqualizedConvolution2d(in_ch, out_ch, 3, 1, 1, initialW=irelu)
            self.c1 = EqualizedConvolution2d(out_ch, out_ch, 3, 1, 1, initialW=irelu)
            
    def __call__(self, x):
        h = F.unpooling_2d(x, 2, 2, 0, outsize=(x.shape[2]*2, x.shape[3]*2))
        h = feature_vector_normalization(leaky_relu(self.c0(h)))
        h = feature_vector_normalization(leaky_relu(self.c1(h)))
        return h

    
class DiscriminatorBlock(chainer.Chain):
    # conv-conv-downsample
    def __init__(self, in_ch, out_ch):
        super(DiscriminatorBlock, self).__init__()
        #self.pooling_comp = pooling_comp
        self.out_ch = out_ch
        irelu = chainer.initializers.HeNormal(scale=1.0)
        
        with self.init_scope():
            self.c0 = EqualizedConvolution2d(in_ch, in_ch, 3, 1, 1, initialW=irelu)
            self.c1 = EqualizedConvolution2d(in_ch, out_ch, 3, 1, 1, initialW=irelu)
            
    def __call__(self, x):
        #
        h = leaky_relu(self.c0(x))
        h = leaky_relu(self.c1(h))
        h = F.average_pooling_2d(h, 2, 2, 0)
        return h
    
def feature_vector_normalization(x, eps=1e-8):
    # x: (B, C, H, W)
    alpha = F.rsqrt(F.mean(x**2, axis=1, keepdims=True) + eps)
    return F.broadcast_to(alpha, x.data.shape) * x


def minibatch_std(x):
    m = F.mean(x, axis=0, keepdims=True)
    tmp = (x - F.broadcast_to(m, x.shape))
    v = F.mean(tmp**2, axis=0, keepdims=True)
    std = F.mean(F.sqrt(v + 1e-8), keepdims=True)
    std = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
    return F.concat([x, std], axis=1)

class Generator(chainer.Chain):
    def __init__(self, n_hidden=512, max_stage=12):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.R = (128,128,128,128,128,64,32,16)
        self.max_stage = max_stage
        irelu = chainer.initializers.HeNormal(scale=1.0)
        ilinear = chainer.initializers.HeNormal(scale=1.0/np.sqrt(2))
        with self.init_scope():
            
            self.c0 = EqualizedConvolution2d(n_hidden, self.R[0], 4, 1, 3, initialW=irelu)
            self.c1 = EqualizedConvolution2d(self.R[0], self.R[0], 3, 1, 1, initialW=irelu)
            self.out0 = EqualizedConvolution2d(self.R[0], 3, 1, 1, 0, initialW=ilinear)
            self.b1 = GeneratorBlock(self.R[0], self.R[1])
            self.out1 = EqualizedConvolution2d(self.R[1], 3, 1, 1, 0, initialW=ilinear)
            self.b2 = GeneratorBlock(self.R[1], self.R[2])
            self.out2 = EqualizedConvolution2d(self.R[2], 3, 1, 1, 0, initialW=ilinear)
            self.b3 = GeneratorBlock(self.R[2], self.R[3])
            self.out3 = EqualizedConvolution2d(self.R[3], 3, 1, 1, 0, initialW=ilinear)
            self.b4 = GeneratorBlock(self.R[3], self.R[4])
            self.out4 = EqualizedConvolution2d(self.R[4], 3, 1, 1, 0, initialW=ilinear)
            self.b5 = GeneratorBlock(self.R[4], self.R[5])
            self.out5 = EqualizedConvolution2d(self.R[5], 3, 1, 1, 0, initialW=ilinear)
            self.b6 = GeneratorBlock(self.R[5], self.R[6])
            self.out6 = EqualizedConvolution2d(self.R[6], 3, 1, 1, 0, initialW=ilinear)
            self.b7 = GeneratorBlock(self.R[6], self.R[7])
            self.out7 = EqualizedConvolution2d(self.R[7], 3, 1, 1, 0, initialW=ilinear)
            
            

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/float(self.n_hidden) + 1e-8)
        return z

    def __call__(self, z, stage):
        # stage0: c0->c1->out0
        # stage1: c0->c1-> (1-a)*(up->out0) + (a)*(b1->out1)
        # stage2: c0->c1->b1->out1
        # stage3: c0->c1->b1-> (1-a)*(up->out1) + (a)*(b2->out2)
        # stage4: c0->c1->b2->out2
        # ...
        #print(np.prod(self.c0.c.b.data.shape))
        
        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)
        
        h = F.reshape(z,(-1, self.n_hidden, 1, 1))
        h = feature_vector_normalization(leaky_relu(self.c0(h)))
        h = feature_vector_normalization(leaky_relu(self.c1(h)))

        for i in range(1, int(stage//2+1)):
            h = getattr(self, "b%d"%i)(h)

        if int(stage)%2==0:
            out = getattr(self, "out%d"%(stage//2))
            x = out(feature_vector_normalization(h))
        else:
            out_prev = getattr(self, "out%d"%(stage//2))
            out_curr = getattr(self, "out%d"%(stage//2+1))
            b_curr = getattr(self, "b%d"%(stage//2+1))

            x_0 = out_prev(F.unpooling_2d(h, 2, 2, 0, outsize=(2*h.shape[2], 2*h.shape[3])))
            x_1 = out_curr(b_curr(h))
            x = (1.0-alpha)*x_0 + alpha*x_1

        if chainer.configuration.config.train:
            return x
        else:
            scale = int(32 // x.data.shape[2])
            return F.unpooling_2d(x, scale, scale, 0, outsize=(32,32))


    
class Discriminator(chainer.Chain):
    def __init__(self, max_stage=12):
        super(Discriminator, self).__init__()
        self.max_stage = max_stage
        #self.pooling_comp = pooling_comp # compensation of ave_pool is 0.5-Lipshitz
        
        self.R = (16,32,64,128,128,128,128,128)
        irelu = chainer.initializers.HeNormal(scale=1.0)
        ilinear = chainer.initializers.HeNormal(scale=1.0/np.sqrt(2))
        with self.init_scope():
            self.in7 = EqualizedConvolution2d(3, self.R[0], 1, 1, 0, initialW=irelu)
            self.b7 = DiscriminatorBlock(self.R[0], self.R[1])
            self.in6 = EqualizedConvolution2d(3, self.R[1], 1, 1, 0, initialW=irelu)
            self.b6 = DiscriminatorBlock(self.R[1], self.R[2])
            self.in5 = EqualizedConvolution2d(3, self.R[2], 1, 1, 0, initialW=irelu)
            self.b5 = DiscriminatorBlock(self.R[2], self.R[3])
            self.in4 = EqualizedConvolution2d(3, self.R[3], 1, 1, 0, initialW=irelu)
            self.b4 = DiscriminatorBlock(self.R[3], self.R[4])
            self.in3 = EqualizedConvolution2d(3, self.R[4], 1, 1, 0, initialW=irelu)
            self.b3 = DiscriminatorBlock(self.R[4], self.R[5])
            self.in2 = EqualizedConvolution2d(3, self.R[5], 1, 1, 0, initialW=irelu)
            self.b2 = DiscriminatorBlock(self.R[5], self.R[6])
            self.in1 = EqualizedConvolution2d(3, self.R[6], 1, 1, 0, initialW=irelu)
            self.b1 = DiscriminatorBlock(self.R[6], self.R[7])
            self.in0 = EqualizedConvolution2d(3, self.R[7], 1, 1, 0, initialW=irelu)

            self.out0 = EqualizedConvolution2d(self.R[7]+1, self.R[7], 3, 1, 1, initialW=irelu)
            self.out1 = EqualizedConvolution2d(self.R[7], self.R[7], 4, 1, 0, initialW=irelu)
            self.out2 = EqualizedLinear(self.R[7], 1, initialW=ilinear)

    def __call__(self, x, stage):
        # stage0: in0->m_std->out0_0->out0_1->out0_2
        # stage1: (1-a)*(down->in0) + (a)*(in1->b1) ->m_std->out0->out1->out2
        # stage2: in1->b1->m_std->out0_0->out0_1->out0_2
        # stage3: (1-a)*(down->in1) + (a)*(in2->b2) ->b1->m_std->out0->out1->out2
        # stage4: in2->b2->b1->m_std->out0->out1->out2
        # ...

        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if int(stage)%2==0:
            fromRGB = getattr(self, "in%d"%(stage//2))
            h = leaky_relu(fromRGB(x))
        else:
            fromRGB0 = getattr(self, "in%d"%(stage//2))
            fromRGB1 = getattr(self, "in%d"%(stage//2+1))
            b1 = getattr(self, "b%d"%(stage//2+1))


            h0 = leaky_relu(fromRGB0(F.average_pooling_2d(x, 2, 2, 0)))
            h1 = b1(leaky_relu(fromRGB1(x)))
            h = (1-alpha)*h0 + alpha*h1

        for i in range(int(stage // 2), 0, -1):
            h = getattr(self, "b%d" % i)(h)

        h = minibatch_std(h)
        h = leaky_relu(self.out0(h))
        h = leaky_relu(self.out1(h))
        return self.out2(h)
    
def copy_param(target_link, source_link):
    """Copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] = param.data

    # Copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] = link.avg_mean
            target_bn.avg_var[:] = link.avg_var


def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] *= (1 - tau)
        target_params[param_name].data[:] += tau * param.data

    # Soft-copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] *= (1 - tau)
            target_bn.avg_mean[:] += tau * link.avg_mean
            target_bn.avg_var[:] *= (1 - tau)
            target_bn.avg_var[:] += tau * link.avg_var
