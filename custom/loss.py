import numpy as np

import math
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
import custom.config

iwass = custom.config.loss_params['iwass']

def D_loss(G, D, stage, reals, latents, batch_size):
    
    xp = G.xp
    
    # real
    y_real = D(reals, stage=stage)
    
    # latents
    x_fake = G(latents, stage=stage)
    y_fake = D(x_fake, stage=stage)
    
    x_fake.unchain_backward()
    
    real_loss = F.sum(-y_real) / batch_size
    fake_loss = F.sum(y_fake) / batch_size
    d_w_loss = real_loss + fake_loss
    
    
    
    wass = -fake_loss.data - real_loss.data
    
    # gradient_penalty
    eps = xp.random.uniform(0., 1., size=(reals.shape)).astype("f")
    x_mid = eps * reals.data + (1.0 - eps) * x_fake.data
    
    x_mid_v = Variable(x_mid)
    
    dydx, = chainer.grad([D(x_mid_v, stage=stage)], [x_mid_v], enable_double_backprop=True)
    dydx = F.sqrt(F.batch_l2_norm_squared(dydx))
    
    # gradient_penalty_loss
    grad_loss = 10.0 * F.mean_squared_error(dydx, iwass * xp.ones_like(dydx.data, dtype=np.float32)) * (1.0/iwass**2)
    
    # drift_loss
    drift_loss = 0.001 * F.sum(y_real**2) / batch_size
    
    return d_w_loss, grad_loss, drift_loss, wass
    
    
def G_loss(G, D, stage, latents, batch_size):
    
    xp = G.xp
    
    x_fake = G(latents, stage=stage)
    y_fake = D(x_fake, stage=stage)

    g_loss = F.sum(-y_fake) / batch_size
    
    return g_loss