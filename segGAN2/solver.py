from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
from torch import distributions
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import sys
import until
from loss import naive_cross_entropy_loss

class Solver:

    def __init__(self, loader):
        
        self.loader = loader

        self.c_dim = 4
        
        self.lambda_cls = 10.0
        self.lambda_rec = 10.0
        self.lambda_gp = 10.0

        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.n_critic = 6
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.smooth_beta = 0.999
        
        self.model_save_step = 1000
        self.lr_update_step = 1000

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_size = 256

        self.num_iters = 200000
        self.num_iters_decay = 100000

        self.log_step = 10
        self.sample_step = 10

        # Directories.
        self.log_dir = "log"
        self.sample_dir = "sample"
        self.model_save_dir = "model"
        self.result_dir = "result"
        
        # colors
        self.colors = until.colors
        self.void_classes = until.void_classes
        self.valid_classes = until.valid_classes
        self.class_names = until.class_names
        self.ignore_index = until.ignore_index

        self.n_classes = until.n_classes

        self.label_colours = dict(zip(range(19), self.colors))

        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.class_names = dict(zip(self.class_names, range(19)))
        print(self.class_names)

        self.build_model()

    def build_model(self):
        self.G = Generator(conv_dim=64, c_dim=self.c_dim)
        self.G_test = Generator(conv_dim=64, c_dim=self.c_dim)
        self.D = Discriminator(self.image_size, 64, self.c_dim)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        #self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.g_lr, alpha=0.99, eps=1e-8)
        #self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.d_lr, alpha=0.99, eps=1e-8)
        
        self.G.to(self.device)
        self.G_test.to(self.device)
        self.D.to(self.device)

        self.update_average(self.G_test, self.G, 0.)

    def eval_model(self):
        self.G.eval()
        self.G_test.eval()
        self.D.eval()

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        G_test_path = os.path.join(self.model_save_dir, '{}-G_test.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.G_test.load_state_dict(torch.load(G_test_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = torch.flip(x, [1])
        #out = (x + 1) / 2
        return out.clamp_(0, 1)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def update_average(self, model_tgt, model_src, beta):
        toogle_grad(model_src, False)
        toogle_grad(model_tgt, False)

        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    def get_zdist(self, dist_name, dim, device=None):
        # Get distribution
        if dist_name == 'uniform':
            low = -torch.ones(dim, device=device)
            high = torch.ones(dim, device=device)
            zdist = distributions.Uniform(low, high)
        elif dist_name == 'gauss':
            mu = torch.zeros(dim, device=device)
            scale = torch.ones(dim, device=device)
            zdist = distributions.Normal(mu, scale)
        else:
            raise NotImplementedError

        # Add dim attribute
        zdist.dim = dim

        return zdist

    def getBatch(self):
        try:
            x_real, label_org = next(self.data_iter)
        except:
            while True:
                try:
                    self.data_iter = iter(self.loader)
                    x_real, label_org = next(self.data_iter)
                    break
                except:
                    #a=0/0
                    pass
        return x_real, label_org

    def onehot(self, label):
        
        label = label.numpy()
        
        label_onehot = np.zeros((label.shape[0],self.n_classes,label.shape[1],label.shape[2])).astype(np.uint8)
        #print(label_onehot)
        for i in range(self.n_classes):
            label_onehot[:,i,:,:] = (label == i)
            
        #print(np.max(label_onehot))
        label_onehot = torch.from_numpy(label_onehot)
        #print(label_onehot.shape)
        return label_onehot.to(self.device)
    
    def to_label(self, label_onehot):
        
        label = np.zeros((label_onehot.shape[0],1,label_onehot.shape[2],label_onehot.shape[3])).astype(np.uint8)
        label[:,0,:,:] = np.argmax(label_onehot, axis=1)
        label = torch.from_numpy(label)

        return label

    def vis(self, real, label_onehot):
        label = self.to_label(label_onehot)
        label = label.numpy()

        label_colors = np.zeros((label.shape[0],3,label.shape[2],label.shape[3])).astype(np.uint8)

        r = label.copy()
        g = label.copy()
        b = label.copy()

        for l in range(0, self.n_classes):
            r[label == l] = self.label_colours[l][0]
            g[label == l] = self.label_colours[l][1]
            b[label == l] = self.label_colours[l][2]

        r = np.reshape(r, ((label.shape[0], label.shape[2], label.shape[3])))
        g = np.reshape(g, ((label.shape[0], label.shape[2], label.shape[3])))
        b = np.reshape(b, ((label.shape[0], label.shape[2], label.shape[3])))

        rgb = np.zeros((label.shape[0], 3, label.shape[2], label.shape[3]))
        rgb[:, 0, :, :] = r / 255.0
        rgb[:, 1, :, :] = g / 255.0
        rgb[:, 2, :, :] = b / 255.0

        rgb = torch.from_numpy(rgb)
        save_image(rgb, "label.jpg", nrow=1, padding=0)
        save_image(self.denorm(real.data.cpu()), "real.jpg", nrow=1, padding=0)
        #print(label)


    # それぞれのラベルが何％を占めているか
    def label_contain_persent(self, label, index=None):
        #num_labels=255
        label_per = np.zeros((label.shape[0],self.n_classes,1,1)).astype(np.float32)
        Ns = torch.sum(label<self.n_classes, (1,2), dtype=torch.float32)
        
        if index is None:
            for i in range(self.n_classes):
                label_per[:,i,0,0] = torch.sum(label==i, (1,2), dtype=torch.float32) / Ns 
        else:
            for i in range(len(index)):
                #print(torch.sum(label==index[i], (1,2), dtype=torch.float32))
                label_per[i,index[i],0,0] = torch.sum(label==index[i], (1,2), dtype=torch.float32)[i] / Ns[i]
        label_per = torch.from_numpy(label_per)
        
        #print(torch.sum(label_per, (1)))
        return label_per.view(label_per.size()[0], -1).to(self.device)

    def train(self, start_iter=0):

        g_lr = self.g_lr
        d_lr = self.d_lr

        zdist = None

        BCELoss = torch.nn.BCELoss()

        if start_iter > 0:
            self.restore_model(start_iter)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iter, self.num_iters):
            
            x_real, label = self.getBatch()

            label = label.clone()
            label_onehot = self.onehot(label)

            #print(c_org)

            # input images
            x_real = x_real.to(self.device)
            
            if zdist is None:
                zdist = self.get_zdist("uniform", (3,x_real.size(2),x_real.size(3)), device=self.device)

            # make noise
            noise = zdist.sample((x_real.size(0),))

            

            if (i) % 1 == 0:
                # train discriminator
                toogle_grad(self.G, False)
                toogle_grad(self.D, True)

                #print(x_real.shape, label_org.shape)
                #print(x_real.shape)
                self.vis(x_real, label_onehot)

                # 隠したいカテゴリ
                hidden_categorys = [np.random.randint(self.n_classes) for _ in range(x_real.size()[0])]
                hidden_categorys = [self.class_names['car'] for _ in range(x_real.size()[0])]
                # onehotに変換
                hidden_categorys_onehot = np.eye(self.n_classes, dtype=np.float32)[hidden_categorys]           # one hot表現に変換
                hidden_categorys_onehot = torch.from_numpy(hidden_categorys_onehot).to(self.device)
                #print(hidden_categorys_onehot)

                # 教師データにそれぞれ何割のラベルが付与されているか
                label_per_real = self.label_contain_persent(label)
                
                #print(label_per_real) # shape [batch, 19, 1, 1]
                out_src, out_cls_real = self.D(x_real)
                #label_real = torch.full((x_real.size(0),1), 1.0, device=self.device)
                
                d_loss_real = -torch.mean(out_src)
                
                # クラス割合loss
                d_loss_cls_real = naive_cross_entropy_loss(out_cls_real, label_per_real) 

                #print(d_loss_cls_real) # shape 1
                
                x_mask = self.G(x_real, hidden_categorys_onehot)
                x_fake = x_mask * x_real + (1.0-x_mask) * noise
                out_src_fake, out_cls_fake = self.D(x_fake.detach())

                #label_fake = torch.full((x_real.size(0),1), 0.0, device=self.device)

                # クラス割合loss
                d_loss_cls_fake = naive_cross_entropy_loss(out_cls_fake, label_per_real) 
                d_loss_fake = torch.mean(out_src_fake)

                # gp_loss
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1.0 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                d_loss =d_loss_real + d_loss_fake + self.lambda_cls * (d_loss_cls_real+d_loss_cls_fake) + self.lambda_gp * d_loss_gp
                
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls_real'] = d_loss_cls_real.item()
                loss['D/loss_cls_fake'] = d_loss_cls_fake.item()
                loss['D/loss_gp'] = d_loss_gp.item()

            # train generator
            if (i+1) % self.n_critic == 0:
                toogle_grad(self.G, True)
                toogle_grad(self.D, False)
                x_mask = self.G(x_real, hidden_categorys_onehot)
                x_fake = x_mask * x_real + (1.0-x_mask) * noise
                out_src, out_cls = self.D(x_fake)

                label_real = torch.full((x_real.size(0),1), 1.0, device=self.device)
                
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.classification_loss(-out_cls+1.0, hidden_categorys_onehot) #naive_cross_entropy_loss(-out_cls+1.0, label_per_real)

                # backward
                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls
                
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # smoothing
                self.update_average(self.G_test, self.G, self.smooth_beta)


                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)


            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                x_fake_list = [x_real]
                x_fake_list.append(x_fake)
                #x_fake_list.append(x_reconst)
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                G_test_path = os.path.join(self.model_save_dir, '{}-G_test.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.G_test.state_dict(), G_test_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay lr
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self, test_iters=None):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        if test_iters is not None:
            self.restore_model(test_iters)

        #self.eval_model()
            
        # Set data loader.
        data_loader = self.loader
            
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = []
                for j in range(self.c_dim):
                    c_trg = c_org.clone()
                    c_trg[:,:] = 0.0
                    c_trg[:,j] = 1.0
                    c_trg_list.append(c_trg.to(self.device))
                
                # Translate images.
                x_fake_list = []
                
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G_test(x_real, c_trg))
                print(x_fake_list[0])

                # Save the translated images.
                try:
                    x_concat = torch.cat(x_fake_list, dim=3)
                    result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
                except:
                    import traceback
                    traceback.print_exc()
                    print('Error {}...'.format(result_path))
                

# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
