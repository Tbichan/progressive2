
# coding: utf-8

# In[47]:


import numpy as np
import os, sys
import glob
import cv2
import custom.config

# In[37]:


# ファイル全取得
def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


# In[70]:


class MultiLoader:
    def __init__(self, passA, passB):
        
        # A
        fileListA = find_all_files(passA)
        fileListA = list(fileListA)
        self.fileListA = []
        for file in fileListA:
            if file.find('.jpg') != -1 or file.find('.png') != -1:
                self.fileListA.append(file)

        self.permA = np.random.permutation(len(self.fileListA))
        self.indexA = 0
        
        # B
        fileListB = find_all_files(passB)
        fileListB = list(fileListB)
        self.fileListB = []
        for file in fileListB:
            if file.find('.jpg') != -1 or file.find('.png') != -1:
                self.fileListB.append(file)
        
        self.permB = np.random.permutation(len(self.fileListB))
        self.indexB = 0
        
        self.width = custom.config.train_params['width']
        self.height = custom.config.train_params['height']
        self.num = len(self.fileListA)+len(self.fileListB)
        
    def getBatch(self, batch_size=16, width=None, height=None, alpha=1.0, mirror=True):
        rands = np.random.uniform(0.0, 1.0, batch_size)
        res = []

        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        for rand in rands:
            #print(rand)
            if rand <= alpha:
                # A
                img = self.load_image(self.fileListA[self.indexA], width=width, height=height, rev=(np.random.randint(2) == 1) and mirror)
                res.append(img)
                #res.append(self.fileListA[self.indexA])
                self.indexA+=1
                if self.indexA >= len(self.fileListA):
                    self.permA = np.random.permutation(len(self.fileListA))
                    self.indexA = 0
                    print("train A suffle")
            else:
                # B
                img = self.load_image(self.fileListB[self.indexB], width=width, height=height, rev=(np.random.randint(2) == 1) and mirror)
                res.append(img)
                #res.append(self.fileListB[self.indexB])
                self.indexB+=1
                if self.indexB >= len(self.fileListB):
                    self.permB = np.random.permutation(len(self.fileListB))
                    self.indexB = 0
                    print("train B suffle")
        
        return res
    
    def load_image(self, fpass, width=None, height=None, rev=False):
        img = cv2.imread(fpass)

        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        # INPUT_WxINPUT_Hにリサイズ
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

        if rev:
            img = cv2.flip(img, 1)
        
        # 1列にし-1～1のfloatに
        img = 2.0 * img / 255.0 - 1.0
        
        img = np.transpose(img, (2, 1, 0)).astype(np.float32)
        
        return img


# In[71]:


#m = MultiLoader(passA='./trainA',passB='./trainB')


# In[72]:


#m.getBatch(batch_size=16, alpha=0.8)


# In[63]:




