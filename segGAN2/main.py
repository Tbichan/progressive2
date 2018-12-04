import os
import argparse
from solver import Solver
#from loader import get_loader
from torch.backends import cudnn
from ptsemseg.loader.cityscapes_loader import cityscapesLoader

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *

def main():

    # training fast
    cudnn.benchmark = True

    # Directories.
    log_dir = "log"
    sample_dir = "sample"
    model_save_dir = "model"
    result_dir = "result"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    augmentations = Compose([Scale(2048), RandomSizedCrop(256), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/path/to/datasets/cityscapes/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)

    solver = Solver(trainloader)
    solver.train()

if __name__ == "__main__":
    main()
