"""
Helper functions
"""

import pickle
import os.path
import torch
from torchvision import transforms

def pickle_load(filename):
    obj = None
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_ids_txt(filename):
    """
    Load ids from txt
    Input:
        :param filename: the filename to load ids
    Return:
        :return ids: (list of int) ids
    """
    content = ""
    with open(filename, "r") as f:
        content = f.read()
    ids = [int(x) for x in content.split()]

    return ids

def save_img_tensor(img_batch, folder):
    """

    :param img_batch: B*C*W*H
    :param folder:
    :return:
    """
    p = transforms.ToPILImage()

    B, C, W, H = img_batch.shape

    for batch_idx in range(B):
        img_tensor = img_batch[batch_idx, :, :, :]
        for img_idx in range(C):
            img = p(img_tensor[img_idx, :, :])
            img.save(os.path.join(folder, str(batch_idx) + "_" + str(img_idx) + ".png"), "png")


def get_device():
    """
    Return the torch device (device("cpu") when no gpu available, and device("gpu") when it is available)
    Note: Only one gpu card is allowed
    :return:
    """
    device = torch.device("cpu")
    if (torch.cuda.device_count() > 0):
        # make sure this process only access one GPU
        print("[INFO] using gpu device.")
        device = torch.device("gpu")
        if (torch.cuda.device_count() > 1):
            raise RuntimeError("Only one gpu device expected. Maybe you forgot to run with CUDA_VISABLE_DEVICES")
    else:
        print("[INFO] using cpu device.")
    return device


def setup_logger_dir(opts, dump_config = True):
    """
    Setting up logger, names are given by experiment index
    Return: the directory for logging
    """
    existing_logs = [os.path.isdir(x) for x in os.listdir(opts.logging_root_dir)]
    # i = 0
    # while True:
    #     log_dir = os.path.join(opts.logging_dir, "exp" + str(i))
    #     if not os.path.isdir(log_dir):
    #         return SummaryWriter(logdir=log_dir)
    #     i += 1
    log_filename = "exp" + str(len(existing_logs)) + "_batch_" + str(opts.train_batch_size) +\
                    "_lr_" + str(opts.init_learning_rate)
    log_dir = os.path.join(opts.logging_root_dir, log_filename)

    return log_dir

def makedirs(dir, exist_ok=False):
    """
    Make one directory
    """
    res = os.makedirs(dir, exist_ok=exist_ok)
    print("[INFO] Created directory {}".format(dir))
    return res
