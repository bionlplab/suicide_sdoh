import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import argparse

def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='BioBERT', type=str, help="Pre-trained BERT model.")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Batch size train.")
    parser.add_argument("--learning_rate", default=2e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs", )
    parser.add_argument("--gpu_device", type=int, default=5, help="gpu device")
    parser.add_argument("--seed", type=int, default=2020, help="random seed for initialization")
    parser.add_argument("--output_dir", type=str, default='output_3class.txt', help="Output file dir.")

    return parser
