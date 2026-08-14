import argparse
import time

import torch
from thop import profile

from amfbo.core.models.amfbo_model import AMFBOModel


def compute_gflops_and_model_size(model, input):
    macs, params = profile(model, inputs=(input,), verbose=False)
 
    GFlops = macs * 2.0 / pow(10, 9)
    model_size = params * 4.0 / 1024 / 1024
    params_M = params/pow(10, 6)
    return params_M, model_size, GFlops
 
@torch.no_grad()
def compute_fps(model, input, epoch=100, device=None):
    total_time = 0.0
 
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
 
    for i in range(epoch):
 
        start = time.time()
        outputs = model(input)
        end = time.time()
 
        total_time += (end - start)
 
    return total_time / epoch
 
 
def test_model_flops(args, input):
    model = AMFBOModel(args)     
    params_M, model_size, gflops = compute_gflops_and_model_size(model, input)
 
    print('Number of parameters: {:.2f} M '.format(params_M))
    print('Size of model: {:.2f} MB'.format(model_size))
    print('Computational complexity: {:.2f} GFlops'.format(gflops))
 

def test_fps(args, input):
    model = AMFBOModel(args)
    fps = compute_fps(model, input, device='cpu')
    print('device: {} - fps: {:.3f}s'.format('cpu', fps))
 
