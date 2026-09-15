import numpy as np
from autograd import Tensor, conv2d, relu

def he_init(shape, fan_in, rng):
    std = np.sqrt(2.0 / fan_in)
    return rng.standard_normal(shape) * std


def compute_required_receptive_field(n_steps, kernel_size=3):
    radius_per_layer = (kernel_size - 1) / 2
    n_layers_needed = int(np.ceil(n_steps / radius_per_layer))
    return n_layers_needed

class CNNSurrogate:
    def __init__(self, layer_config, rng):
        self.layers = []
        for (C_in, C_out, k, stride, pad, activate) in layer_config:
                weight = Tensor(he_init((C_out, C_in, k, k), C_in*k*k, rng), requires_grad=True)
                bias = Tensor(np.zeros(C_out), requires_grad=True)
                self.layers.append({'weight': weight,
                                    'bias': bias,
                                    'stride': stride,
                                    'pad': pad,
                                    'activate': activate})

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = conv2d(h, layer['weight'], layer['bias'], layer['stride'], layer['pad'])
            if layer['activate']: h = relu(h)
        return h

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer['weight'])
            params.append(layer['bias'])
        return params