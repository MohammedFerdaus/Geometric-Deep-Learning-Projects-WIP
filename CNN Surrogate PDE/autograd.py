import numpy as np
 
class Tensor:
    def __init__(self, data, _prev=(), requires_grad=True):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_prev)
        self._backward = lambda: None
        self.requires_grad = requires_grad
 
    def backward(self):
        topo = []
        visited = set()
 
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)
 
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
 
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
 
def _sum_to_shape(grad, target_shape):
    ndim_diff = grad.ndim - len(target_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)
 
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
 
    return grad

def add(a, b):
    out = Tensor(a.data + b.data, _prev=(a, b))
 
    def _backward():
        a.grad += _sum_to_shape(out.grad, a.data.shape)
        b.grad += _sum_to_shape(out.grad, b.data.shape)
 
    out._backward = _backward
    return out
 
def mul(a, b):
    out = Tensor(a.data * b.data, _prev=(a, b))
 
    def _backward():
        a.grad += _sum_to_shape(out.grad * b.data, a.data.shape)
        b.grad += _sum_to_shape(out.grad * a.data, b.data.shape)
 
    out._backward = _backward
    return out

def matmul(a, b):
    out = Tensor(a.data @ b.data, _prev=(a, b))
 
    def _backward():
        a.grad += out.grad @ b.data.T
        b.grad += a.data.T @ out.grad
 
    out._backward = _backward
    return out
 
 
def relu(a):
    out = Tensor(np.maximum(0, a.data), _prev=(a,))
 
    def _backward():
        a.grad += out.grad * (a.data > 0)
 
    out._backward = _backward
    return out
 
 
def reshape(a, new_shape):
    out = Tensor(a.data.reshape(new_shape), _prev=(a,))
 
    def _backward():
        a.grad += out.grad.reshape(a.data.shape)
 
    out._backward = _backward
    return out
 
 
def sum_all(a):
    out = Tensor(np.array(a.data.sum()), _prev=(a,))
 
    def _backward():
        a.grad += out.grad * np.ones_like(a.data)
 
    out._backward = _backward
    return out
 
 
def mse_loss(pred, target):
    target = np.asarray(target, dtype=float)
    diff = pred.data - target
    out = Tensor(np.mean(diff ** 2), _prev=(pred,))
 
    def _backward():
        pred.grad += out.grad * 2 * diff / diff.size
 
    out._backward = _backward
    return out
 
 
def im2col(x, kernel_h, kernel_w, stride, pad):
    N, C, H, W = x.shape
 
    x_padded = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    H_padded = H + 2 * pad
    W_padded = W + 2 * pad
 
    out_h = ((H_padded - kernel_h) // stride) + 1
    out_w = (W_padded - kernel_w) // stride + 1
 
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=x.dtype)
    for i in range(kernel_h):
        for j in range(kernel_w):
            row_start = i
            row_stop = i + stride * out_h
            col_start = j
            col_stop = j + stride * out_w
            col[:, :, i, j, :, :] = x_padded[:, :, row_start:row_stop:stride, col_start:col_stop:stride]
 
    col_transposed = col.transpose(0, 4, 5, 1, 2, 3)
    col_2d = col_transposed.reshape(N * out_h * out_w, C * kernel_h * kernel_w)
 
    return col_2d, out_h, out_w
 
 
def col2im(col_2d, x_shape, kernel_h, kernel_w, stride, pad, out_h, out_w):
    N, C, H, W = x_shape
 
    col_transposed = col_2d.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    col = col_transposed.transpose(0, 3, 4, 5, 1, 2)
 
    H_padded = H + 2 * pad
    W_padded = W + 2 * pad
    padded_grad = np.zeros((N, C, H_padded, W_padded), dtype=col_2d.dtype)
 
    for i in range(kernel_h):
        for j in range(kernel_w):
            row_start = i
            row_stop = i + stride * out_h
            col_start = j
            col_stop = j + stride * out_w
 
            padded_grad[:, :, row_start:row_stop:stride, col_start:col_stop:stride] += col[:, :, i, j, :, :]
 
    if pad > 0:
        result = padded_grad[:, :, pad:pad + H, pad:pad + W]
    else:
        result = padded_grad
 
    return result
 
 
def conv2d(x, weight, bias, stride=1, pad=0):
    N, C_in, H, W = x.data.shape
    C_out, C_in_w, kh, kw = weight.data.shape
    (C_out_b,) = bias.data.shape
 
    col_2d, out_h, out_w = im2col(x.data, kh, kw, stride, pad)
    weight_mat = weight.data.reshape(C_out, -1)
 
    out_mat = col_2d @ weight_mat.T
    out_mat = out_mat + bias.data
 
    out_data = out_mat.reshape(N, out_h, out_w, C_out)
    out_data = out_data.transpose(0, 3, 1, 2)
 
    out = Tensor(out_data, _prev=(x, weight, bias))
    out.cache = {
        'col_2d': col_2d,
        'weight_mat': weight_mat,
        'out_h': out_h,
        'out_w': out_w,
        'kh': kh,
        'kw': kw,
        'stride': stride,
        'pad': pad}
 
    def _backward():
        col_2d = out.cache['col_2d']
        weight_mat = out.cache['weight_mat']
        out_h = out.cache['out_h']
        out_w = out.cache['out_w']
        kh = out.cache['kh']
        kw = out.cache['kw']
        stride = out.cache['stride']
        pad = out.cache['pad']
 
        grad_out = out.grad.transpose(0, 2, 3, 1)
        grad_out = grad_out.reshape(N * out_h * out_w, C_out)
 
        bias.grad += grad_out.sum(axis=0)
 
        weight_grad_mat = grad_out.T @ col_2d
        weight.grad += weight_grad_mat.reshape(C_out, C_in, kh, kw)
 
        grad_col = grad_out @ weight_mat
 
        x.grad += col2im(grad_col, x.data.shape, kh, kw, stride, pad, out_h, out_w)
 
    out._backward = _backward
    return out
 
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
 
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
 
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()