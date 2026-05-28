#!/usr/bin/env python3
"""
Convert tiny_v4_optimized.pth  →  tiny_v4_optimized.tflite

No ONNX required. TinyUNetV4 is re-implemented in TF/Keras (channels-last),
weights are copied from the PyTorch checkpoint, then exported via TFLiteConverter.

TFLite model I/O  (float32):
  input:  (1, 400, 4)   [batch, time, channels]
  output: (1, 400, 2)
"""

import sys
import os
import numpy as np
import torch
import tensorflow as tf

sys.path.insert(0, '/home/bps-yichin/tommy')
from eeg_denoise.tiny_model_v4 import TinyUNetV4

BASE_DIR    = '/home/bps-yichin/lilia_analysis'
MODEL_PATH  = os.path.join(BASE_DIR, 'tiny_v4_optimized.pth')
TFLITE_PATH = os.path.join(BASE_DIR, 'tiny_v4_optimized.tflite')

N_CH      = 4
N_CH_OUT  = 2
MODEL_WIN = 400
BASE_CH   = 16


# ── Keras building blocks ──────────────────────────────────────────────────────

class DWConv1D(tf.keras.layers.Layer):
    """Depthwise Conv1D using grouped Conv1D with explicit symmetric padding.

    TF padding='same' with stride=2 is asymmetric (0 left, 1 right), but
    PyTorch uses symmetric padding=1. We pad manually and use padding='valid'.
    """
    def __init__(self, kernel_size=3, stride=1, **kw):
        super().__init__(**kw)
        self._ks  = kernel_size
        self._st  = stride
        self._pad = kernel_size // 2  # matches PyTorch padding=kernel_size//2
        self._dw  = None

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        self._dw = tf.keras.layers.Conv1D(
            in_ch, self._ks,
            strides=self._st,
            padding='valid',
            use_bias=False,
            groups=in_ch,
        )
        super().build(input_shape)

    def call(self, x):
        if self._pad > 0:
            x = tf.pad(x, [[0, 0], [self._pad, self._pad], [0, 0]])
        return self._dw(x)


class DSConv(tf.keras.layers.Layer):
    """DepthwiseSeparableConv1dV4 equivalent (channels-last)."""
    def __init__(self, out_ch, kernel_size=3, stride=1, **kw):
        super().__init__(**kw)
        self.dw    = DWConv1D(kernel_size=kernel_size, stride=stride)
        self.pw    = tf.keras.layers.Conv1D(out_ch, 1, use_bias=False)
        self.bn    = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1])  # one α per channel

    def call(self, x, training=False):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x, training=training)
        return self.prelu(x)


class UpBlock(tf.keras.layers.Layer):
    """TinyUpBlockV4 equivalent (channels-last)."""
    def __init__(self, out_ch, **kw):
        super().__init__(**kw)
        self.up       = tf.keras.layers.UpSampling1D(size=2)
        self.up_conv  = DSConv(out_ch, kernel_size=3)
        self.up_bn    = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.up_prelu = tf.keras.layers.PReLU(shared_axes=[1])
        self.conv     = DSConv(out_ch, kernel_size=3)
        self.fuse     = DSConv(out_ch, kernel_size=3)

    def call(self, x, skip, training=False):
        x = self.up(x)
        x = self.up_conv(x,  training=training)
        x = self.up_bn(x,    training=training)
        x = self.up_prelu(x)
        x = self.conv(x,     training=training)
        x = x + skip
        return self.fuse(x,  training=training)


class TinyUNetV4_TF(tf.keras.Model):
    def __init__(self, in_ch=N_CH, out_ch=N_CH_OUT, base=BASE_CH, **kw):
        super().__init__(**kw)
        c = base
        self.baseline   = tf.keras.layers.Conv1D(out_ch, 1)
        self.first_conv = DSConv(c,   kernel_size=3)
        self.down1_0    = DSConv(c*2, kernel_size=3, stride=2)
        self.down1_1    = DSConv(c*2, kernel_size=3)
        self.down2_0    = DSConv(c*4, kernel_size=3, stride=2)
        self.down2_1    = DSConv(c*4, kernel_size=3)
        self.bottleneck = DSConv(c*4, kernel_size=3)
        self.up2        = UpBlock(c*2)
        self.up1        = UpBlock(c)
        self.final      = tf.keras.layers.Conv1D(out_ch, 1)

    def call(self, x, training=False):
        base = self.baseline(x)
        x1   = self.first_conv(x,  training=training)
        tmp  = self.down1_0(x1,    training=training)
        x2   = self.down1_1(tmp,   training=training)
        tmp  = self.down2_0(x2,    training=training)
        x3   = self.down2_1(tmp,   training=training)
        bn   = self.bottleneck(x3, training=training)
        d2   = self.up2(bn, x2,    training=training)
        d1   = self.up1(d2, x1,    training=training)
        return base + self.final(d1)


# ── Weight copying ─────────────────────────────────────────────────────────────

def copy_weights(tf_model, pt_sd):
    def npy(key):
        return pt_sd[key].numpy()

    # Conv1D: PT (out, in, k) → TF (k, in, out)
    def set_conv1d(layer, prefix):
        ws = [np.transpose(npy(f'{prefix}.weight'), (2, 1, 0))]
        if f'{prefix}.bias' in pt_sd:
            ws.append(npy(f'{prefix}.bias'))
        layer.set_weights(ws)

    # DWConv: PT (in_ch, 1, k) → TF grouped Conv1D (k, 1, in_ch)
    def set_dw(dw1d_layer, prefix):
        w = npy(f'{prefix}.weight')            # (in_ch, 1, k)
        w = np.transpose(w, (2, 1, 0))         # (k, 1, in_ch)
        dw1d_layer._dw.set_weights([w])

    # Pointwise Conv1D (k=1): PT (out, in, 1) → TF (1, in, out)
    def set_pw(layer, prefix):
        layer.set_weights([np.transpose(npy(f'{prefix}.weight'), (2, 1, 0))])

    # BatchNorm: PT (weight, bias, running_mean, running_var) → TF (gamma, beta, mean, var)
    def set_bn(layer, prefix):
        layer.set_weights([
            npy(f'{prefix}.weight'),
            npy(f'{prefix}.bias'),
            npy(f'{prefix}.running_mean'),
            npy(f'{prefix}.running_var'),
        ])

    # PReLU: PT (out_ch,) → TF (1, out_ch)  [shared_axes=[1]]
    def set_prelu(layer, prefix):
        layer.set_weights([npy(f'{prefix}.weight').reshape(1, -1)])

    def set_dsconv(ds, prefix):
        set_dw(ds.dw,      f'{prefix}.depthwise')
        set_pw(ds.pw,      f'{prefix}.pointwise')
        set_bn(ds.bn,      f'{prefix}.bn')
        set_prelu(ds.prelu, f'{prefix}.act')

    def set_upblock(up, prefix):
        set_dsconv(up.up_conv,  f'{prefix}.up_conv')
        set_bn(up.up_bn,        f'{prefix}.up_bn')
        set_prelu(up.up_prelu,  f'{prefix}.up_act')
        set_dsconv(up.conv,     f'{prefix}.conv')
        set_dsconv(up.fuse,     f'{prefix}.fuse')

    set_conv1d(tf_model.baseline,   'baseline')
    set_dsconv(tf_model.first_conv, 'first_conv')
    set_dsconv(tf_model.down1_0,    'down1.0')
    set_dsconv(tf_model.down1_1,    'down1.1')
    set_dsconv(tf_model.down2_0,    'down2.0')
    set_dsconv(tf_model.down2_1,    'down2.1')
    set_dsconv(tf_model.bottleneck, 'bottleneck')
    set_upblock(tf_model.up2,       'up2')
    set_upblock(tf_model.up1,       'up1')
    set_conv1d(tf_model.final,      'final')
    print('Weight copy complete.')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load PyTorch model
    pt_model = TinyUNetV4(in_channels=N_CH, out_channels=N_CH_OUT)
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    pt_model.load_state_dict(ckpt['state_dict'])
    pt_model.eval()
    print(f'Loaded PyTorch model: {MODEL_PATH}')

    # Build TF model and copy weights
    tf_model = TinyUNetV4_TF()
    tf_model(tf.zeros((1, MODEL_WIN, N_CH), tf.float32), training=False)  # build
    print(f'TF model params: {tf_model.count_params():,}')
    copy_weights(tf_model, pt_model.state_dict())

    # Validate numerics: run same random input through both models
    np.random.seed(0)
    x_nchw = np.random.randn(1, N_CH, MODEL_WIN).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(x_nchw)).numpy()   # (1, 2, 400)
    x_ntc  = np.transpose(x_nchw, (0, 2, 1))                  # (1, 400, 4)
    tf_out = tf_model(x_ntc, training=False).numpy()           # (1, 400, 2)
    tf_out_t = np.transpose(tf_out, (0, 2, 1))                 # (1, 2, 400)
    err = np.abs(pt_out - tf_out_t).max()
    print(f'PT vs TF max absolute error: {err:.3e}')
    if err > 5e-2:
        print('WARNING: large error — check weight mapping!')
        return
    print('Validation OK (residual is GPU/CPU fp32 rounding).')

    # Export TFLite (float32)
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, MODEL_WIN, N_CH), dtype=tf.float32)])
    def infer(x):
        return tf_model(x, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [infer.get_concrete_function()])
    tflite_bytes = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_bytes)
    print(f'Saved: {TFLITE_PATH}  ({len(tflite_bytes) / 1024:.1f} KB)')

    # Smoke-test: run TFLite interpreter and compare to PT
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    interp.set_tensor(inp_d['index'], x_ntc)
    interp.invoke()
    tfl_out = np.transpose(interp.get_tensor(out_d['index']), (0, 2, 1))
    tfl_err = np.abs(pt_out - tfl_out).max()
    print(f'PT vs TFLite max absolute error: {tfl_err:.3e}')
    print('Done.')


if __name__ == '__main__':
    main()
