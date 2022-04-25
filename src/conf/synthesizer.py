from .timbreEncoder import DEC_CHANNELS as ti_dims


ti_dim = sum(ti_dims) * 2

ENC_CHANNELS = [ti_dim + 89 + 1,512,256,128,64,32]
MID_CHANNEL = [32,16]
DEC_CHANNELS = [16,32,64,128,256,512]
BOTTOM_CHANNEL = [512,1024]
