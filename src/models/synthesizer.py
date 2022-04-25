import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

from .repUnet1D import RepUnet1D
from .layers import (LinearBlock2D, ConvBlock1D, DecoderBlock)

from conf.synthesizer import *


class Synthesizer(nn.Module):
	def __init__(self):
		super(Synthesizer, self).__init__()
		syn = nn.ModuleList()
		self.syn = RepUnet1D(ENC_CHANNELS, MID_CHANNEL, DEC_CHANNELS, BOTTOM_CHANNEL)

	def forward(self, ti, note, f0):
		x = torch.cat([ti, note, f0], 1)
		curve = F.softmax(self.syn(x), 1)
		return curve

