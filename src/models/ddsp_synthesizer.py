import torch
import torch.nn as nn
from ddsp.core import mlp, gru, scale_function, remove_above_nyquist, upsample
from ddsp.core import harmonic_synth, amp_to_impulse_response, fft_convolve
from ddsp.core import resample
import math
from conf.models import *

class DDSP(nn.Module):
	def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
				 block_size):
		super().__init__()
		self. sampling_rate = sampling_rate
		self.block_size = block_size

		self.mlp = mlp(hidden_size + 2, hidden_size, 3)

		self.proj_matrices = nn.ModuleList([
			nn.Linear(hidden_size, n_harmonic + 1),
			nn.Linear(hidden_size, n_bands),
		])

		#self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
		#self.register_buffer("phase", torch.zeros(1))

	def forward(self, z, pitch, loudness):
		hidden = torch.cat([z, pitch, loudness], -1)
		hidden = self.mlp(hidden)
		# harmonic part
		param = scale_function(self.proj_matrices[0](hidden))

		total_amp = param[..., :1]
		amplitudes = param[..., 1:]

		amplitudes = remove_above_nyquist(
			amplitudes,
			pitch,
			self.sampling_rate,
		)
		amplitudes /= amplitudes.sum(-1, keepdim=True)
		amplitudes *= total_amp

		amplitudes = upsample(amplitudes, self.block_size)
		pitch = upsample(pitch, self.block_size)

		harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

		# noise part
		param = scale_function(self.proj_matrices[1](hidden) - 5)

		impulse = amp_to_impulse_response(param, self.block_size)
		noise = torch.rand(
			impulse.shape[0],
			impulse.shape[1],
			self.block_size,
		).to(impulse) * 2 - 1

		noise = fft_convolve(noise, impulse).contiguous()
		noise = noise.reshape(noise.shape[0], -1, 1)

		signal = harmonic + noise

		return signal

class DDSPSynthesizer(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		model_name = "synthesizer"
		hp = MODEL_CONFIG[model_name]

		self.ddsp = DDSP(hidden_size=hidden_size,
																	n_harmonic=hp["n_harmonic"],
																	n_bands=hp["n_bands"],
																	sampling_rate=hp["sample_rate"],
																	block_size=hp["block_size"])

	def forward(self, input):
		z, pitch, loudness = [v.transpose(-1, -2) for v in input]
		wav = self.ddsp(z, pitch, loudness)
		return wav.squeeze(-1)

