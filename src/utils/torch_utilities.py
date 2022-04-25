import torch
import math
import librosa
import numpy as np
import torch.nn.functional as F
import torchaudio.functional as AF


from conf.feature import *

def device(x):
	return torch.from_numpy(x).float().cuda()

def predict(x, dt=0.5):
	x = torch.sigmoid(x)
	x[x > dt] = 1
	x[x < 1] = 0
	return x

def get_fft_window():
	fft_window = librosa.filters.get_window(WINDOW, WINDOW_SIZE, fftbins=True)
	fft_window = librosa.util.pad_center(fft_window, N_FFT)
	return torch.from_numpy(fft_window)

FFT_WINDOW = get_fft_window()

def spec_2_wav(x, cos, sin, wav_len, phase=False):
	if not x.shape[1] == cos.shape[1]:
		x = F.pad(x, (0, 1, 0, 1), "constant", 0)
	fft_window = FFT_WINDOW.to(x.device)
	if phase:
		spec = torch.stack([x * cos, x * sin], -1)
		wav = torch.istft(spec,
											n_fft=N_FFT,
											hop_length=HOP_SIZE,
											win_length=WINDOW_SIZE,
											window=fft_window,
											center=True,
											normalized=False,
											onesided=None,
											length=wav_len,
											return_complex=False)
	else:
		wav = AF.griffinlim(x,
												window=fft_window,
												n_fft=N_FFT,
												hop_length=HOP_SIZE,
												win_length=WINDOW_SIZE,
												power=1,
												length=wav_len,
												n_iter=50,
												momentum=0,
												rand_init=False)

	return wav

def wav_2_spec(wav, scale=None):
	fft_window = FFT_WINDOW.to(wav.device)
	spec = torch.stft(wav,
										N_FFT,
										hop_length=HOP_SIZE,
										win_length=WINDOW_SIZE,
										window=fft_window,
										center=True,
										pad_mode='reflect',
										normalized=False,
										onesided=None,
										return_complex=False)
	real = spec[:, :, :, 0]
	imag = spec[:, :, :, 1]
	mag = (real ** 2 + imag ** 2) ** 0.5
	cos = real / torch.clamp(mag, 1e-10, np.inf)
	sin = imag / torch.clamp(mag, 1e-10, np.inf)
	if scale is not None:
		mag = mag / scale[:, None, None]
	return mag[:, :-1, :-1].float(), cos, sin


def onehot_tensor(x, classes_num, last_mask=False):
	eye = torch.eye(classes_num)
	if last_mask:
		eye[classes_num - 1, :] = 1
	return eye.to(x.device)[x.long()]


def positional_encoding(d_model, length):
	if d_model % 2 != 0:
			raise ValueError("Cannot use sin/cos positional encoding with "
											 "odd dim (got dim={:d})".format(d_model))
	pe = torch.zeros(length, d_model)
	position = torch.arange(0, length).unsqueeze(1)
	div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
											 -(math.log(10000.0) / d_model)))
	pe[:, 0::2] = torch.sin(position.float() * div_term)
	pe[:, 1::2] = torch.cos(position.float() * div_term)

	return pe
