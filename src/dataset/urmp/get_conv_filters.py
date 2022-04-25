import numpy as np

from conf.feature import *

def get_filters():
	data = np.ones([NOTE_RES]) * 4
	data[200:400] = 2
	data[600:800] = 2
	

	res = np.zeros([NOTE_RES, NOTE_RES, NOTE_RES])
	
	pot = NOTE_RES // 2
	scale = 1
	for i in range(0, NOTE_RES - 1):
		win_len = NOTE_RES - i - 1
		if win_len * 2 > NOTE_RES - 1:
			st = 0
			dt = 2**scale
			pad_len = (win_len - pot) * (2**scale)
			dt = 2**(scale - 1)
			for j in range(win_len):
				if st == 0 and j * dt >= pad_len:
					st = j
					dt = 2**scale
				if st > 0:
					res[win_len - 1, j, pad_len + (j - st) * dt : pad_len + (j - st + 1) * dt] = 1. / dt
				else:
					res[win_len - 1, j, j * dt : (j + 1) * dt] = 1. / dt
		else:
			for j in range(win_len):
				res[win_len - 1, j] = (res[win_len * 2 - 1, 2 * j] +  res[win_len * 2 - 1, 2 * j + 1]) * 0.5

#	test_len = 3
#	for test_len in range(3, 20):
#		tmp = []
#		for i in range(test_len):
#			tmp.append((res[test_len - 1, i] * data).sum())
#		print(test_len, tmp)
	return res


if __name__=="__main__":
	res = get_filters()
	#for i, k in enumerate(filters):
	#	print(i, k)
