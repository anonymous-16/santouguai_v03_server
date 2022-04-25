import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from src.dataset.musicNet.musicnet_test import MusicnetTest
from src.dataset.urmp.urmp_test import UrmpTest


def load_test_data(dataset, data_path):
	if dataset == "musicnet":
		return MusicnetTest(data_path)
	elif dataset == "urmp":
		return UrmpTest(data_path)
	assert False
