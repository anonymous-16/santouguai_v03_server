from .feature import *
from .urmp import *
from .sample import MAX_SOURCES_NUM

MODEL_CONFIG = {
	"santouguai":
		{"f0_res" : F0_RES,
			"f0s_num" : F0S_NUM, 
			"notes_num" : NOTES_NUM - 1,
			"ctx_num" : 2**ONSET_RES,
			"blocks" : [4, 8, 16, 32, 64],
			"note_timbre_dim" : 256,
			"ab_timbre_dim" : 16,
			"max_clusters_num" : 1,
			"timbre_dim" : 1024,
			"feature_size" : WINDOW_SIZE // 2
		},
}

