from conf.feature import *
from conf.sample import *
import numpy as np

def extract_context(hf, st):
	context_segment = []
	for i in range(st, st + FRAME_DURATION):
		context = np.zeros([CONTEXT_WIN]) - NOTES_NUM
		duration_ratio = np.zeros([CONTEXT_WIN])
		n = hf["context"][hf["context_mapping"][i], 0]
			
		cst = hf["context_mapping"][i] - CONTEXT_WIN // 2
		ced = hf["context_mapping"][i] + CONTEXT_WIN // 2
		cxt_st = 0 if cst < 0 else cst
		cxt_ed = hf["context"].shape[0] if ced > hf["context"].shape[0] else ced
		duration_ratio[cxt_st - cst : cxt_ed - cst] = hf["context"][cxt_st : cxt_ed, 1] /  hf["context"][hf["context_mapping"][i], 1]


		if n < NOTES_NUM - 1:
			context[cxt_st - cst : cxt_ed - cst] = hf["context"][cxt_st : cxt_ed, 0]
			mask = (context == NOTES_NUM)
			sil = (context == NOTES_NUM - 1)
			context = (context - n) / NOTES_NUM
			context[mask] = - 1
			context[sil] = 1
		else:
			context = context / NOTES_NUM
			context[cxt_st - cst : cxt_ed - cst] = 1
		context_segment.append(np.concatenate([context, duration_ratio], 0))
		#print(context)
		#print(duration_ratio)
		context = np.stack(context_segment, -1)
	return context
