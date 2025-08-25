import sys
sys.path.append('./ssc/src')

from soundshapecode.ssc_similarity.compute_ssc_similarity import computeSSCSimilaruty


def compute_similarity_list(ssc1, ssc2, encode_way='ALL'):
    cnt = 0
    score = 0
    while cnt < len(ssc1) and cnt < len(ssc2):
        score += computeSSCSimilaruty(ssc1[cnt], ssc2[cnt], encode_way)
        cnt += 1
    return score / max(len(ssc1), len(ssc2))
