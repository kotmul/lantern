import torch

def DL_to_LD(DL):
    '''
    Dict[List] -> List[Dict]
    '''
    v = [dict(zip(DL,t)) for t in zip(*DL.values())]
    return v