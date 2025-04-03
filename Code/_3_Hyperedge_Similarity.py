import numpy as np

def compute_Hyperedge_Similarity(H_matrix : np.ndarray):
    #   Computes S_h, S_v, S and returns S

    H_matrix_T = H_matrix.T

    S_h = np.dot( H_matrix , H_matrix_T ) 

    S_v = np.dot( H_matrix_T , H_matrix ) 

    S = S_h * S_v

    return S