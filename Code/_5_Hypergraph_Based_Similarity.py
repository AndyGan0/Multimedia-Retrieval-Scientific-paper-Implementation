import numpy as np



def compute_Hypergraph_based_similarity( S_similairty : np.ndarray,  C_similarity: np.ndarray):    

    W_similarity = C_similarity * S_similairty

    return W_similarity
