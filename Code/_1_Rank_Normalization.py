from _0_Feature_Extraction import compute_T_Lists

import numpy as np





def Rank_Normalization(T_Lists : np.ndarray, T_inverse_permutation : np.ndarray):
    #   Returns the normalized T_Lists
    #   Aims to make T_List symmetric

    L_size = T_Lists.shape[0]
    R_N_similarity_measure = np.empty_like(T_Lists)

    for i in range(L_size):
        #   for the first itteration i == j
        position_of_imagei_in_Tj = T_inverse_permutation[i, i]
        similarity = 2 * L_size - 2 * position_of_imagei_in_Tj
        R_N_similarity_measure[i, i] = similarity

        for j in range(i, L_size):

            position_of_imagej_in_Ti = T_inverse_permutation[i, j]
            position_of_imagei_in_Tj = T_inverse_permutation[j, i]

            similarity = 2 * L_size - position_of_imagej_in_Ti - position_of_imagei_in_Tj

            #   similarity is symmetrical regardless of the order
            R_N_similarity_measure[i, j] = similarity
            R_N_similarity_measure[j, i] = similarity


    T_Lists = compute_T_Lists(R_N_similarity_measure)
    T_inverse_permutation = calculate_T_List_inverse_permutation(T_Lists)

    return R_N_similarity_measure, T_Lists, T_inverse_permutation






def calculate_T_List_inverse_permutation(T_Lists):
    #   Get T_lists and produce T_inverse_permutation for each row
        
    T_inv_perm = np.empty_like(T_Lists)

    for i, row in enumerate(T_Lists):
        inverse_permutation_row = np.empty_like(row)
        inverse_permutation_row[row] = np.arange(len(row))

        T_inv_perm[i] = inverse_permutation_row
    
    
    return T_inv_perm