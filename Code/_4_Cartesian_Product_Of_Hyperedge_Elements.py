import numpy as np




def compute_cartesian_product_similarity(Edges : np.ndarray, 
                                         weights_of_edges : np.ndarray,
                                         H_matrix : np.ndarray ):


    L_size = Edges.shape[0]
    K_size = Edges.shape[1]

    c_similarity = np.zeros((L_size, L_size))

    for q in range(L_size):
        e_q = Edges[q]

        for i in range(K_size):
            v_i = e_q[i]

            for j in range(K_size):
                v_j = e_q[j]

                #   v_i and v_j belong to (e_q)^2
                #   r_similarity_e_q_and_v_i_and_v_j is ρ( e_q , v_i , v_j )

                r_similarity_e_q_and_v_i_and_v_j = weights_of_edges[q] + H_matrix[q, v_i] + H_matrix[q, v_j]

                c_similarity[v_i, v_j] += r_similarity_e_q_and_v_i_and_v_j



    return c_similarity

