import numpy as np
import math



def construct_hypergraph( T_Lists : np.ndarray, 
                          T_inverse_permutation : np.ndarray, 
                          Neighbors_K : np.ndarray ):
    #   Returns v, E, w and H_matrix
    #   V is the set of vertices
    #   E is the set of Edges (and the vertices that each edge connects)
    #   w are the weights of each Edge


    L_size = T_Lists.shape[0]
    K_size = Neighbors_K.shape[1]



    #   Each image is represented by a vertex
    #   Image O_1 is represented by 1 in V
    Vertices = {i for i in range(L_size)}



    #   One edge is defined for each vertex
    #   Each edge e_i contains the K neighbors of the vertex v_i
    #   Edges can be defined by the same array as Neighbor_K 
    #   Each edge is defined by a row, and the items contained in it are the images that it connects
    Edges = Neighbors_K


    #
    #   Lastly we need to define the weights of each edge
    #


    #   We need w (weights) for image O_x for T_i
    #   w( i , x )
    Weights = np.zeros_like(T_Lists, dtype=float)

    for e_i in range(L_size):
        for x in range(L_size):
            position_of_image_x_in_T_i = T_inverse_permutation[e_i, x]
            
            #   if position is 0 then we cant calculate the log.
            #   instead we will skip those values and leave them 0
            if (position_of_image_x_in_T_i != 0):
                Weights[e_i, x] = 1 - math.log( position_of_image_x_in_T_i , K_size + 1)



    #   then compute the degree to which V_j belongs to e_i
    #   r( e_i , v_j )
    #   this will be represented by a matrix where the row is i and column is j
    r_degree = np.zeros_like(T_Lists, dtype=float)

    for e_i in range (L_size):
        for v_j in range(L_size):

            for x in range(1, K_size):                
                O_x = Neighbors_K[e_i, x]

                #   if O_j isn't a neighbor of O_x then it doesnt affect the score
                if ( T_inverse_permutation[O_x, v_j] < K_size ):
                    #   if position of O_j in T_List[O_x] is smaller than k  then
                    #   O_j is neighbor of O_x
                    #   it affects the score

                    r_degree[e_i, v_j] += Weights[e_i, O_x] * Weights[O_x, v_j]



    #   Define H_matrix matrix
    H_matrix = np.zeros_like(T_Lists, dtype=float)

    for e_i in range(L_size):

        #   for each edge e_i only the vertices connected to it will have a score in H_matrix
        #   H_matrix is already initialized with zeroes, so we update only the positions for the vertices that belong to the edge
        #   the vertices connected to e_i are the k neighbors

        for j in range(K_size):
            v_j = Neighbors_K[e_i, j]

            H_matrix[e_i, v_j] = r_degree[e_i, v_j]
    



    #   finally defining the weights for each edge
    weights_for_edges = np.zeros(L_size, dtype=float)

    for e_i in range(L_size):

        for j in range(K_size):
            v_j = Neighbors_K[e_i, j]
            weights_for_edges[e_i] += H_matrix[e_i, v_j]
        





    return Vertices, Edges, weights_for_edges, H_matrix