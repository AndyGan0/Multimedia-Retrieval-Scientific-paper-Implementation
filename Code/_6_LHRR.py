from _0_Feature_Extraction import getDataList, ExtractAllFeatureVectors, computeSimilarityMeasure, compute_T_Lists
from _1_Rank_Normalization import calculate_T_List_inverse_permutation, Rank_Normalization
from _2_Hypergraph_Construction import construct_hypergraph
from _3_Hyperedge_Similarity import compute_Hyperedge_Similarity
from _4_Cartesian_Product_Of_Hyperedge_Elements import compute_cartesian_product_similarity
from _5_Hypergraph_Based_Similarity import compute_Hypergraph_based_similarity

import numpy as np
import matplotlib.pyplot as plt
import cv2
import random







def Log_Based_Hypergraph_Of_Ranking_References( T_Lists : np.ndarray, W_similarity : np.ndarray):    

    #   calculating the inverse permutation (it will help later. It is explained in the documentation)
    T_inverse_permutation = calculate_T_List_inverse_permutation(T_Lists)

    #   Rank Normalization
    Similarity_Measure, T_Lists, T_inverse_permutation = Rank_Normalization(T_Lists, T_inverse_permutation)

    #   Calculating k neightbors
    Neighbors_K = np.array([row[:K_size] for row in T_Lists])

    #   Hypergraph Construction
    Vertices, Edges, Weights_for_Edges, H_matrix = construct_hypergraph(T_Lists, T_inverse_permutation, Neighbors_K)

    #   Hyperedge Similarity
    S_matrix_similarity = compute_Hyperedge_Similarity(H_matrix)

    C_matrix_similarity = compute_cartesian_product_similarity(Edges, Weights_for_Edges, H_matrix)

    New_W_similarity = compute_Hypergraph_based_similarity(S_matrix_similarity, C_matrix_similarity)

    new_T_Lists = compute_T_Lists(New_W_similarity)

    return New_W_similarity, new_T_Lists







def plot_random_targets(T_Lists, files_list, Accuracy_Scores, K_size):

    target_List = random.sample(range(len(T_Lists)), 5)

    for target in target_List:

        current_target_neighbors = T_Lists[target, :K_size]

        image_paths = files_list[current_target_neighbors]

        images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in image_paths]

        rows, cols = 1, 10
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i])
            ax.set_title(image_paths[i], fontsize=10)
            ax.axis('off')
        
        title = 'Ranked List for' + files_list[target] + "\nAccuracy: " + str(Accuracy_Scores[target])
        plt.suptitle( title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    return








def save_Ranked_Lists_T(T_Lists : np.ndarray, files_list :np.ndarray, K_size, Accuracy_Scores : np.ndarray):


    L_size = T_Lists.shape[0]

    with open('RankedLists.txt', 'w') as file:

        file.write('Average Accuracy: ')
        averageScore= np.mean(Accuracy_Scores)
        file.write( str( averageScore ) )
        file.write('\n\n\n')



        for i in range(L_size):
            file.write('Ranked List of Image ')
            file.write(files_list[i])
            file.write(' :\n')
            file.write('Accuracy: ' + str(Accuracy_Scores[i]) + '\n')

            for j in range(K_size):
                image_index = T_Lists[i,j]
                
                file.write('    ')
                file.write(files_list[image_index])
                file.write('\n')
            
            
            file.write('\n\n')

    return







def calculate_Accuracy_of_T_list(T_Lists : np.ndarray, Labels : np.ndarray, K_size):
    #   returns a list with the accuracy of each T_List

    L_size = T_Lists.shape[0]
    Scores = []

    for i in range(L_size):
        Correct_Category_Count = 0

        for j in range(K_size):
            image_index = T_Lists[i,j]
            if ( Labels[image_index] == Labels[i] ):
                Correct_Category_Count += 1
        
        Accuracy = Correct_Category_Count / K_size  

        Scores.append( Accuracy )

    return np.array(Scores)








print("Extracting Dataset Feature Vectors...\n"  )

K_size = 10
L_size = 1000

#   Getting file list
files_list, Labels = getDataList(L_size=L_size)
if ( L_size > len(files_list)):
    L_size = len(files_list)

#   Extracting feature vectors
Feature_Vectors = ExtractAllFeatureVectors(files_list)

#   Computing Euclidean Distance, Similarity Measure and T_Lists
Euclidean_Distance, Similarity_Measure = computeSimilarityMeasure(Feature_Vectors)

Old_T_Lists = compute_T_Lists(Similarity_Measure)
Old_similarity = Similarity_Measure

print("Log Based Hypergraph Of Ranking References Started..."  )
print("-----------------------------------------------------"  )

for itteration in range(5):

    New_W_similarity, New_T_Lists = Log_Based_Hypergraph_Of_Ranking_References( Old_T_Lists, Old_similarity)

    
    print("     Itteration ", itteration + 1, "...\n"  )
    itteration += 1
    Old_T_Lists = New_T_Lists
    Old_similarity = New_W_similarity


print("-----------------------------------------------------\n"  )



final_T_List = Old_T_Lists


Accuracy = calculate_Accuracy_of_T_list(final_T_List, Labels, K_size)
save_Ranked_Lists_T(final_T_List, files_list, K_size, Accuracy)
print("\nAccuracy: ", np.mean(Accuracy), "\n"  )

print("Results  Saved in RankedLists.txt"  )


plot_random_targets(final_T_List, files_list, Accuracy, K_size)
