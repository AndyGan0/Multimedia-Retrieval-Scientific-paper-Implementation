import numpy as np

from glob import glob

import torch
from torchvision import models, transforms
from PIL import Image
import random





def getDataList(DatasetPath: str = "Dataset/", L_size = 500):
    #   Read all Data from the folders and return a list with size L

    DatasetInnerFolders = glob(DatasetPath + "*")
    DatasetInnerFolders = [file.replace('\\', '/') for file in DatasetInnerFolders]    

    #   Adding all files into lists
    AllFiles = []
    Labels = []
    currentLabel = 0

    for folder in DatasetInnerFolders:
        CurrentFolderFiles = glob(folder + "/" + "*")
        CurrentFolderFiles = [file.replace('\\', '/') for file in CurrentFolderFiles]
        AllFiles.extend(CurrentFolderFiles)

        Labels.extend( [currentLabel] * len(CurrentFolderFiles))
        currentLabel += 1

    if ( L_size > len(AllFiles) ):
        L_size = len(AllFiles)


    #   Reducing Dataset Size to L in order to reduce computing power
    L_indices = random.sample(range(len(AllFiles)), L_size)
    L_files = [AllFiles[i] for i in L_indices]
    L_labels = [Labels[i] for i in L_indices]

    return np.array(L_files), np.array(L_labels)






#   Getting the Image Descriptor
ImageDescriptor = models.resnet50(pretrained=True)
#   Removing the last layer (classification layer)
ImageDescriptor = torch.nn.Sequential(*(list(ImageDescriptor.children())[:-1]))
ImageDescriptor.eval()
#print(ImageDescriptor)


def ExtractAllFeatureVectors(ImagePaths):
    #   Extract Feature Vectors from a list of all images

    FeatureVectors = []
    for image in ImagePaths:
        FeatureVectors.append( ExtractFeatureVectorFromSingleImage(image) )            
    FeatureVectors = np.array(FeatureVectors)

    return FeatureVectors



def ExtractFeatureVectorFromSingleImage(ImagePath: str):
    #   Exctract Feature Vector from a single image 

    #   Loading
    image = Image.open(ImagePath).convert('RGB')

    #   Pre processing
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        Feature_Vector = ImageDescriptor(image)
    
    Feature_Vector = Feature_Vector.flatten()

    return Feature_Vector






def computeSimilarityMeasure(FeatureVectors):
    #   returns the euclidean distance, the similarity measure

    #   Calculating eunclidean distance
    Euclidean_Distance = calculateEuclideanDistance(FeatureVectors)  

    #   Calcularting similarity measure based on euclidean distance
    Similarity_Measure = 1 / (Euclidean_Distance + 1)    
        
    return Euclidean_Distance, Similarity_Measure




def calculateEuclideanDistance(Feature_Vectors):
    #   Calculate teh euclidean Distance between each image and all others    

    # Initialize EuclideanDistance with 0
    Euclidean_Distance = np.zeros((len(Feature_Vectors), len(Feature_Vectors)))

    # Calculate the distances 
    for i in range(len(Feature_Vectors)):
        for j in range(i + 1):
            distance = np.linalg.norm(Feature_Vectors[i] - Feature_Vectors[j])

            #   Euclidean distance is symmetrical regardless of the order
            Euclidean_Distance[i, j] = distance
            Euclidean_Distance[j, i] = distance

          
    return Euclidean_Distance
    




def compute_T_Lists(Similarity_Measure):
    #   Calculating T permuatation lists from similarity measure

    T_Lists = np.argsort(Similarity_Measure, axis=1)
    T_Lists = np.fliplr(T_Lists)

    return T_Lists
