import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.linalg import pinv  

def LoadYaleFaces(Start_from_person=0, Number_of_People=5):
    FacesMatrix = None
    base_path = r'C:\Users\Flick\Desktop\uni\rom\asignment2\part 3\YaleFaces'  # The base directory 
    for i in range(Start_from_person, Number_of_People):
        if i < 9:
            path = f'{base_path}\\yaleB0{i+1}\\'
        else:
            path = f'{base_path}\\yaleB{i+1}\\'
        
        for filename in os.listdir(path):
            image_in_numpy = plt.imread(os.path.join(path, filename))
            if FacesMatrix is None:
                FacesMatrix = image_in_numpy.reshape(-1, 1)
            else:
                FacesMatrix = np.c_[FacesMatrix, image_in_numpy.reshape(-1, 1)]
    return FacesMatrix

FacesMatrix = LoadYaleFaces(Number_of_People=10)  
print('\nFaces matrix size: ', FacesMatrix.shape)

def PrintAFace(face, title="Face"):
    plt.imshow(face.reshape(192, 168), cmap='gray')
    plt.title(title)
    plt.axis('off')  
    plt.show()

EigenFaces, s, v_transposed = np.linalg.svd(FacesMatrix, full_matrices=False)

def apply_mask(face, percentage_to_keep=0.20):  
    mask = np.random.rand(len(face)) < percentage_to_keep  
    return face * mask, mask 

def reconstruct_masked_face(U_truncated, masked_face, mask):
    U_masked = U_truncated * mask[:, np.newaxis]
    
    q_hat = pinv(U_masked) @ masked_face
    
    reconstructed_face = U_truncated @ q_hat
    return reconstructed_face

face_index = 80  
face_to_apply_mask_on = FacesMatrix[:, face_index]  
PrintAFace(face_to_apply_mask_on, title="Original Face")

masked_face, mask = apply_mask(face_to_apply_mask_on, percentage_to_keep=0.075)
PrintAFace(masked_face, title="Masked Face")

U_truncated = EigenFaces[:, :100]

reconstructed_face = reconstruct_masked_face(U_truncated, masked_face, mask)
PrintAFace(reconstructed_face, title="Reconstructed Face")
