import nrrd
import numpy as np
import nibabel as nib


def nrrd2Nifti(nrrdFile : str , output: str) :
    data , header = nrrd.read(nrrdFile)
    space = header["space"]
    directions = header["space directions"]
    origin = np.absolute(header["space origin"]) 

    if "left" in space :
        dir_coefi = -1
    elif "right" in space : 
        dir_coefi = 1
    if "posterior" in space :
        dir_coefj = -1
    elif "anterior" in space : 
        dir_coefj = 1
    if "inferior" in space :
        dir_coefk = -1
    elif "superior" in space : 
        dir_coefk = 1
    
    ijk_coeff   = np.array([dir_coefi , dir_coefj , dir_coefk])
    orgin_coeff = np.array([-dir_coefi , -dir_coefj , -dir_coefk])
    niftiDir    = directions*ijk_coeff
    niftiOrigin = np.expand_dims(origin*orgin_coeff , axis=1)
    niftiAffine = np.concatenate(
        [
            niftiDir , 
            niftiOrigin
        ] , 
        axis=1
    )
    niftiAffineFix = np.concatenate([niftiAffine , np.array([[0.0 , 0.0 , 0.0 , 1.0]])] , axis=0)
    nifiImg = nib.Nifti1Image(
        data , 
        affine=niftiAffineFix
    )

    nib.save(nifiImg , output)