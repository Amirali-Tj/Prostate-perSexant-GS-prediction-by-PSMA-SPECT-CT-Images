import subprocess as sub
import nibabel as nib
import numpy as np
import os
import shutil

mainPath = "sample"
for name in os.listdir(mainPath) :
    if name != ".DS_Store" : # for mac
        print(name)
        container      = os.path.join(mainPath , name , "PSMA")
        pFolderName    = os.listdir(container)
        pFolderName.remove(".DS_Store") # for mac
        fullFolderName = os.path.join(container , pFolderName[0])

        try :
            os.mkdir(os.path.join("PSMA nii" , name))
        except :
            pass

        for file in os.listdir(fullFolderName) :
            if "CTACL" in file :
                shutil.copy(
                    os.path.join(fullFolderName , file) ,
                    os.path.join("PSMA nii" , name)
                )
            elif "IRAC" in file :
                shutil.copy(
                    os.path.join(fullFolderName , file) ,
                    os.path.join("PSMA nii" , name)
                )