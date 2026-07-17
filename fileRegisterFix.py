import os
import nibabel as nib
import numpy as np


mainPath = "PSMA nii"
for name in os.listdir(mainPath) :
    if name != ".DS_Store" :
        pFolder = os.path.join(mainPath , name)
        for file in os.listdir(os.path.join(pFolder , "nii files")) :
            if "Oncology" in file and "nii" in file :

                spect_file = nib.load(os.path.join(pFolder , "nii files" , file))

                # array fix
                array      = spect_file.get_fdata()
                array      =  np.flip(
                    array ,
                    axis=2
                )

                # origin fix on K dim
                spacing_k = spect_file.affine[2][2]
                origin_k  = spect_file.affine[2][3]
                originFix = origin_k - (127*spacing_k) # spect matrix 128*128
                spect_file.affine[2][3] = originFix
                print(spect_file.affine)

                # save fixed spect
                spect_obj = nib.Nifti1Image(array , affine=spect_file.affine , header=spect_file.header)
                nib.save(spect_obj , os.path.join(pFolder , "nii files" , file))