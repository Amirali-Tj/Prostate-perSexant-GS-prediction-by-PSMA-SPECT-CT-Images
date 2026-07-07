import subprocess as sub
import os

mainPath = "PSMA nii"
for name in os.listdir(mainPath) :
    if name != ".DS_Store" :
        fullName = os.path.join(mainPath , name)

        try :
            os.mkdir(os.path.join(fullName , "nii files"))
        except :
            pass

        sub.run(
            [
                "./dcm2niix" ,
                "-o" ,
                os.path.join(fullName , 'nii files') ,
                fullName
            ]
        )
