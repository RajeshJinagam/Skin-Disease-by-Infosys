import cv2
import numpy as np
 
img= cv2.imread("Skin_Disease\media\psoriasis-treatment-skinpase-clinic-2-300x200.jpg")
img_resize=cv2.resize(img,(250,250))
cv2.imshow("without normalization",img)
img_nor=img_resize/255.0
cv2.imshow('with normalization',img_nor)
img_dim=np.expand_dims(img_nor,axis=0)
print(img_dim)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

