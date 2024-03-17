import cv2
import numpy as np
from map_utils import detect_water

is_water = detect_water()

out = np.where(is_water, 255, 0).astype(np.uint8)
cv2.imwrite('water.png', out)
cv2.imshow('water', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
