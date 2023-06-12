import numpy as np
import cv2

cv2.setNumThreads(1)


x = np.arange(3 * 225 * 225).reshape(225, 225, 3).astype("uint8")
print(x.shape)

out = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
print(out.shape)
