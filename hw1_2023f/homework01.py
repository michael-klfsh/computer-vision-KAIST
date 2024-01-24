# import cv2
# import numpy as np
# import time

# Task 1.1
# B = cv2.imread('grizzlypeakg.jpg',0)
# start = time.time()
# for n in range(100):
#     A = B
#     m1, n1 = A.shape
#     print(n)
#     for i in range(m1):
#         for j in range(n1):
#             if A[i,j] <= 10:
#                 A[i,j] = 0
# estimated = time.time() - start
# print(estimated)

#Task 1.1 and 1.3

# import cv2
# import numpy as np

# original = cv2.imread('grizzlypeakg.jpg')
# start = time.time()
# for n in range(100):
#     A = original
#     print(n)
#     B = A <= 10
#     print(B.shape)
#     A[B] = 0
# estimated = time.time() - start
# print(estimated)
# cv2.imwrite('task 1.3.jpg', A)

#Task 2.1
import cv2
import numpy as np
I = cv2.imread('gigi.jpg').astype(np.uint8)
I = I - 40
#I[I>215] = int(0)
cv2.imwrite('result_old.png', I)