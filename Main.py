import cv2 as cv

image = cv.imread("Lock.png")
cv.imshow("set", image)

cv.waitKey(0)
