import cv2
import matplotlib.pyplot as plt
import numpy as np


def mask_image(image):
    height = image.shape[0]
    width = image.shape[1]
    points = np.array([[(width, height),(2300, 1200),(1400, 1200),(-2300,height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, points, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_image(img, title):
    plt.figure(), plt.imshow(img, cmap="gray"), plt.title(title)


img = cv2.imread("roadlines.jpg")

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.blur(gray_image, (5, 5))

(thresh, blackAndWhiteImage) = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY)

edge_image = cv2.Canny(blackAndWhiteImage, 200, 255)

masked_image = mask_image(edge_image)

lines = cv2.HoughLinesP(masked_image, 1, np.pi/180,40)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),4)

display_image(img, "Original Image")
display_image(blackAndWhiteImage, "Black and White Image")
display_image(edge_image, "Edge Image")
display_image(masked_image, "Masked Image")

plt.show()
