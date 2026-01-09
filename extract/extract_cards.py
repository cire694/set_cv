import cv2 as cv
import numpy as np
import os
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.predict import predict_from_opencv

# Preprocessing: 
# 1) remove color data to simplify edge math
# 2) Remove Gaussian noise (variations in pixel values follow a normal distribution)
# 3) use canny to highlight areas of high contrast
# 4) use morphologyEx to close gaps in outline

# Detection: 
# 1) find contours
# 2) approximate contours into polygons using cv.approxPolyDP. 
# 3) sort by contour area (might not be needed)

# Transformation: 
# 1) sort the four points into TL, TR, BR, BL
# 2) use cv.getPerspectiveTransform to calculate the math needed to "flatten" the perspective, make it top-down
# 3) use cv.warpPerspective to get the cropped top down image

#Canny edge detection: 
# - first denoise
# - then we the sobel kernel to get the first derivative of the vertical and horiontal direction.
#   using that information we can then calculate the magnitude and direction of the gradient for each pixel.
# - Non maximum suppression: scan the entire image to remove non-edge pixels. 
#   For every pixel, check if it is a local maxiumum in its neighborhood in the direction of the gradient.
#   If so, move on to the next stage, otherwise suppressed (set to zero). 
# - Thresholding using minVal and maxVal: any edge with intensity more than maxVal are edges, and those
#   below minVal are non-edges. 
#   Anything in-between is judged based on their connectivity, aka if they are connected to pixels we know are edges. 

def find_card_contours(path, get_contours = False):

    #preprocessing
    img = cv.imread(path) #all our edge detection algorithms use grayscale
    if img is None: return []

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0) #kernel (3, 3). If SD in X (and Y) is both zero -> SD is computed from kernel width and height
    edges = cv.Canny(blurred, 100, 200)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(3, 3))
    closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    #detection
    contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#external retreival mode, no nesting of contours. chain_approx_simple means only store boundary points, not intermediate
    
    if get_contours: 
        all_card_contours = []
    cards = []
    for cnt in contours: 
        area = cv.contourArea(cnt)

        if area > 5000:
            perimeter = cv.arcLength(cnt, True)

            #input, epsilon, closed curve or not
            #epsilon: if line deviates by less than 0.02*perimeter, treat it as straight line. 
            approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True) 
            if len(approx) != 4:
                continue
            
            pts = approx.reshape(4, 2) #(4, 1, 2) -> (4, 2)

            ordered_pts = reorder(pts)
            
            width, height = 200, 150 #match input to model
            pts_dst = np.array([ #where should the points go in the final crop
                [0, 0],
                [width, 0],
                [width, height], 
                [0, height]
            ], dtype="float32")

            M = cv.getPerspectiveTransform(ordered_pts.astype("float32"), pts_dst) # creates the warping to a (W, H) img
            warped = cv.warpPerspective(img, M, (width, height))

            cards.append(warped)
            if get_contours:
                all_card_contours.append(approx)
    if get_contours: 
        return cards, all_card_contours
    return cards



def reorder(pts): 
    '''
    Returns the points in the order [Top left, Top right, Bottom right, Bottom left]
    '''
    assert len(pts) == 4
    #TL: smallest x+y
    #BR: largest x+y
    #TR: smallest y-x
    #BL: largest y-x
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis = 1) #sum the rows (x + y)
    
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1) #computes y-x
    rect[1] = pts[np.argmax(d)]
    rect[3] = pts[np.argmin(d)]
    return rect


def get_all_cards(path): 
    return [predict_from_opencv(card_img) for card_img in find_card_contours(path)]

def get_card_contours(path):
    return find_card_contours(path, get_contours=True)[1]