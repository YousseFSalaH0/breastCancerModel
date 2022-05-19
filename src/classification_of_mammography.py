import cv2
from PIL import Image
import numpy as np
import math
import pickle


def breast_display(img):  
    """ Respresent breast part """
    # imgCopy = np.copy(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    # Contour breast part
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Add the area of each contour result into contour_area_list
    contour_area_list = [len(contours)]
    count = 0
    for i in contours:
        area = cv2.contourArea(i)
        contour_area_list.append(area)
        # print "contour index ", count, " area = ", area
        count += 1
    # Get max area and draw contour on it.
    max_area_index = contour_area_list.index(max(contour_area_list)) - 1
            
    # Making mask to extract image to new image
    mask = np.zeros(img.shape, np.uint8)
    if max_area_index >= 0:
        cnt = contours[max_area_index]
        cv2.drawContours(mask, [cnt], 0, 255, -1)
    
    res = cv2.bitwise_and(img, img, mask=mask)
    return res, max(contour_area_list), mask


def getBreastCoordinates(img):
    '''get breast coordinations'''
    _, _,BinaryMask = breast_display(img)
    active_px = np.argwhere(BinaryMask!=0)
    active_px = active_px[:,[1,0]]
    x,y,w,h = cv2.boundingRect(active_px)
    return x,y,w,h

def classify_right_left_mlo_cc(image_array):
    '''classification algorithm'''
    features = []
    image = image_array
    # check if input image is grayscale image
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    x, _, w, _ =  getBreastCoordinates(image)

    # generate a new image for feature extraction
    breast_image = image[:, x:x+w]

    # generate a binary image from breast image
    min_ = image.min()
    max_ = image.max()
    breast_image_bw = cv2.threshold(breast_image, min_, max_, cv2.THRESH_BINARY)[1]

    ## Classify if image left or right
    im = Image.fromarray(image)
    # divide image into sparate five images vertically
    pictures = 5
    x_width, y_height = im.size
    edges = np.linspace(0, x_width, pictures+1)
    result = []
    for start, end in zip(edges[:-1], edges[1:]):
        box = (start, 0, end, y_height)
        a = im.crop(box)
        arr = np.array(a)
        result.append(arr.sum())

    img_type = None
    feature1 = None
    feature2 = None
    # check if pixels summation of first image (left) >  pixels summation of last image (right)
    if result[0] > result[-1]:
        # then, breast will be at the left (Right view)
        img_type = "Right"
        feature1 = extract_feature_1(img_type, breast_image_bw)
        feature2 = extract_feature_2(img_type, breast_image_bw)

    # check if pixels summation of last image (right) >  pixels summation of first image (left)
    if result[-1] > result[0]:
        # then, breast will be at the right (LEft View)
        img_type = "Left"
        feature1 = extract_feature_1(img_type, breast_image_bw)
        feature2 = extract_feature_2(img_type, breast_image_bw)
    
    # Handling some errors may occure from extracting feature
    if feature1 == math.inf:
        feature1 = 100

    if feature2 == math.inf:
        feature2 = 100

    if feature1 == math.nan:
        feature1 = 0
    if feature2 == math.nan:
        feature2 = 0

    # features.append(feature1)
    # features.append(feature2)
    # print(features)
    # Prediction of the input image
    # final_result = LOADED_MODEL.predict([features])[0]

    # if final_result == 0:
    #     final_result = img_type + " - MLO"
    # if final_result == 1:
    #     final_result = img_type + " - CC"
    final_result = ''
    if img_type == 'Right':
        img_type = 'L'
    else: 
        img_type = 'R'
    # print(feature1)
    # print(feature2)
    # print('----------------------')
    if feature1 >= 0.8 and feature2 >= 0.8:
        final_result = img_type + "_MLO"
    else:
        final_result = img_type + "_CC"

    return final_result


def extract_feature_1(image_type, image):
    '''
    division of (10% , 10%) kernel at left or right corner 
    by  (10%, 10%) kernel at left or right side at middle of an image
    '''
    height = image.shape[0]
    width = image.shape[1]

    center_height = height // 2
    center_width = width // 2

    if image_type == "Right":
        img_1 = image[:int(height*.10), :int(width*.10)]
        img_2 = image[center_height-int(height*.05):center_height+int(height*.05), :int(width*.10)]
        
        return (img_1.sum() / img_2.sum())

    if image_type == "Left":
        img_1 = image[:int(height*.10), width-int(width*.10):width]
        img_2 = image[center_height-int(height*.05):center_height+int(height*.05), width-int(width*.10):width] 
        return (img_1.sum() / img_2.sum())


def extract_feature_2(image_type, image):
    '''
    division of (50%, 15%) kernel at the left or right corner
    by (50%, 15%) kernel at the middle of an image
    '''
    height = image.shape[0]
    width = image.shape[1]

    center_height = height // 2
    center_width = width // 2

    new_height = int(height * .5)
    new_width = int(width * .15)

    if image_type == "Right":
        # cropped image at the center
        im_1 = image[(center_height - (new_height // 2)) : (center_height + (new_height // 2)) , (center_width - (new_width // 2)) : (center_width + (new_width // 2))] 

        # crop image at the left of image
        im_2 = image[:new_height, :new_width]

        return im_2.sum() / im_1.sum()

    if image_type == "Left":
        im_1 = image[(center_height - (new_height // 2)) : (center_height + (new_height // 2)) , (center_width - (new_width // 2)) : (center_width + (new_width // 2))] 

        # crop image at the right of image
        new_width_2 = width - new_width
        im_2 = image[:new_height, new_width_2:]

        return im_2.sum() / im_1.sum()

# Read an image for testing
# image = cv2.imread(r"C:\Users\moham\OneDrive\Desktop\breast_cancer_model\images\3_R_MLO.png")
# Calling function to classify the image
# result = classify_right_left_mlo_cc(image)
# Print the result
# print(result)

# test_images = [f for f in os.listdir("breast_detection/")]
# for image in test_images:
#     # Read an image for testing
#     image = cv2.imread("breast_detection/"+image)
#     # Calling function to classify the image
#     result = classify_right_left_mlo_cc(image)
#     # Print the result
#     print(result)
#     print("==========================")
