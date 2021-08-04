# import the necessary packages
import numpy as np
import cv2
import imutils
import pytesseract
import tempfile
from PIL import Image
from image_ocr import detect_characters,detect_words
from key_info_extract import key_info_extract
import os
from write_csv import append_list_as_row


# Resize Image (PIL)
def set_image_size(im):
    im.thumbnail((1200, 1200))
    return im


# Called by image_preprocess
# Set Image DPI(PIL)
def set_image_dpi(directory, filename):
    file_path = os.path.join(directory, filename)
    im = Image.open(file_path)
    im = set_image_size(im)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(file_path, dpi=(300, 300))
    return file_path


# Set Image DPI(PIL)
def set_image_dpi_without_rotate(directory, filename):
    file_path = os.path.join(directory, filename)
    im = Image.open(file_path)
    im = set_image_size(im)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    # Store in Temporary File
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


# Called by image_preprocessing & image_preprocessing_without_wraped
# Increase Sharpness
def increase_sharpness(rotated):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)
    return sharpened


# Called by four_point_transform
# Get Order Point
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


# Called by canny_edge_detection
# Four Point Transform
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# Called by image_preprocessing
# Image Background Removal
def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 10, 50)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    ratio = 1
    orig = image.copy()
    pts = np.array(screenCnt.reshape(4, 2) * ratio)
    warped = four_point_transform(orig, pts)
    return warped


# Called by image_preprocessing & image_preprocessing_without_wraped
# Otsu's thresholding
def otsu_thresholding(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


# Called by image_preprocess
# Process Image with Candy Edge Detection
def image_preprocessing(image):
    # Call Candy Edge Detection
    try:
        warped = canny_edge_detection(image)
    except:
        warped = image
    # Threshold the Image
    threshold = otsu_thresholding(warped)
    # Increase Image Sharpness
    threshold = increase_sharpness(threshold)
    # Boxed the words
    boxed, issuer = detect_words(threshold)
    # Extract Text from Image
    x = pytesseract.image_to_string(threshold)
    return x


# Called by image_preprocess
# Process Image
def image_preprocessing_without_wraped(image):
    warped = image
    # Threshold the Image
    threshold = otsu_thresholding(warped)
    # Increase Image Sharpness
    threshold = increase_sharpness(threshold)
    # Boxed the words
    boxed, issuer = detect_words(threshold)
    # Extract Text from Image
    x = pytesseract.image_to_string(threshold)
    return x


# Called by main_program
def image_preprocess(directory):
    os.chdir(directory)
    # Define Tesseract OCR
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    raw_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            label = (filename.rsplit('.', 1)[0])[-1]
            image = set_image_dpi(directory, filename)
            image = cv2.imread(image)
            #label = filename[0]
            image = set_image_dpi(directory, filename)
            image = cv2.imread(image)
            x = image_preprocessing_without_wraped(image)
            # Save text extracted by OCR to file
            f = open("text.txt", "w")
            f.write(x)
            f.close()
            # Extract key information from receipt
            issuer, issue_date_list, price_list, item_list, keyword_list, total_price = key_info_extract(x)
            # Reprocess the receipt if price not found
            if len(price_list) == 0:
                image1 = set_image_dpi(directory, filename)
                image1 = cv2.imread(image1)
                x = image_preprocessing(image1)
                # Save text extracted by OCR to file
                f = open("text.txt", "w")
                f.write(x)
                f.close()
                # Extract key information from receipt
                issuer, issue_date_list, price_list, item_list, keyword_list, total_price = key_info_extract(x)
            # Save information into csv file
            row_contents = [filename, issuer, issue_date_list, item_list, total_price, keyword_list, label]
            append_list_as_row('output.csv', row_contents)

