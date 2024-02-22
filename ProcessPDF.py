import numpy as np
import cv2
import fitz
from PIL import Image
import matplotlib.pyplot as plt

def PDF_to_images(path):
    """ 
    Open the pdf and return all pages as a list of array
    Args:
        path (path): python readable path
        POPPLER (path): Defaults to POPPLER_PATH.

    Returns:
        list of arrays: all pages as array
    """
    images = fitz.open(path)
    res = []
    for image in images:
        pix = image.get_pixmap(dpi=300)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        img = np.array(img)
        res.append(img)
    return res

def binarized_image(image):
    """ 
    Binarized one image thanks to OpenCV thersholding. niBlackThresholding has been tried.
    Args:
        image (np.array) : Input image

    Returns:
        np.array : binarized image
    """
    y, x = image.shape[:2]
    blur = cv2.bilateralFilter(image,5,200,200)

    gray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Del black borders
    mask = np.zeros(thresh.shape, dtype=np.uint8)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rects = [cv2.minAreaRect(cnt) for cnt in cnts]

    main_contours = [cnts[i] for i, rect in enumerate(rects) if rect[1][0]*rect[1][1]>x*y*0.6]

    if len(main_contours) == 0:
        return thresh

    cv2.fillPoly(mask, cnts, [255,255,255])
    mask = 255 - mask


    result = cv2.bitwise_or(thresh, mask) 

    # plt.imshow(result)
    # plt.show()

    return result

def get_rectangles(bin_image, kernel_size=(9,9), show=False):
    """
    Extract the minimum area rectangle containg the text. 
    Thanks to that detect if the image is a TABLE format or not.
    Args:
        bin_image (np.array): The binarized images
        kernel_size (tuple, optional): . Defaults to (3,3).
        interations (int, optional): _description_. Defaults to 2.

    Returns:
        format (str) : "table" or "other"
        rectangle (cv2.MinAreaRect) : The biggest rectangle of text found in the image
    """
    y, x = bin_image.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(~bin_image, kernel, iterations=10)

    contours,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contour found : Crop is impossible, processed image is return.")
        return []

    rectangles = [list(cv2.minAreaRect(contour)) for contour in contours]
        
    # if show:
    #     print("all boxes")
    #     im = bin_image
    #     for rect in rectangles:
    #         box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
    #         box = np.int0(box)
    #         cv2.drawContours(im,[box],0,(0,0,0),5)
    #     plt.imshow(im, cmap="gray")
    #     plt.show()

    filtered_rects = []
    coord = lambda x: [int(x[0][0]-x[1][0]/2), int(x[0][1]-x[1][1]/2), int(x[0][0]+x[1][0]/2), int(x[0][1]+x[1][1]/2)]
    for rect in rectangles:
        rect=list(rect)
        if 45<rect[-1]: # Normalize to get x,y,w,h in the image refrential
                rect[1] = (rect[1][1], rect[1][0])
                rect[-1] = rect[-1]-90
        overlap_found = False
        for f_rect in filtered_rects:
            coord1 = coord(rect)
            coord2 = coord(f_rect)
            iou = get_iou(coord1, coord2)
            if iou > 0.2 :
                overlap_found = True
                break
        if not overlap_found:
            filtered_rects.append(rect)

    rectangles =  [list(rect) for rect in rectangles if ((rect[1][0]>0.5*x and rect[1][1]>0.05*y) or (rect[1][0]>0.05*x and rect[1][1]>0.5*y) or (rect[1][0]>0.95*x))]  

    if show:
        print("Filtered boxes")
        im = bin_image.copy()
        for rect in rectangles:
            box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
            box = np.int0(box)
            cv2.drawContours(im,[box],0,(0,0,0),8)
        plt.imshow(im, cmap="gray")
        plt.show()

    return rectangles

def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def get_main_rect(bin_image, rectangles):
    
    y,x = bin_image.shape 
    rot = 0     
    maxarea = 0
    for i, rect in enumerate(rectangles):
        # Correct the X and Y value if landscape rect
        area = rectangles[i][1][0]*rectangles[i][1][1]
        if 45<rect[-1]:
            rectangles[i][1] = (rectangles[i][1][1], rectangles[i][1][0])
            if maxarea < area:
                rot = rect[-1]-90
                maxarea = area
        elif maxarea < rectangles[i][1][0]*rectangles[i][1][1]:
            rot = rect[-1] # The rot angle is chosen by taken the biggest rect angle
            maxarea = area
    rot = 0 if abs(rot)>4 else rot

    x1_y1_x2_y2 = []
    for rect in rectangles:
        x1_y1_x2_y2.append([rect[0][0]-rect[1][0]*0.5, rect[0][1]-rect[1][1]*0.5, rect[0][0]+rect[1][0]*0.5, rect[0][1]+rect[1][1]*0.5])
    

    x, y = min(x1_y1_x2_y2, key=lambda x: x[0])[0], min(x1_y1_x2_y2,  key=lambda x: x[1])[1]
    x2, y2 = max(x1_y1_x2_y2,  key=lambda x: x[2])[2], max(x1_y1_x2_y2,  key=lambda x: x[3])[3]
    w, h = x2-x, y2-y

    main_box = [(int(x+w/2),int(y+h/2)), (int(w),int(h)), rot]

    return main_box
        
def crop_and_adjust(bin_image, rect):
    """Crop the blank part around the found rectangle.

    Args:
        bin_image (np.array): The binarized image
        rect (cv2.MinAreaRect) : The biggest rectangle of text found in the image ; format (x, y , w, h)
    Returns:
        cropped_image (np.array) : The image cropped thanks to the rectangle
    """
    def _points_filter(points):
        """
        Get the endpoint along each axis
        """
        points[points < 0] = 0
        xpoints = sorted(points, key=lambda x:x[0])
        ypoints = sorted(points, key=lambda x:x[1])
        tpl_x0, tpl_x1 = xpoints[::len(xpoints)-1]
        tpl_y0, tpl_y1 = ypoints[::len(ypoints)-1]
        return tpl_y0[1], tpl_y1[1], tpl_x0[0], tpl_x1[0]
    
    if len(rect)==0 : 
        return bin_image

    box = np.intp(cv2.boxPoints(rect))    
    # Rotate image
    angle = rect[2]
    rows, cols = bin_image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1) # Rotation matrix
    img_rot = cv2.warpAffine(bin_image,M,(cols,rows))

    rect_points = np.intp(cv2.transform(np.array([box]), M))[0] # points of the box after rotation
    y0, y1, x0, x1 = _points_filter(rect_points) # get corners
    cropped_image = img_rot[y0:y1, x0:x1]
    
    return cropped_image

def get_adjusted_image(bin_image, show=False):

    rectangles = get_rectangles(bin_image, show=False)
    main_rect = get_main_rect(bin_image, rectangles)
    adjusted_image = crop_and_adjust(bin_image, main_rect)

    if show:
        print("final image")
        plt.imshow(adjusted_image, cmap="gray")
        plt.show()

    return adjusted_image

# LINE FUNCTIONS

def delete_lines(bin_image): # Unused function wich delete (approximatly) lines on an image
    
    bin_image = np.array(bin_image).astype(np.uint8)
    copy = bin_image.copy()
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    detected_lines = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    bin_image[detected_lines==255] = 0
    cnts = cv2.findContours(detected_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(copy, [c], -1,(255, 255,255), 2)
    return copy

def HoughLines(bin_image, mode="vertical"):
    (cst, _) = (0,1) if mode == "vertical"  else (1,0) # The specified axis is the constant one
    image = bin_image.copy()
    ksize = (1,6) if mode == "vertical" else (6,1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=ksize)
    bin_image = cv2.dilate(bin_image, kernel, iterations=4)
    # plt.imshow(bin_image)
    # plt.show()
    edges = cv2.Canny(bin_image,50,150,apertureSize=3)
 
    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=60, # Min number of votes for valid line
                minLineLength=20, # Min allowed length of line
                maxLineGap=290 # Max allowed gap between line for joining them ; Set according to the SEMAE format
                )
    
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        line  = [(x1,y1),(x2,y2)]
        if abs(line[0][cst]-line[1][cst])<40:
            lines_list.append(line)
    return lines_list 

def agglomerate_lines(lines_list, mode = "vertical"):
    (cst, var) = (0,1) if mode == "vertical"  else (1,0) # The specified axis is the constant one
    #Delete to closed vertical lines
    threshold = 90 if mode == "vertical" else 50
    clean_lines_list = []
    agglomerate_lines = []
    for i, line in enumerate(lines_list):
        if not i in agglomerate_lines:
            new_line = line
            for j in range(i+1, len(lines_list)):
                if not j in agglomerate_lines :
                    test_line = lines_list[j]
                    condition_1 = abs((new_line[0][cst]+new_line[1][cst])/2 - (test_line[0][cst]+test_line[1][cst])/2) < threshold # Close enough
                    m, M = min(new_line[0][var], new_line[1][var]), max(new_line[0][var], new_line[1][var])
                    condition_2 = (m<max(test_line[0][var], test_line[1][var]<M) or m<min(test_line[0][var], test_line[1][var])<M) # Overlap
                    if condition_1 and condition_2:
                        agglomerate_lines.append(j)
                        cst_M = int((min(new_line[0][cst], new_line[1][cst], test_line[0][cst], test_line[1][cst]) + max(new_line[0][cst], new_line[1][cst], test_line[0][cst], test_line[1][cst]))/2)
                        var_1, var_2 = min(new_line[0][var], new_line[1][var], test_line[0][var], test_line[1][var]), max(new_line[0][var], new_line[1][var], test_line[0][var], test_line[1][var])
                        res = [[0,0], [0,0]] # New line may not support tuple assignment
                        res[0][cst], res[1][cst], res[0][var], res[1][var] = cst_M, cst_M, var_1, var_2
                        new_line = res
            clean_lines_list.append(new_line)
    clean_lines_list = sorted(clean_lines_list, key=lambda x: x[0][cst])
    return(clean_lines_list)

if __name__ == "__main__":

    print("No")
    path = r"C:\Users\CF6P\Desktop\ELPV\Data\scan7.pdf"
    images = PDF_to_images(path)
    images = images[0:]
    for im in images:
        bin_image = binarized_image(im)
        rect = get_rectangles(bin_image)