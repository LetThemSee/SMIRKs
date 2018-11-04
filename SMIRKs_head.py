import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.ndimage
from collections import Counter

def test_filter_size(img_div, diff_staff):
    cv2.imshow('test0', img_div)
    cv2.waitKey(1)

    disk_filter = generate_disk_filter(diff_staff//2.5)
    img_note1 = opening(img_div, disk_filter) 

    cv2.imshow('test1', img_note1)
    cv2.waitKey(1)
        
    disk_filter2 = generate_disk_filter(diff_staff//3)
    img_note2 = opening(img_div, disk_filter2) 

    cv2.imshow('test2', img_note2)
    cv2.waitKey(1)

    cv2.destroyAllWindows() 

def convert(image):
    ''' 
    # Attempt 1
    m, n = image.shape
    
    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if image[i, j] == 255:
                result[i, j] = 0
                
            if image[i, j] == 0:
                result[i, j] = 255
    
    return result
    '''
    # Attempt 2
    result = copy.copy(image)
    result[image == 255] = 0
    result[image == 0] = 255
    
    return result  


def get_most_common(data):
     most_common = Counter(data).most_common(1)
     
     return most_common[0][0] 

def find_note_interval(moments):
    space = []
    for i in range(len(moments)-1):
        space.append(moments[i+1, 1] - moments[i, 1])
    interval = np.median(space)

    return interval

def get_most_common_(hist):
    total = 0
    count = 0
    for i in range(len(hist)):
        if hist[i] > 0:
            count += 1
            total += hist[i]
    most_common = int(total / count)

    return most_common

def get_hist_byColumn(img):
    height, width = img.shape
    hist = [0] * width    
    
    for j in range(width):
        hist[j] = sum(img[:, j]) / 255
    
    return hist
        
def get_staff_lines(img, dash_filter, staff_line_filter):
    
    # Dilation: Smooth the horizontal lines
    temp = cv2.dilate(img, dash_filter, iterations = 1)
    # Erosion: keep the horizontal lines only
    img_staff_lines = cv2.erode(temp, staff_line_filter, iterations = 1)
    img_staff_lines = cv2.dilate(img_staff_lines, staff_line_filter, iterations = 1)
    
    height, width = img_staff_lines.shape
    
    hist = [0] * height
    for i in range(height):
        hist[i] = np.sum(img_staff_lines[i, :]) / 255  
    '''
    plt.figure
    plt.bar(range(height), hist)
    '''
    idx_staff = []
    for i in range(height):
        if hist[i] > 0.1 * width:
            idx_staff.append(i)
            
    # Modfiy the index of staff to make sure it only has 5 entries
    # One staff might take up two or more lines
    idx_staff_new = []
    for i in range(len(idx_staff) - 1):
        if idx_staff[i+1] - idx_staff[i] > 1:
            idx_staff_new.append(idx_staff[i])     
    idx_staff_new.append(idx_staff[-1])
    
    return img_staff_lines, idx_staff_new

def partition_image(img_orig, idx_staff):
    n_staff = len(idx_staff) // 5
    diff_staff = (idx_staff[4] - idx_staff[0]) // 4
    
    height, width = img_orig.shape
    
    img_div_set = []
    for i in range(n_staff):
        idx_start = idx_staff[5*i] - 2 * diff_staff
        idx_end = idx_staff[5*i+4] + 2 * diff_staff
        
        if idx_start < 0:
            idx_start = 0
            
        if idx_end > height:
            idx_end = height
        
        img_div = img_orig[idx_start:idx_end, :]
        img_div_set.append(img_div)
    
    return img_div_set


def remove_staff_lines(img, staff_lines):
    image_result = copy.copy(img)
    
    image_result[staff_lines == 255] = 0
    
    # Use closing to fill up missing parts
    tmp = 2
    # 1. Vertical closing
    vertical_filter = np.ones([tmp, 1]) 
    image_result = cv2.dilate(image_result, vertical_filter, iterations = 1)
    image_result = cv2.erode(image_result, vertical_filter, iterations = 1)

    # 2. Horizontal closing
    horizontal_filter = np.ones([1, tmp])
    image_result = cv2.dilate(image_result, horizontal_filter, iterations = 1)
    image_result = cv2.erode(image_result, horizontal_filter, iterations = 1)
    
    return image_result

def remove_staff_lines_DL(img, staff_lines):
    image_result = copy.copy(img)
    image_result[staff_lines == 255] = 0
    
    # Use closing to fill up missing parts
    tmp = 10
    # 1. Vertical closing
    vertical_filter = np.ones([tmp, 1]) 
    image_result = cv2.dilate(image_result, vertical_filter, iterations = 1)
    image_result = cv2.erode(image_result, vertical_filter, iterations = 1)

    # 2. Horizontal closing
    horizontal_filter = np.ones([1, tmp])
    image_result = cv2.dilate(image_result, horizontal_filter, iterations = 1)
    image_result = cv2.erode(image_result, horizontal_filter, iterations = 1)
    
    return image_result


def generate_disk_filter(radius):
    
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype = 'uint8')
    
    return disk

def opening(image, mor_filter):
    
    temp = cv2.erode(image, mor_filter, iterations = 1)
    image_result = cv2.dilate(temp, mor_filter, iterations = 1)
    
    return image_result
 
def determine_note_pos(index_staff, moment_row_ind):
    # Add the 6th staff line
    diff_staff = index_staff[1] - index_staff[0]
    index_staff = index_staff + [index_staff[-1] + diff_staff]
    
    # Partition the staff line
    step = diff_staff // 4

    if moment_row_ind >= index_staff[5]-step and moment_row_ind < index_staff[5]+step:
        return 1 #Do
    if moment_row_ind >= index_staff[4]+step and moment_row_ind < index_staff[5]-step:
        return 2 #Re
    if moment_row_ind >= index_staff[4]-step and moment_row_ind < index_staff[4]+step:
        return 3 #Mi
    if moment_row_ind >= index_staff[3]+step and moment_row_ind < index_staff[4]-step:
        return 4 #Fa
    if moment_row_ind >= index_staff[3]-step and moment_row_ind < index_staff[3]+step:
        return 5 #Sol
    if moment_row_ind >= index_staff[2]+step and moment_row_ind < index_staff[3]-step:
        return 6 #La
    if moment_row_ind >= index_staff[2]-step and moment_row_ind < index_staff[2]+step:
        return 7 #Ti
    
    if moment_row_ind >= index_staff[1]+step and moment_row_ind < index_staff[2]-step:
        return 8 #Do
    if moment_row_ind >= index_staff[1]-step and moment_row_ind < index_staff[1]+step:
        return 9 #Re
    if moment_row_ind >= index_staff[0]+step and moment_row_ind < index_staff[1]-step:
        return 10 #Mi
    if moment_row_ind >= index_staff[0]-step and moment_row_ind < index_staff[0]+step:
        return 11 #Fa
    
def compute_moments(contours):
    n_notes = len(contours)
    moments = np.empty((0, 2))
    
    for i in range(n_notes):
        cnt = contours[i]
        M = cv2.moments(cnt)
        col_ind = int(M['m10']/M['m00'])
        row_ind = int(M['m01']/M['m00'])  # We only care about its row index
        centroid = np.array([row_ind, col_ind])
        moments = np.vstack((moments, centroid))
    
    # Sort by column index (from left to right)
    tmp_arg = np.argsort(moments[:, 1]) 
    moments = moments[tmp_arg]
    
    return moments

def determine_edge(hist, median):
    width = len(hist)
    
    count = 0
    for i in range(width):        
        if hist[i] > median and count == 0:
            count += 1
            start = i - 1
            
        if hist[i] <= median and count == 1:
            end = i
            break
    return start, end
