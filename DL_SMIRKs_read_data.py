import numpy as np
import os
import cv2   
from SMIRKs_head import *
import copy

def read_data(folder_name):
    
    raw_data = {}     
    for i in range(len(folder_name)):
        file_name = [] 
        for file in os.listdir(r'./%s' %(folder_name[i])):
            file_name.append(file) 
            
        file_name.remove('.DS_Store')
        
        img_set = []
        for j in range(len(file_name)):
            img = cv2.cvtColor(cv2.imread(r'./%s/%s' %(folder_name[i], file_name[j])), cv2.COLOR_BGR2GRAY)
            thresh, img_bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_bw = convert(img_bw) 
            img_set.append(img_bw)   
            
        raw_data['%s' %(folder_name[i])] = img_set
    return raw_data

# 'raw_data' and 'label_raw_data' are both dictionary.
def generate_data_set(raw_data, label_raw_data):
    
    train_X = np.ones([50*80, 1])
    train_y = np.ones([1, 1])
    for key in raw_data:
        note_set = raw_data[key] 
        for j in range(len(note_set)):
            img = note_set[j] / 255
            img_rz = cv2.resize(img, (50, 80), interpolation = cv2.INTER_NEAREST)
            img_ = img_rz.reshape(50*80, 1)
            
            train_X = np.column_stack((train_X, img_))
            label = label_raw_data[key]
            train_y = np.column_stack((train_y, label))
            
    train_X = np.delete(train_X, 0, axis = 1)
    train_y = np.delete(train_y, 0, axis = 1)
    return train_X, train_y
   
def preprocess_data(raw_data):
    dash_filter = np.ones([1, 2])  
    data_new = {}
    
    for key in raw_data:
        note_set = raw_data[key] 
        note_set_new = []
        
        for j in range(len(note_set)):
            img = note_set[j]
            
            height, width = img.shape
            staff_line_filter = np.ones([1, width//3])
            staff, idx_staff = get_staff_lines(img, dash_filter, staff_line_filter)
            img_ = remove_staff_lines_DL(img, staff)
            #img_ = cv2.dilate(img_, np.ones((2, 2)), iterations = 1)
            
            note_set_new.append(img_)
            
        data_new[key] = note_set_new
    
    return data_new

