import numpy as np
import matplotlib.pyplot as plt  
# from DL_SMIRKs_header import *
from DL_SMIRKs_read_data import *
from DL_model import *

if __name__=="__main__":
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    # Set up
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----  
    # Read training set
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    folder_name = ['quarter_note', 'eighth_note']
    label_raw_data = {"quarter_note": 1, "eighth_note":0}
    
    raw_data = read_data(folder_name)
    data_pre = preprocess_data(raw_data)
    train_X, train_y = generate_data_set(data_pre, label_raw_data)
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----  
    # Read testing set
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    folder_name_test = ['test quarter_note', 'test eighth_note']
    label_raw_data = {"test quarter_note": 1, "test eighth_note":0}
    
    raw_data_test = read_data(folder_name_test)
    data_pre_test = preprocess_data(raw_data_test)
    test_X, test_y = generate_data_set(data_pre_test, label_raw_data)
# Show!
    cv2.imshow('test', data_pre["quarter_note"][2])
    cv2.waitKey(1)
    cv2.destroyAllWindows() 


    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    # Two-layer deep learning
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    n_x = 50 * 80     
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    
    parameters = two_layer_model(train_X, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 4000, print_cost = True)   
    # Prediction
    pred = predict(test_X, test_y, parameters)
  
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    # N-layer deep learning
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    n_x = 50 * 80
    n_h1 = 20
    n_h2 = 7
    n_h3 = 5
    n_y = 1
    
    # 4-layer model 
    layers_dims = [50*80, 20, 7, 5, 1] 
    
    parameters = L_layer_model(train_X, train_y, layers_dims, num_iterations = 4000, print_cost = True)
    # Prediction
    pred = predict(test_X, test_y, parameters)
    
    
    