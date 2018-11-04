from CNN_SMIRKs import *
import tensorflow as tf
from SMIRKs_head import *
import os
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
def create_folder(directory):
	try:
		os.mkdir(directory)
	except OSError:
		print('Error')

if __name__ == "__main__":
    img_name = 'music4'
    img_name_ = img_name + '.png'

    path = '/Users/jinzhao/Desktop/Robotics/DIP/'
    folder_path = path + img_name # img_name
    create_folder(folder_path)
    # Read image
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    img = cv2.imread(img_name_)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bw = convert(img_bw)
    height, width = img_bw.shape
    
    '''
    cv2.imshow('test', img_bw)
    cv2.waitKey(1)
    cv2.destroyAllWindows() 
    '''
    
    # Preprocess image
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    dash_filter = np.ones([1, 2])  
    staff_line_filter = np.ones([1, width//10]) # These two filters can also be applied to partitioned images

    # Partition the music sheet based on staff lines
    img_staff, idx_staff = get_staff_lines(img_bw, dash_filter, staff_line_filter)
    img_div_set = partition_image(img_bw, idx_staff)
    
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    # Test!!!
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    '''
    # Show!
    for i in range(len(img_div_set)):    
        cv2.imshow('test'+str(i), img_div_set[i])
    
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    img_div = img_div_set[0]
    test_filter_size(img_div, diff_staff)
    
    '''
    # Determine a reasonable window's size
    # ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    diff_staff = (idx_staff[4] - idx_staff[0]) // 4
    disk_filter = generate_disk_filter(diff_staff//2.5)

    file_path = folder_path + '/note position.txt'
    f = open(file_path, 'w')
    count = 0 # count the image
    for img_div in img_div_set:
	    img_notes = opening(img_div, disk_filter) 
	    im, contours, hierarchy = cv2.findContours(img_notes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	    moments = compute_moments(contours)
	    interval = find_note_interval(moments)

	    div_staff_lines, div_idx_staff = get_staff_lines(img_div, dash_filter, staff_line_filter)

	    note_pos = []

	    for i in range(len(moments)):
	    	# Determine the position
	        row_idx = moments[i, 0]
	        output = determine_note_pos(div_idx_staff, row_idx)
	        note_pos.append(output)

	        # Extract note
	        col_idx = moments[i, 1]
	        start = int(col_idx - interval/2)
	        end = int(col_idx + interval/2)
	        im_note = img_div[:, start:end]

	        filename = 'note' + str(count) + '.jpg' 
	        cv2.imwrite(os.path.join(folder_path, filename), im_note)
	        count += 1
	        
	    for pos in note_pos:
             f.write(str(pos) + '\n')
    f.close()
    









	    