from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import csv

directory_pre = "/home/astar/Desktop/smallish_images/smalldepth_"

directory_left = "/home/astar/Desktop/DL_Images_320/LEFT/left_"
directory_right = "/home/astar/Desktop/DL_Images_320/RIGHT/right_"
directory_forward = "/home/astar/Desktop/DL_Images_320/FORWARD/forward_"
directory_stop = "/home/astar/Desktop/DL_Images_320/STOP/stop_"

img_type = ".jpg"
img_type_orig = ".jpg"

left = 1
forward = 1
right = 1
stop = 1

with open('/home/astar/Desktop/smallish_images/Joystick.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
    print (your_list)
    print (len(your_list))

for i in range(1,2746):
    img_path = directory_pre + str(i) + img_type_orig
    img = Image.open(img_path)
    print("image opened" + str(i))

    if (float(your_list[i][0])) > 0.0: #Robot is turning left
        img_path_save = directory_left + str(left) + img_type
        img.save(img_path_save)
        img.save(img_path_save, optimize=True)
        left += 1

    elif (float(your_list[i][0])) < 0.0:  # Robot is turning right
        img_path_save = directory_right + str(right) + img_type
        img.save(img_path_save)
        img.save(img_path_save, optimize=True)
        right += 1

    elif (float(your_list[i][4])) > 0.0:  # Robot is moving forward
        img_path_save = directory_forward + str(forward) + img_type
        img.save(img_path_save)
        img.save(img_path_save, optimize=True)
        forward += 1

    else:
        img_path_save = directory_stop + str(stop) + img_type
        img.save(img_path_save)
        img.save(img_path_save, optimize=True)
        stop += 1




