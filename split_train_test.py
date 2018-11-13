from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import csv

#directory_pre = "/home/astar/Desktop/smaller_images/smalldepth_"

directory_left = "/home/astar/Desktop/DL_Images/LEFT/left_"
directory_right = "/home/astar/Desktop/DL_Images/RIGHT/right_"
directory_forward = "/home/astar/Desktop/DL_Images/FORWARD/forward_"
directory_stop = "/home/astar/Desktop/DL_Images/STOP/stop_"

#directory_train = "/home/astar/Desktop/DL_Images/training_set/"

img_type = ".jpg"
img_type_orig = ".jpg"

left = 1
forward = 1
right = 1
stop = 1

def splitter(classname, picname, size):

    directory_pre = "/home/astar/Desktop/DL_Images_320/"
    directory_train = "/home/astar/Desktop/DL_Images_New/training_set/"
    directory_test = "/home/astar/Desktop/DL_Images_New/test_set/"

    j = 1
    k = 1

    """with open('/home/astar/Desktop/smallish_images/Joystick.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        print (your_list)
        print (len(your_list))"""

    for i in range(1,size):
        img_path = directory_pre + classname + "/" + picname + "_" + str(i) + img_type_orig
        img = Image.open(img_path)
        print("image opened" + str(i))

        if (i%5==0):
            img_path_save = directory_train + classname + "/" + picname + "_" + str(j) + img_type
            img.save(img_path_save)
            img.save(img_path_save, optimize=True)
            j += 1

        else:
            img_path_save = directory_test + classname + "/" + picname + "_" + str(k) + img_type
            img.save(img_path_save)
            img.save(img_path_save, optimize=True)
            k += 1


splitter("FORWARD", "forward", 1501)
splitter("LEFT", "left", 392)
splitter("RIGHT", "right", 12)
splitter("STOP", "stop", 844)





