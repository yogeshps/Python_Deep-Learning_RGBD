from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import csv

directory_pre = "/home/astar/Desktop/DL_Stuff/Small_Images_Joy/small_depth_43_04/smalldepth_"

directory_left = "/home/astar/Desktop/DL_Stuff/Small_Images_Joy/label_43/LEFT/left_"
directory_right = "/home/astar/Desktop/DL_Stuff/Small_Images_Joy/label_43/RIGHT/right_"
directory_forward = "/home/astar/Desktop/DL_Stuff/Small_Images_Joy/label_43/FORWARD/forward_"
directory_stop = "/home/astar/Desktop/DL_Stuff/Small_Images_Joy/label_43/STOP/stop_"

img_type = ".jpg"
img_type_orig = ".jpg"

left = 1
forward = 1
right = 1
stop = 1

with open('/home/astar/Desktop/DL_Stuff/Small_Images_Joy/joystick_files/joy_43.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
    print (your_list)
    print (len(your_list))

for i in range(1,944):
    img_path = directory_pre + str(i) + img_type_orig
    img = Image.open(img_path)
    print("image opened" + str(i))

    if (float(your_list[i][0]) > 0.0) and abs(float(your_list[i][0])) > abs(float(your_list[i][4])): #Robot is turning left
        img_path_save = directory_left + str(125+87+88+80+76+left) + img_type
        img.save(img_path_save)
        img.save(img_path_save, optimize=True)
        left += 1

    elif (float(your_list[i][0]) < 0.0) and abs(float(your_list[i][0])) > abs(float(your_list[i][4])):  # Robot is turning right
        img_path_save = directory_right + str(105+164+84+124+83+right) + img_type
        img.save(img_path_save)
        img.save(img_path_save, optimize=True)
        right += 1

    elif (float(your_list[i][4]) > 0.0) and abs(float(your_list[i][0])) <= abs(float(your_list[i][4])):  # Robot is moving forward
        img_path_save = directory_forward + str(733+697+751+695+710+forward) + img_type
        img.save(img_path_save)
        img.save(img_path_save, optimize=True)
        forward += 1

    else:
        img_path_save = directory_stop + str(88+69+69+49+83+stop) + img_type
        img.save(img_path_save)
        img.save(img_path_save, optimize=True)
        stop += 1




