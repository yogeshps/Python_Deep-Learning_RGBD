from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import csv

directory_pre = "/home/astar/Desktop/for.yogesh/depth_"
directory_post = "/home/astar/Desktop/smaller_images/smalldepth_"
img_type = ".jpg"
img_type_saved = ".bmp"

for i in range(1,2747):
    img_path = directory_pre + str(i) + img_type
    img_path_saved = directory_post + str(i) + img_type
    img = Image.open(img_path)
    print (img.size)
    img = img.resize((160, 60), Image.ANTIALIAS)
    print (img.size)
    img.save(img_path_saved)
    img.save(img_path_saved, optimize=True)