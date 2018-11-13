from keras.preprocessing.image import img_to_array, load_img
import csv

directory = "/home/astar/Desktop/for.yogesh/depth_"
img_type = ".jpg"


with open('image_arrays.csv', mode='w') as image_array:
    flat_list = [None]*60
    """for i in range(1,2):
        img_path = directory + str(i) + img_type
        img = load_img(img_path)
        x = img_to_array(img)
        for sublist in x:
            for item in sublist:
                flat_list.append(item)
        image_writer = csv.writer(image_array, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_writer.writerow([x])"""

print (flat_list)
