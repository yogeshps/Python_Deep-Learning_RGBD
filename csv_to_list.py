import csv

with open('/home/astar/Desktop/smaller_images/Joystick.csv', 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)

print(float(your_list[0][1]))
# [['This is the first line', 'Line1'],
#  ['This is the second line', 'Line2'],
#  ['This is the third line', 'Line3']]