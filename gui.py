import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import numpy as np

class GUI: # class for image
    def __init__(self):
        self.image_loc = 'scaled_america_mex_can.png' 
        im = Image.open(self.image_loc) # opens image
        self.width, self.height = im.size
        im.close()

        self.extreme_points = {'N': 50, 'S': 24, 'E': -66, 'W': -126} # assigns extreme points
        self.total_lat = self.extreme_points['N'] - self.extreme_points['S']
        self.total_long = abs(self.extreme_points['W'] - self.extreme_points['E'])
        self.show_ticks = True
        self.dots = []

    def pixel_loc_to_lat_long(self, w, h):
        return (round(self.extreme_points['N'] - h/self.height * (self.total_lat), 2), round(self.extreme_points['W'] - w/self.width * (self.total_long), 2))

    def long_lat_to_pixel(self, long, lat):
        # Adjust the calculation for longitude and latitude relative position
        long_rel = (long - self.extreme_points['W']) / self.total_long
        lat_rel = (self.extreme_points['N'] - lat) / self.total_lat

        x_pixel = int(long_rel * self.width)
        y_pixel = int(lat_rel * self.height)

        return (x_pixel, y_pixel)


    def init(self): 
        self.x_ticks = np.arange(0, self.width, 100)
        self.y_ticks = np.arange(0, self.height, 100)

        self.x_labels = [f"{round(self.extreme_points['W'] + i/self.width * self.total_long, 2)}" for i in self.x_ticks]
        self.y_labels = [f"{round(self.extreme_points['N'] - i/self.height * self.total_lat, 2)}" for i in self.y_ticks]


    def place_dot(self, long, lat, color=None, r=10):
        x_pixel, y_pixel = self.long_lat_to_pixel(long, lat)
        circle = plt.Circle((x_pixel, y_pixel), r, color=(color if color else 'blue'), fill=True)
        self.dots.append(circle) # store the dot object

    def clear_dots(self):
        for dot in self.dots:
            dot.remove()            # Remove the dot from the plot
        self.dots.clear()           # Clear the list of stored dots

    def toggle_ticks(self, show_ticks):
        self.show_ticks = show_ticks
    
    def show(self): 
        _, map = plt.subplots()
        map.imshow(Image.open(self.image_loc))

        if self.show_ticks:
            map.set_xticks(self.x_ticks)
            map.set_yticks(self.y_ticks)

            map.set_xticklabels(self.x_labels, rotation=90)
            map.set_yticklabels(self.y_labels)
        else:
            map.axis('off')

        map.set_xlabel("Longitude (West)")
        map.set_ylabel("Latitude (North)")
        
        for dot in self.dots:
            map.add_patch(dot)

        plt.tight_layout()
        plt.show()


