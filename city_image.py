import os
import math
import numpy as np
from PIL import  ImageOps
import matplotlib.pyplot as plt
import cv2


class CityImage:
    def __init__(self, image, lat, long, shape):
        img = image.resize(shape)
        self.img_grayscale = np.array(ImageOps.grayscale(img), dtype=np.float32) / 255.0  # Normalized
        self.grid_x, self.grid_y = None, None
        self.lat = lat
        self.long = long
        img.close()

    # Get the location of the image
    def get_loc(self):
        return self.lat, self.long

    # Distance formula with context of curvature of earth
    def haversine_distance(self, pred_lat: float, pred_long: float, in_meters=False):
        R = 6371 # Radius of the Earth in km
        dLat = math.radians(pred_lat - self.lat)
        dLon = math.radians(pred_long - self.long)
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(self.lat)) * math.cos(math.radians(pred_lat)) * math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c # Distance in km
        return distance * 1000 if in_meters else distance # Convert to meters if required

    # Get the image
    def get_image(self, noisy=False):
        return self.img_grayscale if not noisy else self.get_image_with_noise()
    
    # Get Image with added noise
    def get_image_with_noise(self, mean=0, var=0.01):
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, self.img_grayscale.shape)
        noisy_img = np.clip(self.img_grayscale + gaussian, 0, 1)  # Ensuring it's still between 0 and 1
        return noisy_img

    # # Display image
    # def show(self, noisy=False):
    #     plt.imshow(self.get_image(noisy=noisy), cmap='gray')
    #     plt.title(f"Location: {self.get_loc()}")
    #     plt.xticks([])  
    #     plt.yticks([])  
    #     plt.show()
    def show(self, noisy=False, heat_map=None):
            plt.figure()  # Add this line to create a new figure
            plt.imshow(self.get_image(noisy=noisy), cmap='gray', aspect='auto')

            if heat_map is not None:
                # Assuming heat_map is normalized between 0 and 1
                heat_map_resized = cv2.resize(heat_map, (self.img_grayscale.shape[1], self.img_grayscale.shape[0]))
                plt.imshow(heat_map_resized, cmap='jet', alpha=0.5, aspect='auto')  # Overlaying the heatmap

            plt.title(f"Location: {self.get_loc()}")
            plt.xticks([])  
            plt.yticks([])  
            plt.show()
            
    def set_grid_pos(self, gui):
        pixel_x, pixel_y = gui.long_lat_to_pixel(self.long, self.lat)

        self.grid_x = int(pixel_x // gui.square_amount[0])
        self.grid_y = int(pixel_y // gui.square_amount[1])



