import numpy as np
import requests
from io import BytesIO
from PIL import Image
from city_image import CityImage
from gui import GUI
from agent import Agent
import os
import time

SHAPE = (300, 300)
agent_path = "Models/agent_model.keras"

# Save a backup of the agent model and name it with a unique number based on the current time
backup_agent_path = f"Models/Backups/agent_model_{int(time.time())}.keras"

class Game:
    def __init__(self) -> None:
        self.key = os.getenv("MY_API_KEY")
        self.gui = GUI()
        self.gui.init()
        self.extreme_points = self.gui.extreme_points
        self.agent = Agent(SHAPE, self.extreme_points, path=agent_path, backup_path=backup_agent_path)

    def normalize_coordinates(self, long, lat):
        normalized_long = (long - self.extreme_points['W']) / (self.extreme_points['E'] - self.extreme_points['W'])
        normalized_lat = (lat - self.extreme_points['S']) / (self.extreme_points['N'] - self.extreme_points['S'])
        return np.array([normalized_long, normalized_lat])

    def denormalize_coordinates(self, normalized_coords):
        denormalized_long = normalized_coords[0] * (self.extreme_points['E'] - self.extreme_points['W']) + self.extreme_points['W']
        denormalized_lat = normalized_coords[1] * (self.extreme_points['N'] - self.extreme_points['S']) + self.extreme_points['S']
        return np.array([denormalized_long, denormalized_lat])

    def get_round(self, show=True, verbose=False, learn=True, limit=100, heat=False):
        image, lat, long = self.generate_image(verbose=verbose, limit=limit)
        if image is None:  # Adding a check in case the image is not successfully fetched
            return
        
        # Normalize correct coordinates
        correct_coords_normalized = self.normalize_coordinates(long, lat)
        
        city_img = CityImage(image, lat, long, SHAPE)
        if show:
            print(city_img.get_loc())
            self.gui.clear_dots()
            self.gui.place_dot(long, lat, color='red')  # Place the red dot at the correct location
            
            agent_answer, heat_map = self.agent.predict(city_img, get_heat_map=heat, layer_name='conv2d_2')
            agent_answer_denormalized = self.denormalize_coordinates(agent_answer[0])  

            if heat:
                city_img.show(heat_map=None) # Show the original image without the heatmap
            city_img.show(heat_map=heat_map)
            
            # Place the black dot at the agent's predicted location
            self.gui.place_dot(agent_answer_denormalized[0], agent_answer_denormalized[1], color='black')
            
            print(f"Correct [Norm]: {correct_coords_normalized} | Agent [Norm]: {agent_answer[0]}")
            print(f"Correct [Denorm]: {[long, lat]} | Agent [Denorm]: {agent_answer_denormalized}")
            self.gui.show()
        if learn:
            self.agent.deep_learn(city_img, correct_coords_normalized)

        
    def play_rounds(self, rounds, show=True, verbose=True, learn=True, limit=100, save_after=10, show_after=-1, heat=False):
        for i in range(rounds):
            print(f"Round {i + 1}/{rounds}")
            show_this_round = show or ((i + 1) % show_after == 0)
            if not show and show_after == -1:
                show_this_round = False
            self.get_round(show=show_this_round, verbose=verbose, learn=learn, limit=limit, heat=heat)
            if (i + 1) % save_after == 0:
                self.agent.save_model()
        self.agent.save_model()

    def generate_random_location(self, verbose=True, limit=100):
        MAX_TRIES = limit
        for i in range(MAX_TRIES):  # Limit the number of attempts
            if verbose:
                print(f"{i}/{MAX_TRIES}", end="\r")
            lat = np.random.uniform(low=self.extreme_points['S'], high=self.extreme_points['N'])
            lng = np.random.uniform(low=self.extreme_points['W'], high=self.extreme_points['E'])

            # Check Street View availability
            street_view_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&key={self.key}"
            response = requests.get(street_view_url).json()

            if response['status'] == 'OK':  # Street View is available
                return lat, lng
        return None  


    def fetch_street_view_image(self, lat, lng, size='600x400', fov=90, heading=None, pitch=0):
        if lat is None or lng is None:
            print("No valid location found with Street View availability.")
            return None
        
        base_url = "https://maps.googleapis.com/maps/api/streetview?"
        if heading is None:
            heading = np.random.randint(0, 360)
        url = f"{base_url}size={size}&location={lat},{lng}&fov={fov}&heading={heading}&pitch={pitch}&key={self.key}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            print(f"Error fetching Street View image: {response.status_code}")
            return None

    def generate_image(self, verbose=True, limit=100):
        coordinates = self.generate_random_location(verbose=verbose, limit=limit)
        if coordinates is None:
            if verbose:
                print("Failed to find a location with Street View availability after multiple attempts.")
            return None, None, None
        lat, lng = coordinates
        if lat is None or lng is None:
            if verbose:
                print("Failed to find a location with Street View availability after multiple attempts. Trying again.")
            return self.generate_image()

        image = self.fetch_street_view_image(lat, lng)
        if image:
            print("Successfully fetched Street View image.", end="\r")
            return image, lat, lng
        else:
            if verbose:
                print("Failed to fetch Street View image.")
            return None, None, None
