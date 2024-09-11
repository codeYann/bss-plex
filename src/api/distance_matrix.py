import googlemaps as maps
import os
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("GOOGLE_KEY")
gmaps = maps.Client()

