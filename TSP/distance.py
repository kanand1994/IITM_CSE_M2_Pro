import math
from collections import namedtuple
from typing import List

Place = namedtuple('Place', ['name', 'lat', 'lon'])

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between coordinates using Haversine formula"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def compute_distance_matrix(places: List[Place]) -> List[List[float]]:
    """Compute N x N distance matrix"""
    return [[haversine(p1.lat, p1.lon, p2.lat, p2.lon) if i != j else 0 
            for j, p2 in enumerate(places)] 
            for i, p1 in enumerate(places)]