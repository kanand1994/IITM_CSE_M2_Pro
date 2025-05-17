import sys
import csv
import json
import re
import random
import folium
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from distance import Place, compute_distance_matrix
from tsp_solver import (
    greedy_tsp,
    two_opt_swap,
    simulated_annealing,
    christofides_tsp,
    AntColonyOptimizer,
    GeneticAlgorithm,
    calculate_total_distance
)

class TSPSession:
    def __init__(self):
        self.results: List[Dict] = []
    
    def add_result(self, result: Dict):
        self.results.append(result)
    
    def print_summary(self):
        print("\n=== Session Summary ===")
        if not self.results:
            print("No optimizations performed yet")
            return
        
        sorted_results = sorted(self.results, key=lambda x: x['distance'])
        
        print("\nAll Results:")
        for idx, res in enumerate(sorted_results, 1):
            print(f"{idx}. {res['city']} ({res['algorithm']}): {res['distance']:.2f} km")
        
        if len(sorted_results) >= 3:
            print("\nTop 3 Performers:")
            for idx, res in enumerate(sorted_results[:3], 1):
                print(f"{idx}. {res['algorithm']}: {res['distance']:.2f} km")
                print(f"   City: {res['city']}")
                print(f"   Start Point: {res['start_point']}")
                print(f"   Route File: {res['geojson']}")
                print(f"   Map File: {res['html']}")
        elif len(sorted_results) > 0:
            print("\nTop Performer:")
            res = sorted_results[0]
            print(f"1. {res['algorithm']}: {res['distance']:.2f} km")
            print(f"   City: {res['city']}")
            print(f"   Start Point: {res['start_point']}")
            print(f"   Route File: {res['geojson']}")
            print(f"   Map File: {res['html']}")

def main():
    session = TSPSession()
    
    while True:
        print("\n=== TSP City Tour Optimizer ===")
        print("1. Start New Optimization")
        print("2. Exit")
        choice = input("Choose option: ").strip()
        
        if choice == '2':
            session.print_summary()
            print("\nGoodbye!")
            break
            
        # Get cities CSV path
        cities_path = get_csv_path("Enter path to cities CSV file: ")
        cities = load_cities(cities_path)
        if not cities:
            continue
            
        # Select city
        city = select_city(cities)
        if not city:
            continue
            
        # Handle city places
        places = handle_city_places(city)
        if not places:
            continue
            
        # Select starting point
        start_point = select_start_point(places)
        if not start_point:
            continue
            
        # Select algorithm
        algorithm = select_algorithm()
        if not algorithm:
            continue
            
        # Run optimization
        result = run_optimization(city.name, places, start_point, algorithm)
        if result:
            session.add_result(result)
            print("\nOptimization Results:")
            print(f"Algorithm: {result['algorithm']}")
            print(f"Distance: {result['distance']:.2f} km")
            print(f"GeoJSON File: {result['geojson']}")
            print(f"Map File: {result['html']}")

def get_csv_path(prompt: str) -> Path:
    while True:
        try:
            path = Path(input("\n" + prompt).strip())
            if path.exists() and path.suffix == '.csv':
                return path
            print(f"Error: File not found or not a CSV file: {path}")
        except Exception as e:
            print(f"Invalid path: {str(e)}")

def load_cities(path: Path) -> List[Place]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return [
                Place(row[0].strip(), float(row[1]), float(row[2]))
                for row in reader if len(row) >= 3
            ]
    except Exception as e:
        print(f"Error loading cities: {str(e)}")
        return []

def select_city(cities: List[Place]) -> Optional[Place]:
    if not cities:
        print("No cities found in the provided CSV file")
        return None
    
    print("\nAvailable Cities:")
    for idx, city in enumerate(cities, 1):
        print(f"{idx}. {city.name}")
    
    while True:
        try:
            choice = int(input("\nSelect city (number): "))
            if 1 <= choice <= len(cities):
                return cities[choice-1]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def handle_city_places(city: Place) -> List[Place]:
    """Handle city places with validation and correction"""
    base_dir = Path("tourist_spots")
    city_file = base_dir / f"{sanitize_filename(city.name)}.csv"
    
    if not city_file.exists():
        print(f"\nCity file not found: {city_file}")
        print("Please create a CSV file with columns: Name,Lat,Lon")
        return None

    while True:
        places, invalid_indices = load_and_validate_places(city_file)
        
        if not invalid_indices:
            return places
            
        print(f"\nFound {len(invalid_indices)} invalid entries in {city_file.name}")
        print("Please correct the invalid coordinates:")
        places = correct_invalid_entries(city_file, places, invalid_indices)
        
        # Reload to verify corrections
        places, invalid_indices = load_and_validate_places(city_file)
        if not invalid_indices:
            return places
        print("Still invalid entries found after correction. Let's try again.")
def load_and_validate_places(path: Path) -> Tuple[List[Place], List[int]]:
    """Load places and validate coordinates"""
    places = []
    invalid_indices = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            
            for idx, row in enumerate(reader):
                if len(row) < 3:
                    invalid_indices.append(idx+1)
                    continue
                    
                try:
                    lat = float(row[1])
                    lon = float(row[2])
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        invalid_indices.append(idx+1)
                        places.append(Place(row[0], lat, lon))
                    else:
                        places.append(Place(row[0], lat, lon))
                except ValueError:
                    invalid_indices.append(idx+1)
                    places.append(Place(row[0], 0, 0))  # Temporary invalid coords

        return places, invalid_indices
    except Exception as e:
        print(f"Error loading places: {str(e)}")
        return [], []

def correct_invalid_entries(path: Path, places: List[Place], invalid_indices: List[int]) -> List[Place]:
    """Interactive correction of invalid entries"""
    corrected = []
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Lat', 'Lon'])
        
        for idx, place in enumerate(places):
            line_num = idx + 2  # Account for header row
            if line_num in invalid_indices:
                print(f"\nCorrecting entry {idx+1}: {place.name}")
                print(f"Current coordinates: {place.lat}, {place.lon}")
                
                new_lat = get_valid_coordinate("Enter new latitude (-90 to 90): ", -90, 90)
                new_lon = get_valid_coordinate("Enter new longitude (-180 to 180): ", -180, 180)
                
                writer.writerow([place.name, new_lat, new_lon])
                corrected.append(Place(place.name, new_lat, new_lon))
            else:
                writer.writerow([place.name, place.lat, place.lon])
                corrected.append(place)
    
    print(f"\nUpdated {len(invalid_indices)} entries in {path.name}")
    return corrected

def get_valid_coordinate(prompt: str, min_val: float, max_val: float) -> float:
    """Get validated coordinate input"""
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Value must be between {min_val} and {max_val}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def generate_landmarks(city: Place, num=50) -> List[Place]:
    places = [Place(f"{city.name} Center", city.lat, city.lon)]
    for i in range(1, num+1):
        places.append(Place(
            f"{city.name} Landmark {i}",
            city.lat + random.uniform(-0.2, 0.2),
            city.lon + random.uniform(-0.2, 0.2)
        ))
    return places

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w-]', '', name.strip().lower())

def save_places(path: Path, places: List[Place]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Lat', 'Lon'])
        for p in places:
            writer.writerow([p.name, p.lat, p.lon])

def load_places(path: Path) -> List[Place]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return [
                Place(row[0].strip(), float(row[1]), float(row[2]))
                for row in reader if len(row) >= 3
            ]
    except Exception as e:
        print(f"Error loading places: {str(e)}")
        return []

def select_start_point(places: List[Place]) -> Optional[Place]:
    print("\nAvailable Starting Points:")
    for idx, place in enumerate(places[:20], 1):
        print(f"{idx}. {place.name}")
    
    while True:
        try:
            choice = int(input("\nSelect starting point (number): "))
            if 1 <= choice <= len(places):
                return places[choice-1]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def select_algorithm() -> Optional[str]:
    print("\nAvailable Algorithms:")
    print("1. Greedy + 2-opt Optimization")
    print("2. Simulated Annealing")
    print("3. Christofides Algorithm")
    print("4. Ant Colony Optimization")
    print("5. Genetic Algorithm")
    print("6. Cancel")
    
    algo_map = {
        '1': "Greedy+2opt",
        '2': "SimAnnealing",
        '3': "Christofides",
        '4': "AntColony",
        '5': "GeneticAlgo"
    }
    
    while True:
        choice = input("Select algorithm: ").strip()
        if choice in algo_map:
            return algo_map[choice]
        if choice == '6':
            return None
        print("Invalid choice. Please select 1-6.")

def run_optimization(city_name: str, places: List[Place], 
                    start_point: Place, algorithm: str) -> Optional[Dict]:
    try:
        start_idx = next(i for i, p in enumerate(places) if p == start_point)
        
        print("\nComputing distance matrix...")
        dist_matrix = compute_distance_matrix(places)
        
        print(f"\nRunning {algorithm}...")
        if algorithm == "Greedy+2opt":
            tour = two_opt_swap(greedy_tsp(dist_matrix, start_idx), dist_matrix)
        elif algorithm == "SimAnnealing":
            tour = simulated_annealing(dist_matrix)
        elif algorithm == "Christofides":
            tour = christofides_tsp(dist_matrix)
        elif algorithm == "AntColony":
            tour = AntColonyOptimizer(dist_matrix).run()
        elif algorithm == "GeneticAlgo":
            tour = GeneticAlgorithm(dist_matrix).run()
        else:
            return None
        
        total_dist = calculate_total_distance(tour, dist_matrix)
        geojson_file = generate_geojson(city_name, places, tour, algorithm, dist_matrix)
        html_file = generate_map(geojson_file, places, algorithm, start_point)
        
        return {
            "city": city_name,
            "algorithm": algorithm,
            "distance": total_dist,
            "start_point": start_point.name,
            "geojson": geojson_file,
            "html": html_file
        }
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return None

def generate_geojson(city: str, places: List[Place], 
                    tour: List[int], algorithm: str,
                    dist_matrix: List[List[float]]) -> str:
    filename = f"{sanitize_filename(city)}_{algorithm}_route_{datetime.now().strftime('%Y%m%d%H%M')}.geojson"
    features = []
    
    for i in range(len(tour)):
        start_idx = tour[i]
        end_idx = tour[(i+1) % len(tour)]
        start_place = places[start_idx]
        end_place = places[end_idx]
        distance = dist_matrix[start_idx][end_idx]
        
        features.append({
            "type": "Feature",
            "properties": {
                "segment": i+1,
                "from": start_place.name,
                "to": end_place.name,
                "distance": f"{distance:.2f} km",
                "algorithm": algorithm
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [start_place.lon, start_place.lat],
                    [end_place.lon, end_place.lat]
                ]
            }
        })
    
    geojson = {"type": "FeatureCollection", "features": features}
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)
    
    return filename

def generate_map(geojson_path: str, places: List[Place], 
                algorithm: str, start_point: Place) -> str:
    html_path = geojson_path.replace('.geojson', '.html')
    
    m = folium.Map(location=[start_point.lat, start_point.lon], zoom_start=12)
    
    # Add markers
    for place in places:
        folium.Marker(
            location=[place.lat, place.lon],
            popup=place.name,
            icon=folium.Icon(
                color='green' if place == start_point else 'blue',
                icon='info-sign'
            )
        ).add_to(m)
    
    # Add route segments
    with open(geojson_path, 'r', encoding='utf-8') as f:
        route_data = json.load(f)
        total_distance = sum(float(f['properties']['distance'].split()[0]) 
                           for f in route_data['features'])
        
        for feature in route_data['features']:
            folium.GeoJson(
                feature,
                name=f"Segment {feature['properties']['segment']}",
                style_function=lambda x: {'color': '#FF0000', 'weight': 3},
                tooltip=folium.GeoJsonTooltip(
                    fields=['from', 'to', 'distance'],
                    aliases=['From:', 'To:', 'Distance:'],
                    localize=True
                )
            ).add_to(m)
    
    # Add controls and title
    folium.LayerControl().add_to(m)
    title_html = f'''
        <h3 align="center" style="font-size:16px">
            <b>{algorithm} Route - Total Distance: {total_distance:.2f} km</b>
        </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    m.save(html_path)
    
    return html_path

if __name__ == "__main__":
    main()