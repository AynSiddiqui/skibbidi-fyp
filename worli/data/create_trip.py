import requests
import openpyxl
import xml.etree.ElementTree as ET
import pandas as pd
import json
import os
# import subprocess
import re
from fuzzywuzzy import process
import random
import sumolib
import heapq
from dotenv import load_dotenv

load_dotenv()

random.seed(42)
# CONSTANTS TO EDIT
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

REQUEST_MODE = False

BBOX_NORTH = 19.0066500  # maxlat
BBOX_SOUTH = 18.9970400  # minlat
BBOX_EAST = 72.8230300   # maxlon
BBOX_WEST = 72.8123800   # minlon


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

OSM_FILE = os.path.join(BASE_DIR, "worli.osm")
NET_FILE = os.path.join(BASE_DIR, "worli.net.xml")
ROU_FILE = os.path.join(BASE_DIR, "worli.rou.xml")

TRAFFIC_JSON = os.path.join(BASE_DIR, "traffic_data.json")
TRAFFIC_CSV = os.path.join(BASE_DIR, "traffic_data.csv")
EDGE_NAMES_CSV = os.path.join(BASE_DIR, "edgename_data.csv")
EDGE_WEIGHTS_CSV = os.path.join(BASE_DIR, "edge_weights.csv")

def get_valid_routes(net, start_edge, min_depth=5, max_depth=10):
    def dfs(current_edge, path, visited, depth):
        if depth >= min_depth and random.random() > 0.5:
            paths.append(path[:])
        
        if depth >= max_depth:
            return
        
        for next_edge in current_edge.getOutgoing():
            next_edge_id = next_edge.getID()
            if next_edge_id not in visited:
                visited.add(next_edge_id)
                path.append(next_edge_id)
                dfs(next_edge, path, visited, depth + 1)
                path.pop()
                visited.remove(next_edge_id)
    
    start_edge_obj = net.getEdge(start_edge)
    paths = []
    dfs(start_edge_obj, [start_edge], {start_edge}, 0)
    return paths


def get_bbox_traffic_data(api_url, api_key):
    payload = {
        "in": {
            "type": "bbox",
            "west": BBOX_WEST,
            "south": BBOX_SOUTH,
            "east": BBOX_EAST,
            "north": BBOX_NORTH
        },
        "locationReferencing": ["none"]
    }

    try:
        response = requests.post(
            API_URL, json=payload, params={"apiKey": API_KEY})
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print("Error fetching data from API: {}".format(e))
        return None


def extract_traffic_data(data):
    records = []
    for result in data.get("results", []):
        location = result.get("location", {}).get("description", "Unknown")
        jam_factor = result.get("currentFlow", {}).get("jamFactor", 0)
        records.append((location, jam_factor, jam_factor * 10))

    return pd.DataFrame(records, columns=["Location", "Jam Factor", "Num_Vehicles"])


def extract_ids_from_xml():
    tree = ET.parse(NET_FILE)
    root = tree.getroot()

    name_dict = {}

    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        name = edge.get('name')
        if name:
            if name not in name_dict:
                name_dict[name] = [edge_id]
            else:
                name_dict[name].append(edge_id)

    return pd.DataFrame([(name, edge_ids) for name, edge_ids in name_dict.items()], columns=["Location", "Edge_IDs"])


def generate_edge_weights(df):
    df = df.groupby("Location", as_index=False).agg({
        "Num_Vehicles": "sum",
        "Edge_IDs": "first",
    })

    df["Edge_Weight"] = df["Num_Vehicles"] / df["Edge_IDs"].apply(len)

    df = df.explode("Edge_IDs")
    df = df.rename(columns={"Edge_IDs": "Edge_ID"})

    df = df[["Edge_ID", "Edge_Weight"]]

    return df

def create_src_xml(df, output_prefix):
    src_file = "{}.src.xml".format(output_prefix)
    with open(src_file, "w") as f:
        f.write('<edgedata>\n')
        f.write('    <interval id="1" begin="0" end="3600">\n')
        for _, row in df.iterrows():
            edge_id = row["Edge_ID"]
            weight = int(row["Edge_Weight"])
            f.write('        <edge id="{}" value="{}"/>\n'.format(edge_id, weight))
        f.write('    </interval>\n')
        f.write('</edgedata>\n')
    return src_file

def merge_dataframes(edgename_df, traffic_df, on_column):
    words_xl1 = edgename_df[on_column].tolist()
    words_xl2 = traffic_df[on_column].tolist()

    # Preprocessing function
    def preprocess(text):
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower().strip()          # Convert to lowercase and strip spaces
        text = re.sub(r'\s+', '', text)      # Remove all spaces
        return text

    # Preprocess edge names and traffic data
    processed_xl1 = [preprocess(word) for word in words_xl1]
    processed_xl2 = [preprocess(word) for word in words_xl2]

    # Split traffic data words by '/'
    split_processed_xl2 = []
    split_original_xl2 = []

    for original_word, processed_word in zip(words_xl2, processed_xl2):
        split_words = processed_word.split('/')
        split_processed_xl2.extend(split_words)
        split_original_xl2.extend([original_word] * len(split_words))  # Preserve original names

    # Perform fuzzy matching for each split word in traffic data against edge names
    results = []
    threshold = 80  # Matching threshold

    for original_word, processed_word in zip(split_original_xl2, split_processed_xl2):
        top_match = process.extractOne(processed_word, processed_xl1)
        
        if top_match and top_match[1] >= threshold:
            match_name = words_xl1[processed_xl1.index(top_match[0])]
            edge_id = edgename_df.loc[edgename_df[on_column] == match_name, 'Edge_IDs'].values[0]
        else:
            match_name = "No strong match"
            edge_id = ""
        
        # Get jam factor and num vehicles
        row_data = traffic_df[traffic_df['Location'] == original_word]
        jam_factor = row_data['Jam Factor'].values[0] if not row_data.empty else ""
        num_vehicles = row_data['Num_Vehicles'].values[0] if not row_data.empty else ""
        
        if edge_id:
            results.append({
                'Query_Word': original_word,
                'Location': match_name,
                'Edge_IDs': edge_id,
                'Jam Factor': jam_factor,
                'Num_Vehicles': num_vehicles
            })

    fuzzy_results_df = pd.DataFrame(results)

    return fuzzy_results_df

def generate_routes(df):
    net = sumolib.net.readNet(NET_FILE)
    edges = list(net.getEdges())
    all_edges = [edge.getID() for edge in edges]
    num_edges = df.shape[0]
    inc = 1

    routes = []
    
    for index, row in df.iterrows():
        edge_id = row["Edge_ID"]
        num_vehicles = int(row["Edge_Weight"])

        valid_routes = get_valid_routes(net, edge_id)

        if not valid_routes:
            inc += 1
            continue
        
        for i in range(num_vehicles):
            selected_route = random.choice(valid_routes)
            routes.append((i, edge_id, selected_route))
        
        inc += 1

    with open(ROU_FILE, "w") as f:
        f.write("<routes>\n")
        f.write('   <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="80"/>\n')
        
        count = 0
        for route in routes:
            route_id = "route_{}_{}".format(route[0], route[1])
            f.write('   <route id="{}" edges="{}"/>\n'.format(route_id, " ".join(route[2])))
            f.write('   <vehicle id="{}_veh{}" route="{}" type="car" depart="{}"/>\n'.format(route_id, count, route_id, count))
            count += 1

        f.write("</routes>\n")

def sort_routes_file():
    tree = ET.parse(ROU_FILE)
    root = tree.getroot()
    trips = sorted(root.findall("trip"), key=lambda x: float(x.get("depart")))
    
    for trip in root.findall("trip"):
        root.remove(trip)
    
    for trip in trips:
        root.append(trip)
    
    tree.write(ROU_FILE)

def main():
    if REQUEST_MODE:
        json_data = get_bbox_traffic_data(API_URL, API_KEY)
        with open(TRAFFIC_JSON, "w") as file:
            json.dump(json_data, file)
    else:
        with open(TRAFFIC_JSON) as file:
            json_data = json.load(file)

    traffic_data = extract_traffic_data(json_data)
    traffic_df = traffic_data[traffic_data["Jam Factor"] > 0]
    traffic_df.to_csv(TRAFFIC_CSV, index=False)

    edgename_df = extract_ids_from_xml()
    edgename_df.to_csv(EDGE_NAMES_CSV, index=False)

    inner_join = merge_dataframes(edgename_df=edgename_df, traffic_df=traffic_df, on_column="Location")

    edge_weights_df = generate_edge_weights(inner_join)
    edge_weights_df.to_csv(EDGE_WEIGHTS_CSV, index=False)

    generate_routes(edge_weights_df)
    sort_routes_file()

# MAIN SCRIPT
if __name__ == "__main__":
    main()