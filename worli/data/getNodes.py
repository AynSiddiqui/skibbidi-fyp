import xml.etree.ElementTree as ET
from collections import defaultdict

def generate_secondary_lane_continuity_map(xml_file):
    # Parse the SUMO network XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Dictionary to store the continuity map
    secondary_lane_continuity_map = defaultdict(list)

    # Create a mapping of edges to lanes
    edge_to_lanes = {}
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        edge_type = edge.get('type', '').lower()

        # Skip edges that are not secondary
        if "secondary" not in edge_type:
            continue

        # Collect lanes for the secondary edge
        edge_to_lanes[edge_id] = [lane.get('id') for lane in edge.findall('lane')]

    # Iterate through connections to trace lane continuity
    for connection in root.findall('connection'):
        from_edge = connection.get('from')
        to_edge = connection.get('to')
        from_lane_index = connection.get('fromLane')
        to_lane_index = connection.get('toLane')

        # Skip connections that don't involve secondary edges
        if from_edge not in edge_to_lanes or to_edge not in edge_to_lanes:
            continue

        # Map the from_lane to the to_lane
        from_lane = edge_to_lanes[from_edge][int(from_lane_index)]
        to_lane = edge_to_lanes[to_edge][int(to_lane_index)]
        secondary_lane_continuity_map[from_lane].append(to_lane)

    return secondary_lane_continuity_map

# Path to your SUMO .net.xml file
xml_file = 'worli.net.xml'

# Generate the secondary lane continuity map
secondary_lane_continuity_map = generate_secondary_lane_continuity_map(xml_file)

output_mapping = {item: key for key, values in secondary_lane_continuity_map.items() for item in values}

# Print the result
print(output_mapping)
