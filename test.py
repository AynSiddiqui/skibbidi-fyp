import os
import xml.etree.ElementTree as ET

def collect_tripinfo(episode):
    all_trip_data = []

    trip_file = os.path.join("./worli/data/worli_rr_trip.xml")
    tree = ET.ElementTree(file=trip_file)

    for child in tree.getroot():
        cur_trip = child.attrib

        cur_dict = {
            'episode': episode,
            'id': cur_trip['id'],
            'depart_sec': cur_trip['depart'],
            'arrival_sec': cur_trip['arrival'],
            'duration_sec': cur_trip['duration'],
            'wait_step': cur_trip['waitingCount'],
            'wait_sec': cur_trip['waitingTime']
        }

        all_trip_data.append(cur_dict)

    return all_trip_data

print(collect_tripinfo(1))