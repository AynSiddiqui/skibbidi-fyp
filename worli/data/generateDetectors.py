import xml.etree.ElementTree as ET

def generate_detectors(net_file, output_file):
    tree = ET.parse(net_file)
    root = tree.getroot()

    additional = ET.Element("additional")
    detector_id = 0

    for edge in root.findall("edge"):
        if edge.attrib.get("function") == "internal":
            continue  # Skip internal edges
        for lane in edge.findall("lane"):
            lane_id = lane.attrib["id"]
            length = float(lane.attrib["length"])

            # Create a lane area detector
            lane_detector = ET.SubElement(additional, "laneAreaDetector")
            lane_detector.set("id", lane_id)
            lane_detector.set("lane", lane_id)
            lane_detector.set("pos", str(length / 2))  # Place in the middle of the lane
            lane_detector.set("file", "ild.out")
            lane_detector.set("freq", "1")

    # Write to output file
    tree = ET.ElementTree(additional)
    with open(output_file, "wb") as f:
        tree.write(f)

# Example usage
generate_detectors("worli.net.xml", "worli.add.xml")
