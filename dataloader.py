import xml.etree.ElementTree as ET
import os
import time
import ferryloader
import busloader

namespace = {'netex': 'http://www.netex.org.uk/netex'}

def get_data_from_folder(folder):
	data = []
	for file in os.listdir(folder):
#		print("File: ", file)
#		print("Is dir: ", os.path.isdir(folder + "/" + file))
		if os.path.isdir(folder + "/" + file):
#			print("Starting folder: ", file)
			data.append(get_data_from_folder(folder + "/" + file))
#			print("Ending folder: ", file)
		elif file.endswith(".xml"):
			data.append(get_data_from_file(folder + "/" + file))
	return data

def get_data_from_file(file):
	tree = ET.parse(file)
	root = tree.getroot()
	lines = root.find(".//netex:lines", namespace)
	transportMode = lines.find(".//netex:TransportMode", namespace).text
	print("File: ", file)
	print("Transport mode: ", transportMode)
	if transportMode == "bus":
		return busloader.get_data_from_file(file)
	elif transportMode == "water":
		return ferryloader.get_data_from_file(file)
	else:
		return busloader.get_data_from_file(file)

time_start = time.time()

all_data = get_data_from_folder("Data")
print(all_data.__len__())
print("Time: ", time.time() - time_start)
