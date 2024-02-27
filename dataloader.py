import xml.etree.ElementTree as ET
import os
import time
import ferryloader
import busloader

namespace = {'netex': 'http://www.netex.org.uk/netex'}

def get_data_from_folder(folder):
	errors = []
	data = []
	for file in os.listdir(folder):
#		print("File: ", file)
#		print("Is dir: ", os.path.isdir(folder + "/" + file))
		if os.path.isdir(folder + "/" + file):
			print("Starting folder: ", file)
			all_data, all_errors = get_data_from_folder(folder + "/" + file)
			data += all_data
			errors += all_errors
#			print("Ending folder: ", file)
		elif file.endswith(".xml"):
			print("  File: ", file)
			dataFromFile, error = get_data_from_file(folder + "/" + file)
			print("    Data: ", dataFromFile.__len__() if dataFromFile else 0)
			print("    Error: ", error)
			if error:
				errors.append(error)
			else:
				data.append(dataFromFile)
	print("DataLen: ", data.__len__())
	print("ErrorsLen: ", errors.__len__())
	return data, errors

def get_data_from_file(file):
	tree = ET.parse(file)
	root = tree.getroot()
	lines = root.find(".//netex:lines", namespace)
	transportMode = lines.find(".//netex:TransportMode", namespace).text
#	print("File: ", file)
	print("    Transport mode: ", transportMode)
	if transportMode == "bus":
		return busloader.get_data_from_file(file)
	elif transportMode == "water":
		return ferryloader.get_data_from_file(file)
	else:
		return busloader.get_data_from_file(file)

time_start = time.time()
all_data, errors = get_data_from_folder("Data")
print("Len", all_data.__len__())
print("Errors: ", errors.__len__())
print("Time: ", time.time() - time_start)
