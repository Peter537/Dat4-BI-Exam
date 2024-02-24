import xml.etree.ElementTree as ET

namespace = {'netex': 'http://www.netex.org.uk/netex'}

def get_data_from_file(file):
	tree = ET.parse(file)
	root = tree.getroot()
	data = {
		"line": {
			"name": "",
			"shortName": "",
		},
		"stopPlaces": [],
		"routes": []
	}

	try:
		dataObjectsTree = root.find(".//netex:dataObjects", namespace)
		stopPlacesTree = dataObjectsTree.find(".//netex:stopPlaces", namespace)
		for stopPlace in stopPlacesTree.findall(".//netex:StopPlace", namespace):
			stopPlaceObj = {
				"id": stopPlace.attrib['id'],
				"name": stopPlace.find(".//netex:Name", namespace).text,
				"latitude": stopPlace.find(".//netex:Latitude", namespace).text,
				"longitude": stopPlace.find(".//netex:Longitude", namespace).text
			}
			data["stopPlaces"].append(stopPlaceObj)

		scheduledStopPoints = []
		scheduledStopPointsTree = dataObjectsTree.find(".//netex:scheduledStopPoints", namespace)
		for scheduledStopPoint in scheduledStopPointsTree.findall(".//netex:ScheduledStopPoint", namespace):
			scheduledStopPointObj = {
				"id": scheduledStopPoint.attrib['id'],
				"name": scheduledStopPoint.find(".//netex:Name", namespace).text,
			}
			scheduledStopPoints.append(scheduledStopPointObj)

		serviceLinks = []
		serviceLinksTree = dataObjectsTree.find(".//netex:serviceLinks", namespace)
		for serviceLink in serviceLinksTree.findall(".//netex:ServiceLink", namespace):
			serviceLinkObj = {
				"id": serviceLink.attrib['id'],
				"from": serviceLink.find(".//netex:FromPointRef", namespace).attrib['ref'],
				"to": serviceLink.find(".//netex:ToPointRef", namespace).attrib['ref'],
			}
			serviceLinks.append(serviceLinkObj)

		for serviceLink in serviceLinks:
			for scheduledStopPoint in scheduledStopPoints:
				if serviceLink['from'] == scheduledStopPoint['id']:
					fromPointS = scheduledStopPoint
				if serviceLink['to'] == scheduledStopPoint['id']:
					toPointS = scheduledStopPoint
			for stopPlace in data["stopPlaces"]:
				if fromPointS['name'] in stopPlace['name']:
					fromPoint = stopPlace
				if toPointS['name'] in stopPlace['name']:
					toPoint = stopPlace
			routeObj = {
				"from": fromPoint['id'],
				"to": toPoint['id'],
			}
			data["routes"].append(routeObj)

		# Get line name
		linesTree = dataObjectsTree.find(".//netex:lines", namespace)
		for line in linesTree.findall(".//netex:Line", namespace):
			data["line"]["name"] = line.find(".//netex:Name", namespace).text
			data["line"]["shortName"] = line.find(".//netex:ShortName", namespace).text
	except:
		print("Error in ", file)

	return data

data = get_data_from_file("Data/AErF_616/NX-PI-01_DK_NAP_LINE_AErF-616-AEroefaergerne_20240212.xml")
print(data)
