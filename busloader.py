import xml.etree.ElementTree as ET

namespace = {'netex': 'http://www.netex.org.uk/netex'}

def get_data_from_file(file):
	error = None
	tree = ET.parse(file)
	root = tree.getroot()
	data = {
		"line": {
			"name": "",
			"shortName": "",
		},
		"routePoints": [],
		"routes": []
	}

	dataObjectsTree = root.find(".//netex:dataObjects", namespace)
	routePointsTree = dataObjectsTree.find(".//netex:routePoints", namespace)

	# Check if a Location is NoneType, because some routepoints have location stored differently
	try:
		routes, routePoints = method1(dataObjectsTree)
		data["routePoints"] = routePoints
		data["routes"] = routes
		data["line"]["name"], data["line"]["shortName"] = get_line_name(dataObjectsTree)
	except:
		error = "Error in file " + file

#	routePoints = []
#	for routePoint in routePointsTree.findall(".//netex:RoutePoint", namespace):
#		location = routePoint.find(".//netex:Location", namespace)
#		routePointObj = {
#			"id": routePoint.attrib['id'],
#			"longitude": location.find(".//netex:Longitude", namespace).text,
#			"latitude": location.find(".//netex:Latitude", namespace).text
#		}
#		routePoints.append(routePointObj)

	# LINK ROUTES TO ROUTE POINTS
#	routesTree = dataObjectsTree.find(".//netex:routes", namespace)
#	for route in routesTree.findall(".//netex:Route", namespace):
#		routeObj = {
#			"id": route.attrib['id'],
#			"lineRef": route.find(".//netex:LineRef", namespace).attrib['ref'],
#			"directionRef": route.find(".//netex:DirectionRef", namespace).attrib['ref'],
#			"pointsInSequence": [] 
#		}
#		pointsInSequence = route.find(".//netex:pointsInSequence", namespace)
#		for pointInSequence in pointsInSequence.findall(".//netex:PointOnRoute", namespace):
#			routePointOnOrderRef = { 
#				"routePoint": pointInSequence.attrib['id'], 
#				"routePointRef": pointInSequence.find(".//netex:RoutePointRef", namespace).attrib['ref'], 
#				"order": pointInSequence.attrib['order'] 
#			}
#			routeObj["pointsInSequence"].append(routePointOnOrderRef)
#		data["routes"].append(routeObj)

#	scheduledStopPointsTree = dataObjectsTree.find(".//netex:scheduledStopPoints", namespace)
#	for scheduledStopPoint in scheduledStopPointsTree.findall(".//netex:ScheduledStopPoint", namespace):
#		name = scheduledStopPoint.find(".//netex:Name", namespace).text
#		location = scheduledStopPoint.find(".//netex:Location", namespace)
#		longitude = location.find(".//netex:Longitude", namespace).text
#		latitude = location.find(".//netex:Latitude", namespace).text
#		for rp in routePoints:
#			if rp["longitude"] == longitude and rp["latitude"] == latitude:
#				rp["name"] = name

#	data["routePoints"] = routePoints
	# Get line name

#	linesTree = dataObjectsTree.find(".//netex:lines", namespace)
#	for line in linesTree.findall(".//netex:Line", namespace):
#		data["line"]["name"] = line.find(".//netex:Name", namespace).text
#		data["line"]["shortName"] = line.find(".//netex:ShortName", namespace).text

	return data, error

def method1(dataObjectsTree: ET.ElementTree):
	routes = []
	routePoints = []
	routePointsTree = dataObjectsTree.find(".//netex:routePoints", namespace)
	for routePoint in routePointsTree.findall(".//netex:RoutePoint", namespace):
		location = routePoint.find(".//netex:Location", namespace)
		routePointObj = {
			"id": routePoint.attrib['id'],
			"longitude": location.find(".//netex:Longitude", namespace).text,
			"latitude": location.find(".//netex:Latitude", namespace).text
		}
		routePoints.append(routePointObj)

	# LINK ROUTES TO ROUTE POINTS
	routesTree = dataObjectsTree.find(".//netex:routes", namespace)
	for route in routesTree.findall(".//netex:Route", namespace):
		routeObj = {
			"id": route.attrib['id'],
			"lineRef": route.find(".//netex:LineRef", namespace).attrib['ref'],
			"directionRef": route.find(".//netex:DirectionRef", namespace).attrib['ref'],
			"pointsInSequence": [] 
		}
		pointsInSequence = route.find(".//netex:pointsInSequence", namespace)
		for pointInSequence in pointsInSequence.findall(".//netex:PointOnRoute", namespace):
			routePointOnOrderRef = { 
				"routePoint": pointInSequence.attrib['id'], 
				"routePointRef": pointInSequence.find(".//netex:RoutePointRef", namespace).attrib['ref'], 
				"order": pointInSequence.attrib['order'] 
			}
			routeObj["pointsInSequence"].append(routePointOnOrderRef)
		routes.append(routeObj)

	scheduledStopPointsTree = dataObjectsTree.find(".//netex:scheduledStopPoints", namespace)
	for scheduledStopPoint in scheduledStopPointsTree.findall(".//netex:ScheduledStopPoint", namespace):
		name = scheduledStopPoint.find(".//netex:Name", namespace).text
		location = scheduledStopPoint.find(".//netex:Location", namespace)
		longitude = location.find(".//netex:Longitude", namespace).text
		latitude = location.find(".//netex:Latitude", namespace).text
		for rp in routePoints:
			if rp["longitude"] == longitude and rp["latitude"] == latitude:
				rp["name"] = name

	return routes, routePoints

def method2(dataObjectsTree: ET.ElementTree):
	routes = []
	stopPlaces = []
	stopPlacesTree = dataObjectsTree.find(".//netex:stopPlaces", namespace)
	for stopPlace in stopPlacesTree.findall(".//netex:StopPlace", namespace):
		stopPlaceObj = {
			"id": stopPlace.attrib['id'],
			"name": stopPlace.find(".//netex:Name", namespace).text,
			"latitude": stopPlace.find(".//netex:Latitude", namespace).text,
			"longitude": stopPlace.find(".//netex:Longitude", namespace).text
		}
		stopPlaces.append(stopPlaceObj)
		print("Stop place: ", stopPlaceObj)

	print()
	scheduledStopPoints = []
	scheduledStopPointsTree = dataObjectsTree.find(".//netex:scheduledStopPoints", namespace)
	for scheduledStopPoint in scheduledStopPointsTree.findall(".//netex:ScheduledStopPoint", namespace):
		scheduledStopPointObj = {
			"id": scheduledStopPoint.attrib['id'],
			"name": scheduledStopPoint.find(".//netex:Name", namespace).text,
		}
		scheduledStopPoints.append(scheduledStopPointObj)
		print("Scheduled stop point: ", scheduledStopPointObj)

	print()
	serviceLinks = []
	serviceLinksTree = dataObjectsTree.find(".//netex:serviceLinks", namespace)
	for serviceLink in serviceLinksTree.findall(".//netex:ServiceLink", namespace):
		serviceLinkObj = {
			"id": serviceLink.attrib['id'],
			"from": serviceLink.find(".//netex:FromPointRef", namespace).attrib['ref'],
			"to": serviceLink.find(".//netex:ToPointRef", namespace).attrib['ref'],
		}
		serviceLinks.append(serviceLinkObj)
		print("Service link: ", serviceLinkObj)

	for serviceLink in serviceLinks:
		for scheduledStopPoint in scheduledStopPoints:
			if serviceLink['from'] == scheduledStopPoint['id']:
				fromPointS = scheduledStopPoint
			if serviceLink['to'] == scheduledStopPoint['id']:
				toPointS = scheduledStopPoint
		print("From point: ", fromPointS)
		print("To point: ", toPointS)
		for stopPlace in stopPlaces:
			if fromPointS['name'] in stopPlace['name']:
				fromPoint = stopPlace
			if toPointS['name'] in stopPlace['name']:
				toPoint = stopPlace
		print("From: ", fromPoint)
		print("To: ", toPoint)
		routeObj = {
			"from": fromPoint['id'],
			"to": toPoint['id'],
		}
		routes.append(routeObj)

	return routes, stopPlaces

def get_line_name(dataObjectsTree: ET.ElementTree):
	linesTree = dataObjectsTree.find(".//netex:lines", namespace)
	for line in linesTree.findall(".//netex:Line", namespace):
		return line.find(".//netex:Name", namespace).text, line.find(".//netex:ShortName", namespace).text

data = get_data_from_file("Data/MOVIA_MOVIA/NX-PI-01_DK_NAP_LINE_MOVIA-MOVIA-350S_20240212.xml")
print(data)
print()

data2 = get_data_from_file("Data/AEROE_AEROE/NX-PI-01_DK_NAP_LINE_AEROE-AEROE-790_20240212.xml")
print(data2)