import xml.etree.ElementTree as ET

namespace = {'netex': 'http://www.netex.org.uk/netex'}

def get_data_from_path(path):
    tree = ET.parse(path)
    root = tree.getroot()
    bus_data = { "routePoints": [], "routes": [], "destinationDisplays": [], "scheduledStopPoints": [], "serviceLinks": [], "lines": [] }

    data = {
        "line_name": "",
    }

    dataObjectsTree = root.find(".//netex:dataObjects", namespace)
    routePointsTree = dataObjectsTree.find(".//netex:routePoints", namespace)
    for routePoint in routePointsTree.findall(".//netex:RoutePoint", namespace):
        id = routePoint.attrib['id']
        location = routePoint.find(".//netex:Location", namespace)
        longitude = location.find(".//netex:Longitude", namespace).text
        latitude = location.find(".//netex:Latitude", namespace).text
        routePointObj = { "id": id, "longitude": longitude, "latitude": latitude }
        bus_data["routePoints"].append(routePointObj)
        print("routePointObj", routePointObj)

    # LINK ROUTES TO ROUTE POINTS

    routesTree = dataObjectsTree.find(".//netex:routes", namespace)
    for route in routesTree.findall(".//netex:Route", namespace):
        id = route.attrib['id']
        lineRef = route.find(".//netex:LineRef", namespace).attrib['ref']
        directionRef = route.find(".//netex:DirectionRef", namespace).attrib['ref']
        pointsInSequence = route.find(".//netex:pointsInSequence", namespace)
        routeObj = { "id": id, "lineRef": lineRef, "directionRef": directionRef, "pointsInSequence": [] }
        for pointInSequence in pointsInSequence.findall(".//netex:PointOnRoute", namespace):
            order = pointInSequence.attrib['order']
            routePoint = "RoutePoint", pointInSequence.attrib['id']
            routePointRef = "RoutePointRef", pointInSequence.find(".//netex:RoutePointRef", namespace).attrib['ref']
            routePointOnOrderRef = {"routePoint": routePoint, "routePointRef": routePointRef}
            routeObj["pointsInSequence"].append(routePointOnOrderRef)
        bus_data["routes"].append(routeObj)
        print("routeObj", routeObj)

    # Destination displays

    destinationDisplaysTree = dataObjectsTree.find(".//netex:destinationDisplays", namespace)
    for destinationDisplay in destinationDisplaysTree.findall(".//netex:DestinationDisplay", namespace):
        id = destinationDisplay.attrib['id']
        sideText = destinationDisplay.find(".//netex:SideText", namespace).text
        frontText = destinationDisplay.find(".//netex:FrontText", namespace).text
        destinationDisplayObj = { "id": id, "sideText": sideText, "frontText": frontText }
        bus_data["destinationDisplays"].append(destinationDisplayObj)
        print("destinationDisplayObj", destinationDisplayObj)
        print()

    # Scheduled stop points

    scheduledStopPointsTree = dataObjectsTree.find(".//netex:scheduledStopPoints", namespace)
    for scheduledStopPoint in scheduledStopPointsTree.findall(".//netex:ScheduledStopPoint", namespace):
        id = scheduledStopPoint.attrib['id']
        name = scheduledStopPoint.find(".//netex:Name", namespace).text
        location = scheduledStopPoint.find(".//netex:Location", namespace)
        longitude = location.find(".//netex:Longitude", namespace).text
        latitude = location.find(".//netex:Latitude", namespace).text
        scheduledStopPointObj = { "id": id, "name": name, "longitude": longitude, "latitude": latitude }
        bus_data["scheduledStopPoints"].append(scheduledStopPointObj)
        print("scheduledStopPointObj", scheduledStopPointObj)
        print()

    # Service links

    serviceLinksTree = dataObjectsTree.find(".//netex:serviceLinks", namespace)
    for serviceLink in serviceLinksTree.findall(".//netex:ServiceLink", namespace):
        id = serviceLink.attrib['id']
        fromPointRef = serviceLink.find(".//netex:FromPointRef", namespace).attrib['ref']
        toPointRef = serviceLink.find(".//netex:ToPointRef", namespace).attrib['ref']
        serviceLinkObj = { "id": id, "fromPointRef": fromPointRef, "toPointRef": toPointRef }
        bus_data["serviceLinks"].append(serviceLinkObj)
        print("serviceLinkObj", serviceLinkObj)
        print()

    # Get line name 

    linesTree = dataObjectsTree.find(".//netex:lines", namespace)
    for line in linesTree.findall(".//netex:Line", namespace):
        id = line.attrib['id']
        name = line.find(".//netex:Name", namespace).text
        lineObj = { "id": id, "name": name }
        bus_data["lines"].append(lineObj)
        print("lineObj", lineObj)
        print()

    return bus_data

bus_data = get_data_from_path("Data/MOVIA_MOVIA/NX-PI-01_DK_NAP_LINE_MOVIA-MOVIA-300S_20240212.xml")
print(bus_data)
