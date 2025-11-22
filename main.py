# This is a sample Python script.
import json
from time import sleep

import depthai as dai

from entities.GapResult import GapResult
from entities.geoResult import GeometryResult
from entities.roiResult import RoiResult
from mongodb.mongodb import MongoDB
from oakd.camera import Camera
from flask import Flask, request
from typing import List



# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

class obj:
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)

    def dict2obj(dict1):
        # using json.loads method and passing json.dumps
        # method and custom object hook as arguments
        return json.loads(json.dumps(dict1), object_hook=obj)



app = Flask(__name__)

print("Casemeasurement starting...")

print("Connect to all available devices...")


class obj:

    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

#device_infos = dai.Device.getAllAvailableDevices()
device_infos = [dai.DeviceInfo("192.168.68.200")]
#device_infos = [dai.DeviceInfo("10.127.8.100"),
#                 dai.DeviceInfo("10.127.8.101"),
#                 dai.DeviceInfo("10.127.8.102"),
#                 dai.DeviceInfo("10.127.8.103"),
#                 dai.DeviceInfo("10.127.8.104"),
#                 dai.DeviceInfo("10.127.8.105")]
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

device_infos.sort(key=lambda x: x.name, reverse=True) # sort the cameras by their mxId

cameras: List[Camera] = []

friendly_id = 0
for device_info in device_infos:
    friendly_id += 1
    print("Found camera MXID:", device_info.name)
    cameras.append(Camera(device_info, friendly_id))

selected_camera = cameras[0]

# Connect to DB
mongodb = MongoDB("127.0.0.1", 27017, "ArdoCV")
mongodb.connect()

def select_camera(friendly_id: int):
    global selected_camera
    i = friendly_id - 1
    if i >= len(cameras) or i < 0:
        return None
    selected_camera = cameras[i]
    print(f"Selected camera {friendly_id}")
    return selected_camera

select_camera(1)

@app.route('/getGeometryImageIdAndLines', methods=['GET'])
def getGeometryImageAndLines():
    args = request.args
    camParams = mongodb.getParamsForCam(args.get("address"))
    for camera in cameras:
        if camera.device_info.name == args.get("address"):
            cpHoughImage,refLines = camera.getNewGeometry(camParams)
            refImageId = mongodb.storeImage(cpHoughImage)
            geoResult = GeometryResult(refImageId,refLines)
            response = app.response_class(
                response=geoResult.returnJSON(),
                status=200,
                mimetype='application/json')
            return response


@app.route('/getDepthMeasurement', methods=['GET'])
def getDepthImage():
    args = request.args
    measuredRoiEntity = mongodb.getRoisForCam(args.get("address"))
    for camera in cameras:
        if camera.device_info.name == args.get("address"):
            imageLink, roiList = camera.getRoiMeasurements(measuredRoiEntity['roi'])
            imageId = mongodb.storeImage(imageLink)
            roiResult = RoiResult(imageId,roiList)
            response = app.response_class(
                response=roiResult.returnJSON(),
                status=200,
                mimetype='application/json')
            return response

@app.route('/getGapSurfaces', methods=['GET'])
def getGapSurfaces():
    args = request.args
    cameraParametersFromDB = mongodb.getParamsForCam(args.get("address"))
    for camera in cameras:
        if camera.device_info.name == args.get("address"):
            imageLink, surfaceArray = camera.getGapSurfaces(cameraParametersFromDB)
            imageId = mongodb.storeImage(imageLink)
            gapResult = GapResult(imageId,surfaceArray)
            response = app.response_class(
                response=gapResult.returnJSON(),
                status=200,
                mimetype='application/json')
            return response



@app.route('/getDepthMeasurementWithLastImage', methods=['GET'])
def getDepthImageWithLastImage():
    args = request.args
    measuredRoiEntity = mongodb.getRoisForCam(args.get("address"))
    for camera in cameras:
        if camera.device_info.name == args.get("address"):
            imageLink, roiList = camera.getRoiMeasurementsWithSameImage(measuredRoiEntity['roi'])
            imageId = mongodb.storeImage(imageLink)
            roiResult = RoiResult(imageId,roiList)
            response = app.response_class(
                response=roiResult.returnJSON(),
                status=200,
                mimetype='application/json')
            return response

@app.route('/getOnlineDevices', methods=['GET'])
def getOnlineDevices():
    args = request.args
    for camera in cameras:
        if ((camera.device_info.name == args.get("address")) & (camera.statusBlock == False)):
            try:
                connectedCameras = str(camera.device.getConnectedCameras())
            except:
                connectedCameras = "N/A"
            response = app.response_class(
            response=connectedCameras,
            status=200,
            mimetype='text/plain')
            return response
        else:
            response = app.response_class(
                response="N/A",
                status=200,
                mimetype='text/plain')
            return response

@app.route('/exitDevice', methods=['GET'])
def exitDevice():
    args = request.args
    for camera in cameras:
        if (camera.device_info.name == args.get("address")):
            camera.statusBlock = True
            camera.device.close()
            retval = 1;
            response = app.response_class(
            response=str(retval),
            status=200,
            mimetype='text/plain')
            return response


@app.route('/connectToDevice', methods=['GET'])
def connectToDevice():
    args = request.args
    for camera in cameras:
        if camera.device_info.name == args.get("address"):
            camera.connectToCamera(camera.device_info,camera.friendly_id)
            camera.statusBlock = False
            retval = 3;
            response = app.response_class(
            response=str(retval),
            status=200,
            mimetype='text/plain')
            return response
