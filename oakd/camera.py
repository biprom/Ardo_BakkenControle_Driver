
import math
import time

import depthai as dai
import cv2
import numpy
import numpy as np


from depthaiTools.calc import HostSpatialsCalc
from depthaiTools.utility import TextHelper
from opencv.cvOperations import CVOperations
from configparser import ConfigParser


class Camera:

    config = ConfigParser()
    config.read('app.ini')

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int):

        config = ConfigParser()
        config.read('app.ini')

        self.connectToCamera(device_info, friendly_id)
        self.gapSurfaces = []
        self.polygonList = []
        self.configData = config['Windows']
        #self.configData = config['Mac']
        self.depthFrame = numpy.ndarray

    def check(self, p1, p2, base_array):
        """
        Uses the line defined by p1 and p2 to check array of
        input indices against interpolated value

        Returns boolean array, with True inside and False outside of shape
        """
        idxs = np.indices(base_array.shape)  # Create 3D array of indices

        p1 = p1.astype(float)
        p2 = p2.astype(float)

        # Calculate max column idx for each row idx based on interpolated line between two points
        if p1[0] == p2[0]:
            max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
            sign = np.sign(p2[1] - p1[1])
        else:
            max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
            sign = np.sign(p2[0] - p1[0])
        return idxs[1] * sign <= max_col_idx * sign

    def create_polygon(self, shape, vertices):
        """
        Creates np.array with dimensions defined by shape
        Fills polygon defined by vertices with ones, all other values zero"""
        base_array = np.zeros(shape, dtype=np.uint32)  # Initialize your array of zeros

        fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

        # Create check array for each edge segment, combine into fill array
        for k in range(vertices.shape[0]):
            fill = np.all([fill, self.check(vertices[k - 1], vertices[k], base_array)], axis=0)

        # Set all values inside polygon to one
        base_array[fill] = 1

        return base_array

    def _create_pipeline(self):

        # RGB cam -> 'rgb'
        controlInRGB = self.pipeline.create(dai.node.XLinkIn)
        cam_rgb = self.pipeline.create(dai.node.Camera)
        controlInLEFT = self.pipeline.create(dai.node.XLinkIn)
        monoLeft = self.pipeline.create(dai.node.Camera)
        controlInRIGHT = self.pipeline.create(dai.node.XLinkIn)
        monoRight = self.pipeline.create(dai.node.Camera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)


        #cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setSize(1280, 720)


        # Properties
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoLeft.setSize(1280, 720)

        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        monoRight.setSize(1280, 720)

        self.stereo.initialConfig.setConfidenceThreshold(255)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(False)

        self.xout_rgb = self.pipeline.createXLinkOut()
        self.xout_rgb.setStreamName("still")
        cam_rgb.still.link(self.xout_rgb.input)

        controlInRGB.setStreamName('controlRgb')
        controlInLEFT.setStreamName('controlLeft')
        controlInRIGHT.setStreamName('controlRight')

        # Linking
        monoLeft.still.link(self.stereo.left)
        monoRight.still.link(self.stereo.right)
        controlInRGB.out.link(cam_rgb.inputControl)
        controlInLEFT.out.link(monoLeft.inputControl)
        controlInRIGHT.out.link(monoRight.inputControl)

        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        self.stereo.depth.link(xoutDepth.input)

        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("disp")
        self.stereo.disparity.link(xoutDepth.input)


    # def _create_pipeline(self):
    #
    #     # RGB cam -> 'rgb'
    #     controlIn = self.pipeline.create(dai.node.XLinkIn)
    #     configIn = self.pipeline.create(dai.node.XLinkIn)
    #     cam_rgb = self.pipeline.create(dai.node.ColorCamera)
    #     monoLeft = self.pipeline.create(dai.node.MonoCamera)
    #     monoRight = self.pipeline.create(dai.node.MonoCamera)
    #     self.stereo = self.pipeline.create(dai.node.StereoDepth)
    #
    #
    #     #cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    #     cam_rgb.setFps(0.5)
    #     cam_rgb.setPreviewSize(640, 360)
    #     cam_rgb.setInterleaved(False)
    #     cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    #     cam_rgb.setPreviewKeepAspectRatio(False)
    #
    #
    #     # Properties
    #     monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    #     monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    #     monoLeft.setFps(1)
    #     monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    #     monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    #     monoRight.setFps(1)
    #
    #     self.stereo.initialConfig.setConfidenceThreshold(255)
    #     self.stereo.setLeftRightCheck(True)
    #     self.stereo.setSubpixel(False)
    #
    #     self.xout_rgb = self.pipeline.createXLinkOut()
    #     self.xout_rgb.setStreamName("still")
    #     cam_rgb.still.link(self.xout_rgb.input)
    #
    #     controlIn.setStreamName('control')
    #     configIn.setStreamName('config')
    #
    #     # Linking
    #     monoLeft.out.link(self.stereo.left)
    #     monoRight.out.link(self.stereo.right)
    #     controlIn.out.link(cam_rgb.inputControl)
    #     configIn.out.link(cam_rgb.inputConfig)
    #
    #     xoutDepth = self.pipeline.create(dai.node.XLinkOut)
    #     xoutDepth.setStreamName("depth")
    #     self.stereo.depth.link(xoutDepth.input)
    #
    #     xoutDepth = self.pipeline.create(dai.node.XLinkOut)
    #     xoutDepth.setStreamName("disp")
    #     self.stereo.disparity.link(xoutDepth.input)
    #
    #     # Create a BoardConfig object
    #     config = depthai.BoardConfig()
    #     # Set the parameters
    #     config.watchdogInitialDelayMs = 6000
    #     config.watchdogTimeoutMs = 10000
    #     self.pipeline.setBoardConfig(config)



    #This function is not used
    def getImages(self, params):

        for i in range(3):
           self.controlQueueRGB.send(self.ctrlRGB)
        in_rgb = self.rgb_queue.tryGet()

        if in_rgb is None:
            return

        self.frame_rgb = in_rgb.getCvFrame()
        cpHoughImage = np.copy(self.frame_rgb)

        dstGrayScale = self.cvoperations.grayScaleImage(self.frame_rgb)

        dstBlurry = self.cvoperations.blurImage(dstGrayScale, (params['width'], params['height']))

        dstEdgeDetection = self.cvoperations.edgeDetection(dstBlurry, params['threshold1'], params['threshold2'],
                                                           params['l2Gradient'])

        response = {"Images are stored on hostfolder"}
        cv2.imwrite(self.configData['basepath']+'grayscale.jpg', dstGrayScale)
        cv2.imwrite(self.configData['basepath']+'blurry.jpg', dstBlurry)
        cv2.imwrite(self.configData['basepath']+'edge.jpg', cpHoughImage)
        return self.configData['basepath']+'edge.jpg', response

    def getDepth(self):
        self.controlQueueLEFT.send(self.ctrlLEFT)
        self.controlQueueRIGHT.send(self.ctrlRIGHT)
        time.sleep(0.5)
        self.depthData = self.depthQueue.tryGet()


    def getNewGeometry(self, params):
        self.depthData = None;
        while self.depthData is None:
            self.getDepth()
        self.depthFrame = self.depthData.getFrame()

        # distance from lens to case (
        caseDistance = params['caseDistance']
        thresholdNoise = params['thresholdNoise']
        #minimalSurface = params['minimalSurface']
        #polygonPointList = params['polygonPoints']

        #self.polygonList.clear()
        #for index in range(len(polygonPointList)):
        #    x = polygonPointList[index]['x']
        #    y = polygonPointList[index]['y']
        #    self.polygonList.append([y, x])

        #vertices = np.array(self.polygonList, dtype=np.uint32)


        pic_array = np.asarray(self.depthFrame, dtype=np.uint32)
        #x, y = pic_array.shape
        #polygon_array = self.create_polygon([x, y], vertices)

        # multiply matrices to preform a mask
        result = pic_array

        # set noise to zero so we do not care about it
        result[result > thresholdNoise] = 0

        # filter the gaps
        # result[result > caseDistance] = 255
        result[result < caseDistance] = 0

        result[result > 0] = 255

        # normalize image to max measured value (ex noise)
        # normalize = (result * (255 / np.max(result))).astype(np.uint8)
        result = result.astype(np.uint8)

        #self.frame_rgb = in_rgb.getCvFrame()

        # dilate edges
        #kernel = np.ones((3, 3), np.uint8)
        #dilated_image = cv2.erode(result, kernel, iterations=1)

        cpHoughImage = np.copy(result)

        #dstGrayScale = self.cvoperations.grayScaleImage(self.frame_rgb)
        #cv2.imshow(self.window_name, dstGrayScale )

        #dstBlurry = self.cvoperations.blurImage(cpHoughImage,(params['width'],params['height']))
        #cv2.imshow("blurry",dstBlurry)

        dstEdgeDetection = self.cvoperations.edgeDetection(cpHoughImage,params['threshold1'],params['threshold2'],params['l2Gradient'])
        #cv2.imshow("edgeDetection", dstEdgeDetection)

        #cv2.imwrite(self.configData['basepath']+'grayscale.jpg', dstGrayScale)
        #cv2.imwrite(self.configData['basepath']+'blurry.jpg', dstBlurry)
        cv2.imwrite(self.configData['basepath']+'edgeDetection.jpg', dstEdgeDetection)

        houghLines = self.cvoperations.houghLines(dstEdgeDetection, params['rho'], params['theta'], params['threshold'])
        if houghLines is not None:
            self.geometryList.clear()
            for i in range(0, len(houghLines)):
                rho = houghLines[i][0][0]
                theta = houghLines[i][0][1]

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cpHoughImage, pt1, pt2, (100), 3, cv2.LINE_AA)
                cv2.putText(cpHoughImage, str(i), (pt2[0], pt2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100), 2, cv2.LINE_AA)
                #cv2.putText(cpHoughImage, str(i), (pt1[0], pt1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100), 2, cv2.LINE_AA)
                self.geometryList.append({'id': i, 'rho': str(rho), 'theta': str(theta*57.2958)})
        response = {"status": "status from biprom-driver",
                    "message": "message from biprom-driver",
                    "geometryEntityList": self.geometryList}
        cv2.imwrite(self.configData['basepath']+'edge.jpg', cpHoughImage)
        return self.configData['basepath']+'edge.jpg',response



    def getRoiMeasurements(self, rois):
        index = 0
        self.roiList.clear()

        self.depthData = None;
        while self.depthData is None:
            self.getDepth()
        # self.depthFrame = self.depthData.getFrame()
        #
        #
        # self.controlQueueLEFT.send(self.ctrlLEFT)
        # self.controlQueueRIGHT.send(self.ctrlRIGHT)
        # self.depthData = self.depthQueue.tryGet()
        #
        # while self.depthData == None:
        #     self.controlQueueLEFT.send(self.ctrlLEFT)
        #     self.controlQueueRIGHT.send(self.ctrlRIGHT)
        #     self.depthData = self.depthQueue.tryGet()

        # Get disparity frame for nicer depth visualization
        self.disp = self.dispQ.tryGet().getFrame()
        time.sleep(0.5)
        self.disp = (self.disp * (255 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        self.disp = cv2.applyColorMap(self.disp, cv2.COLORMAP_JET)


        for roi in rois:
            id = rois[index]['_id']

            delta = int(rois[index]['delta'])
            number = int(rois[index]['number'])
            x = int(rois[index]['x'])
            y = int(rois[index]['y'])
            min = int(rois[index]['min'])
            max = int(rois[index]['max'])

            self.hostSpatials.setDeltaRoi(delta)
            #self.hostSpatials.setLowerThreshold(min)
            #self.hostSpatials.setUpperThreshold(max)
            spatials, centroid = self.hostSpatials.calc_spatials(self.depthData, (x, y), id)
            self.roiList.append(spatials)

            self.text.rectangle(self.disp, (x - delta, y - delta), (x + delta, y + delta))
            self.text.putText(self.disp, str(number),(x, y))

            # self.text.putText(disp, "X: " + (
            #     "{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
            #                   (x + 10, y + 20))
            # self.text.putText(disp, "Y: " + (
            #     "{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
            #                   (x + 10, y + 35))
            # self.text.putText(disp,
            #                   "Z: " + ("{:.1f}mm".format(spatials['z']) if not math.isnan(spatials['z']) else "--"),
            #                   (x + 10, y + 50))

            cv2.imwrite(self.configData['basepath']+'disp.jpg', self.disp)
            index += 1
        return self.configData['basepath']+"disp.jpg", self.roiList

    def getGapSurfaces(self, camParams):

        for i in range(3):
           self.controlQueueLEFT.send(self.ctrlLEFT)
           self.controlQueueRIGHT.send(self.ctrlRIGHT)

        self.depthData = self.depthQueue.tryGet()

        while self.depthData is None:
            self.controlQueueLEFT.send(self.ctrlLEFT)
            self.controlQueueRIGHT.send(self.ctrlRIGHT)
            self.depthData = self.depthQueue.tryGet()

        #distance from lens to case (
        caseDistance = camParams['caseDistance']
        thresholdNoise = camParams['thresholdNoise']
        minimalSurface = camParams['minimalSurface']
        polygonPointList = camParams['polygonPoints']
        self.polygonList.clear()
        for index in range(len(polygonPointList)):
             x = polygonPointList[index]['x']
             y = polygonPointList[index]['y']
             self.polygonList.append([y,x])

        vertices = np.array(self.polygonList, dtype=np.uint32)

        self.depthData = None;
        while self.depthData is None:
            self.getDepth()
        self.depthFrame = self.depthData.getFrame()

        pic_array = np.asarray(self.depthFrame, dtype=np.uint32)
        x, y = pic_array.shape
        polygon_array = self.create_polygon([x, y], vertices)

        # multiply matrices to preform a mask
        result = polygon_array * pic_array

        #set noise to zero so we do not care about it
        result[result > thresholdNoise] = 0

        # filter the gaps
        #result[result > caseDistance] = 255
        result[result < caseDistance] = 0

        result[result > 0] = 255

        #normalize image to max measured value (ex noise)
        #normalize = (result * (255 / np.max(result))).astype(np.uint8)
        result = result.astype(np.uint8)



        contours, hierarchy  = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        #Get external countours
        self.gapSurfaces.clear()
        for i in range(len(contours)):
              if hierarchy[0][i][3] == -1:
                  cv2.drawContours(result,contours,i,100,3)
                  area = cv2.contourArea(contours[i], False)
                  if area > minimalSurface:
                    x, y, w, h = cv2.boundingRect(contours[i])
                    self.gapSurfaces.append({'area': area, 'width': w, 'height': h})

        fileName = self.configData['basepath']+'gapNormalize.jpg'
        cv2.imwrite(fileName, result)

        return fileName,self.gapSurfaces


    def getRoiMeasurementsWithSameImage(self, rois):
        index = 0
        self.roiList.clear()

        depthData = self.depthQueue.get()

        #Get disparity frame for nicer depth visualization
        self.disp = self.dispQ.get().getFrame()
        self.disp = (self.disp * (255 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        self.disp = cv2.applyColorMap(self.disp, cv2.COLORMAP_JET)


        for roi in rois:
            id = rois[index]['_id']

            delta = int(rois[index]['delta'])
            number = int(rois[index]['number'])
            x = int(rois[index]['x'])
            y = int(rois[index]['y'])
            min = int(rois[index]['min'])
            max = int(rois[index]['max'])

            self.hostSpatials.setDeltaRoi(delta)
            #self.hostSpatials.setLowerThreshold(min)
            #self.hostSpatials.setUpperThreshold(max)
            spatials, centroid = self.hostSpatials.calc_spatials(self.depthData, (x, y), id)
            self.roiList.append(spatials)

            self.text.rectangle(self.disp, (x - delta, y - delta), (x + delta, y + delta))
            self.text.putText(self.disp, str(number),(x, y))

            # self.text.putText(disp, "X: " + (
            #     "{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
            #                   (x + 10, y + 20))
            # self.text.putText(disp, "Y: " + (
            #     "{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
            #                   (x + 10, y + 35))
            # self.text.putText(disp,
            #                   "Z: " + ("{:.1f}mm".format(spatials['z']) if not math.isnan(spatials['z']) else "--"),
            #                   (x + 10, y + 50))

            cv2.imwrite(self.configData['basepath']+'dispSameAsLastImage.jpg', self.disp)
            index += 1
        return self.configData['basepath']+"dispSameAsLastImage.jpg", self.roiList

    def connectToCamera(self,device_info,friendly_id):
        self.geometryList = []
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self.pipeline = dai.Pipeline()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)
        self.text = TextHelper()
        self.hostSpatials = HostSpatialsCalc(self.device)

        self.rgb_queue = self.device.getOutputQueue(name="still", maxSize=1, blocking=False)
        self.depthQueue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        self.dispQ = self.device.getOutputQueue(name="disp", maxSize=1, blocking=False)
        self.controlQueueRGB = self.device.getInputQueue('controlRgb', maxSize=1, blocking=False)
        self.controlQueueLEFT = self.device.getInputQueue('controlLeft', maxSize=1, blocking=False)
        self.controlQueueRIGHT = self.device.getInputQueue('controlRight', maxSize=1, blocking=False)

        self.ctrlRGB = dai.CameraControl()
        self.ctrlRGB.setCaptureStill(True)

        self.ctrlLEFT = dai.CameraControl()
        self.ctrlLEFT.setCaptureStill(True)

        self.ctrlRIGHT = dai.CameraControl()
        self.ctrlRIGHT.setCaptureStill(True)

        self.roiList = []

        self.cvoperations = CVOperations("message")
        self.cvoperations = CVOperations("message")

        self.x = None
        self.y = None
        self.delta = None
        self.depthData = None

        self.disp = None
        self.statusBlock = False
        print("=== Connected to " + self.device_info.getMxId())

    def my_filtering_function(self,pair):
        unwanted_key1 = '_id'
        unwanted_key2 = 'number'
        key, value = pair
        if (key == unwanted_key1) or (key == unwanted_key2):
            return False  # filter pair out of the dictionary
        else:
            return True  # keep pair in the filtered dictionary


