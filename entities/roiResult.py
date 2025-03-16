import json


class RoiResult():
    def __init__(self,imageId,roiEntityList):
        self.imageId = imageId
        self.roiEntityList = roiEntityList

    def returnJSON(self):
        value = {
            "imageId": str(self.imageId),
            "roiEntityList": self.roiEntityList
        }
        return json.dumps(value)
