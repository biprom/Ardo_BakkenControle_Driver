import json


class GapResult():
    def __init__(self,imageId,gapSurfaceEntityList):
        self.imageId = imageId
        self.gapSurfaceEntityList = gapSurfaceEntityList

    def returnJSON(self):
        value = {
            "imageId": str(self.imageId),
            "gapSurfaceEntityList": self.gapSurfaceEntityList
        }
        return json.dumps(value)