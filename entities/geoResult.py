import json


class GeometryResult:
    def __init__(self,refimageId,geometryEntityList):
        self.refimageId = refimageId
        self.geometryEntityList = geometryEntityList

    def returnJSON(self):
        value = {
            "refimageId": str(self.refimageId),
            "geometry": self.geometryEntityList
        }
        return json.dumps(value)




