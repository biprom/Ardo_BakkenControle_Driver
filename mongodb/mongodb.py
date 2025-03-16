from pymongo import MongoClient

import gridfs

class MongoDB:

    def __init__(self, ipadress, port, databasename):
        self.ipadress = ipadress
        self.port = port
        self.databasename = databasename
        self.database = None

    def connect(self):
        connection = MongoClient(self.ipadress,self.port)
        # Connect to the Database where the images will be stored.
        self.database = connection[self.databasename]

    def storeImage(self,imageLink):
        fs = gridfs.GridFS(self.database)
        file = imageLink

        # Open the image in read-only format.
        with open(file, 'rb') as f:
            contents = f.read()

        # Now store/put the image via GridFs object.
        id = fs.put(contents, filename="file")
        return id

    def getParamsForCam(self,address):
        collection = self.database['camera']
        data = collection.find({"ipAdres":address})
        for x in data:
            return (x)

    def getRoisForCam(self,address):
        collection = self.database['measuredRoiEntity']
        data = collection.find({"$and": [{"status":"REF"}, {"selectedCamera.ipAdres":address}]}).sort({"_id":-1}).limit(1)
        for x in data:
            return (x)


    def getLastGeoRefForCam(self,address):
        collection = self.database['geometryResult']
        #data = collection.find_one({ "$and": [{"status":"REF"}, {"selectedCamera.ipAdres":address}]})
        data = collection.find({ "$and": [{"status":"REF"}, {"selectedCamera.ipAdres":address}]}).sort({"_id":-1}).limit(1)
        return (data)