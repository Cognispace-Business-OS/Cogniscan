from pymongo import MongoClient
#python -m pip install "pymongo[srv]"
uri = "mongodb+srv://chiranjit:unlockit@cluster0.mksex.mongodb.net/?appName=Cluster0"

client = MongoClient(uri)

# create/use database
db = client["cognispace"]

# create/use collection
collection = db["agents"]

# insert JSON
collection.insert_one({
    "name": "startup_agent",
    "status": "running",
    "score": 0.91
})

print("Data inserted!")