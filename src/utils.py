import glob
import json

def collect_INRIA_Holidays_paths(path):
    path = path.strip()
    imgs = None
    if (path[-1] == "/"):
        imgs = glob.glob(path + "jpg/*.jpg", recursive=True)
    else:
        imgs = glob.glob(path + "/jpg/*.jpg", recursive=True)
    return imgs

def collect_Paris_buildings_paths(path):
    path = path.strip()
    imgs = None
    if (path[-1] == "/"):
        imgs = glob.glob(path + "jpg/*/*.jpg", recursive=True)
    else:
        imgs = glob.glob(path + "/jpg/*/*.jpg", recursive=True)
    return imgs

def get_Paris_buildings_queries():
    path = "./Paris_buildings/GT.json"
    f = open(path, "r")
    data = json.load(f)
    f.close()
    queries = list(data.keys())
    for i in range(len(queries)):
        index = queries[i][6:].index('_')
        folder = queries[i][6:index + 6]
        queries[i] = "/".join(["./Paris_buildings/jpg", folder, queries[i]]) + ".jpg"
    return queries