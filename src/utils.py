import glob
from enum import Enum
import json

class Dataset(Enum):
    INRIA = 0
    PARIS = 1

def get_INRIA_path(filename):
    return "./static/INRIA_Holidays/jpg/" + filename + ".jpg"

def get_Paris_path(filename):
    index = filename[6:].index('_')
    folder = filename[6:index + 6]
    return "/".join(["./static/Paris_buildings/jpg", folder, filename]) + ".jpg"

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
    path = "./static/Paris_buildings/GT.json"
    f = open(path, "r")
    data = json.load(f)
    f.close()
    queries = list(data.keys())
    for i in range(len(queries)):
        queries[i] = get_Paris_path(queries[i])
    return queries

def create_sets_from_gt(gt_data, nb_queries = 1, dataset = Dataset.INRIA):
    tmp = []
    values = list(gt_data.values())

    test_x_path = []
    test_y = []
    train_x_path = []
    train_y = []

    label = 0
    for ii, imgid in enumerate(list(gt_data.keys())):
        if (imgid in tmp):
            continue
        if (len(values[ii]) < nb_queries):
            print("Error in create_sets_from_gt: too much nb_queries")
            continue
        tmp.append(imgid)
        test_y.append(label)
        if (dataset == Dataset.INRIA):
            test_x_path.append(get_INRIA_path(imgid))
        else:
            test_x_path.append(get_Paris_path(imgid))
        count = 1
        for posi in values[ii]:
            image_name = None
            if (dataset == Dataset.INRIA):
                image_name = get_INRIA_path(posi)
            else:
                image_name = get_Paris_path(posi)
            if (count < nb_queries):
                test_x_path.append(image_name)
                test_y.append(label)
                count = count + 1
            else:
                train_x_path.append(image_name)
                train_y.append(label)
            tmp.append(posi)
        label = label + 1

    return test_x_path, test_y, train_x_path, train_y

def create_sets_from_gt_Paris(gt_data, nb_queries = 1):
    values = list(gt_data.values())
    test_x_path = []
    test_y = []
    train_x_path = []
    train_y = []
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    classes = ["defense", "triomphe","pompidou","eiffel","invalides","pantheon","moulinrouge","museedorsay","sacrecoeur","notredame", "louvre"]
    for ii, imgid in enumerate(list(gt_data.keys())):
        if (len(values[ii]) < nb_queries):
            print("Error in create_sets_from_gt: too much nb_queries")
            continue
        directories = [direc.split("_")[1] for direc in gt_data[imgid]]
        directories = [i for i in directories if i != "general"]
        counter = Counter(directories)
        index = np.argmax(list(counter.values()))
        label = classes.index(list(counter.keys())[index])
        path = utils.get_Paris_path(imgid)
        if (not path in test_x_path and not path in train_x_path):
            test_x_path.append(path)
            test_y.append(label)
            counts[label] += 1
        for posi in gt_data[imgid]:
            image_name = utils.get_Paris_path(posi)
            if (counts[label] < nb_queries):
                if (not image_name in test_x_path and not image_name in train_x_path):
                    test_x_path.append(image_name)
                    test_y.append(label)
                    counts[label] += 1
            else:
                if (not image_name in train_x_path and not image_name in test_x_path):
                    train_x_path.append(image_name)
                    train_y.append(label)
    return test_x_path, test_y, train_x_path, train_y