import argparse
import utils
import resnet50
import resnet50_triplets
import resnet50_batch_all
import utils
import json
from collections import Counter
import numpy as np

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


def mAp_resnet(results, queries, references, nb_neigh=9):
    PATH_TO_GT = "./Paris_buildings/GT.json"
    gt_data = None
    with open(PATH_TO_GT, 'r') as in_gt:
        gt_data = json.load(in_gt)

    # Reference images from the model
    IMG_IDS = [str(p).split('/')[-1][:-4] for p in references]
    # Ids of the queries
    QUERY_IDS = [str(p).split('/')[-1][:-4] for p in queries]

    gt_mapping = {imgid: ii for ii, imgid in enumerate(gt_data.keys())}

    # Mapping from the image name to index in reference
    reference_indexes = [gt_mapping[imgid] for imgid in IMG_IDS]
    
    aps = []  # list of average precisions for all queries
    for qname, qres in zip(QUERY_IDS, results):
        # collect the positive results in the dataset
        # the positives have the same prefix as the query image
        positive_results = [gt_mapping[img_id] for img_id in gt_data[qname]]
        # print(positive_results)
        # print([list(gt_mapping.keys())[pos] for pos in positive_results])
        # break
        #
        # ranks of positives. We skip the result #0, assumed to be the query image
        qres = [reference_indexes[r] for r in qres]
        ranks = [i for i, res in enumerate(qres) if res in positive_results]
        #
        # accumulate trapezoids with this basis
        recall_step = 0
        if (nb_neigh > len(positive_results)):
            recall_step = 1.0 / len(positive_results)  # FIXME what is the size of a step?
        else:
            recall_step = 1.0 / nb_neigh

        ap = 0
        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far
            # y-size on left side of trapezoid:
            precision_0 = ntp/float(rank) if rank > 0 else 1.0
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += ((precision_0 + precision_1) * recall_step) / 2
        print("query %s, AP = %.3f" % (qname, ap))
        aps.append(ap)

    print("mean AP = %.3f" % np.mean(aps))  # FIXME mean average precision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data", default=None, help="Load path of the dataset folder")
    args = parser.parse_args()
    if (args.data is not None):
        nb_neigh = 10
        args.data = args.data.strip(" /\n")
        jpg_paths = utils.collect_Paris_buildings_paths(args.data)


        path = "./Paris_buildings/GT.json"
        f = open(path, "r")
        data = json.load(f)
        f.close()

        test_x_path, test_y, train_x_path, train_y = create_sets_from_gt_Paris(data, 16)
        # jpg_paths = utils.collect_INRIA_Holidays_paths(args.data)
        model = resnet50.resnetdlim(args.data, train_x_path)
        # model = resnet50_batch_all.resnet_triplets(args.data, jpg_paths, dataset=utils.Dataset.INRIA)
        print("Execute Query")
        # queries = jpg_paths
        distances, results = model.execute_query(test_x_path, nb_neigh)
        # distances, results = model.execute_query(queries)
        # model.mAp_resnet(results, queries, reference)
        mAp_resnet(results, test_x_path, train_x_path, nb_neigh)