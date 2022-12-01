import glob

def collect_INRIA_Holidays_paths(path):
    path = path.strip()
    imgs = None
    if (path[-1] == "/"):
        imgs = glob.glob(path + "*.jpg", recursive=True)
    else:
        imgs = glob.glob(path + "/*.jpg", recursive=True)
    return imgs

def mAp_resnet():
    PATH_TO_GT = os.path.join(PATH_TO_RESOURCES, "INRIA_HOLYDAYS.json")
    gt_data = None
    with open(PATH_TO_GT, 'r') as in_gt:
        gt_data = json.load(in_gt)

    query_imnos = [imgid_to_index[query_id] for query_id in gt_data.keys()]

    aps = []  # list of average precisions for all queries
    for qimno, qres in zip(query_imnos, results):
        qname = IMG_IDS[qimno]
    #     print("query:", qname)
        # collect the positive results in the dataset
        # the positives have the same prefix as the query image
        positive_results = [imgid_to_index[img_id] for img_id in gt_data[IMG_IDS[qimno]]]
    #     print("positive_results:", positive_results)
    #     print("qres:", qres)
        #
        # ranks of positives. We skip the result #0, assumed to be the query image
        ranks = [i for i, res in enumerate(qres[1:]) if res in positive_results]
    #     print("ranks:", ranks)
        #
        # accumulate trapezoids with this basis
        recall_step = 1.0 / len(positive_results)  # FIXME what is the size of a step?
        ap = 0
        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far
            # y-size on left side of trapezoid:
            precision_0 = ntp/float(rank) if rank > 0 else 1.0
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += recall_step * precision_0 + (recall_step * (precision_1 - precision_0)) / 2 # FIXME what is the area under the PR curve?
        print("query %s, AP = %.3f" % (qname, ap))
        aps.append(ap)

    print("mean AP = %.3f" % np.mean(aps))  # FIXME mean average precision