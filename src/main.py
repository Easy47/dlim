import argparse
import utils
import resnet50
import resnet50_triplets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data", default=None, help="Load path of the dataset folder")
    args = parser.parse_args()
    if (args.data is not None):
        nb_neigh = 10
        args.data = args.data.strip(" /\n")
        # jpg_paths = utils.collect_Paris_buildings_paths(args.data)
        jpg_paths = utils.collect_INRIA_Holidays_paths(args.data)
        model = resnet50.resnetdlim(args.data, jpg_paths)
        # model = resnet50_triplets.resnet_triplets(args.data, jpg_paths)
        print("Execute Query")

        # queries = utils.get_Paris_buildings_queries()
        queries = jpg_paths
        reference = model.jpg_paths
        distances, results = model.execute_query(queries, nb_neigh)
        print("Res", results[0])
        print("Dis", distances[0])
        print("jpg path", jpg_paths[0])
        print("Results shape : ", results.shape)
        model.mAp_resnet(results, queries, reference, nb_neigh)