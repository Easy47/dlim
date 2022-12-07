import argparse
import resnet50
import resnet50_triplets
import resnet50_batch_all
import resnet50_batch_hard
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data", default=None, help="Load path of the dataset folder")
    args = parser.parse_args()
    if (args.data is not None):
        nb_neigh = 10
        # nb_neigh = 100
        args.data = args.data.strip(" /\n")
        jpg_paths = utils.collect_Paris_buildings_paths(args.data)
        #jpg_paths = utils.collect_INRIA_Holidays_paths(args.data)
        # model = resnet50.resnetdlim(args.data, jpg_paths)
        model = resnet50.resnetdlim(args.data, jpg_paths)
        print("Execute Query")
        
        queries = utils.get_Paris_buildings_queries()
        reference = model.jpg_paths
        distances, results = model.execute_query(queries, nb_neigh)
        # distances, results = model.execute_query(queries)
        print("Results shape : ", results.shape)
        # model.mAp_resnet(results, queries, reference)
        model.mAp_resnet(results, queries, reference, nb_neigh)
        print("Nombre d'image dans le dataset", len(reference))