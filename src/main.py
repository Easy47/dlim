import argparse
import utils
import resnet50

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data", default=None, help="Load path of the dataset folder")
    args = parser.parse_args()
    if (args.data is not None):
        paths = utils.collect_INRIA_Holidays_paths(args.data)
        model = resnet50.resnetdlim()
        model.get_reference(paths)
        print(paths[0])
        print("Execute Query")
        distance, result = model.execute_query([paths[0]])
        for r in result[0]:
            print(paths[r])
        print(distance, result)