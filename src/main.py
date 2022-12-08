import argparse
import resnet50
import resnet50_proxy_anchor
import utils
import json
import sklearn
import tensorflow as tf
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script used to measure performance of each model on the Paris6k dataset.')
    parser.add_argument("-m", "--model", default="baseline", choices=['baseline', 'proxy_anchor'], help="CBIR models")
    parser.add_argument("--dist", default="euclidean", choices=["cosine", "euclidean"], help="Distance metrics used to measure similarities")
    args = parser.parse_args()
    # Search with one more because the model contains all images of the dataset including the testing set
    nb_neigh = 10
    batch_size = 16
    # Open ground truth to collect classes
    f = open("./static/Paris_buildings/GT.json", "r")
    data = json.load(f)
    f.close()
    # Many images are not annoted in the Paris6k
    test_x_path, test_y, train_x_path, train_y = utils.create_sets_from_gt_Paris(data, nb_queries = 25)
    print("Train dataset have", len(train_x_path), "images.")
    print("Test dataset have", len(test_x_path), "images.")
    model = None
    # Load only images of the Paris6k dataset
    if (args.model == "baseline"):
        model = resnet50.resnetdlim("./static/Paris_buildings", train_x_path, args.dist)
        distances, results = model.execute_query(test_x_path, nb_neigh)
        resnet50_proxy_anchor.mAp_resnet(results, test_x_path, train_x_path, nb_neigh)
    elif (args.model == "proxy_anchor"):
        # Train a new model
        train_x_path, train_y = sklearn.utils.shuffle(train_x_path, train_y)

        if (not os.path.isdir("./models/anchors")):
            print("Load training dataset")
            train_dataset = tf.data.Dataset.from_tensor_slices(np.array([resnet50_proxy_anchor.preprocess_image(file) for file in train_x_path]))
            labels_set = tf.data.Dataset.from_tensor_slices(train_y)
            train_dataset = tf.data.Dataset.zip((train_dataset, labels_set))
            train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
            # Data augmentation
            data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal_and_vertical")])
            train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            train_dataset.prefetch(8)
            model = resnet50_proxy_anchor.ResNet50_ProxyAnchor(nb_classes=11)
            print("Launch training")
            model.training(train_dataset, epochs=20, lr=0.0001)
            print("Re-run the program to run metrics !")
            exit(1)
        model = tf.keras.models.load_model("./models/anchors", compile=False)
        ref = np.array([resnet50_proxy_anchor.preprocess_image(file) for file in train_x_path])
        train = tf.data.Dataset.from_tensor_slices((ref))
        train = train.batch(batch_size)
        testing = np.array([resnet50_proxy_anchor.preprocess_image(file) for file in test_x_path])
        test_dataset = tf.data.Dataset.from_tensor_slices((testing))
        test_dataset = test_dataset.batch(batch_size)
        print("Generate features")
        queries = model.predict(test_dataset)
        references_embeddings = model.predict(train)
        search_engine = NearestNeighbors(metric=args.dist, algorithm="brute")
        search_engine.fit(references_embeddings)
        distances, results = search_engine.kneighbors(queries, nb_neigh)
        resnet50_proxy_anchor.mAp_resnet(results, test_x_path, train_x_path, nb_neigh)
    else:
        print("Specify the model !")
        exit(1)