import numpy as np
import tensorflow as tf
import sklearn
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
import utils
import os
import json
import glob 

target_shape = (224, 224)
class resnetdlim:
    def __init__(self, dataset_path, jpg_paths):
        # Creation of the model using resnet input and flattened resnet output
        self.ResNet = resnet.ResNet50(
            weights="imagenet", include_top=False,  input_shape=target_shape + (3,), pooling="avg"
        )
        output = layers.Flatten()(self.ResNet.output)
        self.resnet_feature_extractor = Model(self.ResNet.input, output, name="Embedding")
        self.resnet_feature_extractor.save("resnet_base")
        self.dataset_path = dataset_path
        self.get_reference(jpg_paths)

    def get_reference(self, jpg_paths):
        # Set jpg_paths to get it back latter
        self.jpg_paths = jpg_paths

        # Get embeddings in cache.npy files or generate them 
        self.ref_embedding = self.get_embeddings(jpg_paths, 16, self.dataset_path + "/cache.npy")

        # Fit search_engine on embeddings
        self.search_engine = NearestNeighbors()
        self.search_engine.fit(self.ref_embedding)

    def preprocess_image(self, filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """
        image = tf.keras.utils.load_img(filename)
        image = image.convert('RGB')
        image = image.resize((target_shape))
        return preprocess_input(np.array(image))

    def get_embeddings(self, jpg_paths, batch_size, cache=None):

        # If embeddings are cached we load them. Else we generate
        ref = None
        if (cache != None and os.path.isfile(cache)):
            with open(cache, "rb") as f:
                ref = np.load(f)
        else:
            ref = np.array([self.preprocess_image(path) for path in jpg_paths])
        
        # Cache the results if the file does not exists
        if (cache != None and not os.path.isfile(cache)):
            with open(cache, "wb") as f:
                np.save(f, ref)
        train_dataset = tf.data.Dataset.from_tensor_slices((ref))
        train_dataset = train_dataset.batch(batch_size)
        ref_embedding = self.resnet_feature_extractor.predict(train_dataset)
        return ref_embedding

    def execute_query(self, paths, nb_neigh):
        distances = None
        indices = None

        # If cache files exists we load them
        if (os.path.isfile(self.dataset_path + "/distances.npy")):
            with open(self.dataset_path + "/distances.npy", "rb") as f:
                distances = np.load(f)
            print ("\n    Loaded distances from cache...")
        else:
            print("\n   NO distances cached, generating.. ")

        if (os.path.isfile(self.dataset_path + "/indices.npy")):
            with open(self.dataset_path + "/indices.npy", "rb") as f:
                indices = np.load(f)
            print ("\n    Loaded indices from cache...")
        else:
            print("\n   NO indices cached, generating.. ")

        # If there are no cache files, we generate them
        if (distances is None or indices is None):
            print("\n   generating ...")
            query_vectors = self.get_embeddings(paths, 16, None)
            distances, indices = self.search_engine.kneighbors(query_vectors, nb_neigh)
           
            with open(self.dataset_path + "/distances.npy", "wb") as f:
                np.save(f, distances)

            with open(self.dataset_path + "/indices.npy", "wb") as f:
                np.save(f, indices)
        
        return (distances, indices)

    def mAp_resnet(self, results, queries, reference, nb_neigh):
        # Load the gt file
        PATH_TO_GT = self.dataset_path + "/GT.json"
        gt_data = None
        with open(PATH_TO_GT, 'r') as in_gt:
            gt_data = json.load(in_gt)

        # Reference images from the model
        IMG_IDS = [p.split('/')[-1][:-4] for p in reference]
        # Ids of the queries
        QUERY_IDS = [p.split('/')[-1][:-4] for p in queries]

        # Mapping from the image name to index in reference
        imgid_to_index = {imgid: ii for ii, imgid in enumerate(IMG_IDS)}
        query_imnos = [imgid_to_index[query_id] for query_id in QUERY_IDS]

        aps = []  # list of average precisions for all queries
        for qimno, qres in zip(query_imnos, results):
            qname = IMG_IDS[qimno]
            
            # collect the positive results in the dataset
            # the positives have the same prefix as the query image
            positive_results = [imgid_to_index[img_id] for img_id in gt_data[IMG_IDS[qimno]]]
            
            # ranks of positives. We skip the result #0, assumed to be the query image
            ranks = [i for i, res in enumerate(qres[1:]) if res in positive_results]
            
            # accumulate trapezoids with this basis
            recall_step = 0
            if (nb_neigh - 1 > len(positive_results)):
                recall_step = 1.0 / len(positive_results)  # FIXME what is the size of a step?
            else:
                recall_step = 1.0 / (nb_neigh - 1)

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

        print("mean AP = %.3f" % np.mean(aps)) 