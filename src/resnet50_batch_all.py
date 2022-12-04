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

# Layer class to join resnet with a siamese network
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, embeddings):
      dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
      square_norm = tf.linalg.diag_part(dot_product)
      distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
      distances = tf.maximum(distances, 0.0)
      return distances

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        imgs, labels = data
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        """ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)"""
        pairwise_dist = self.siamese_network(imgs)
  
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(labels)
        #mask = tf.to_float(mask)

        mask = tf.cast(mask, tf.float32)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
        #valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        #fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
        return triplet_loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
target_shape = (224, 224)
class resnet_triplets:
    def __init__(self, dataset_path, jpg_paths, dataset):
        # set class variables
        self.dataset_path = dataset_path
        self.resnet_feature_extractor = None
        self.train_dataset = None
        self.val_dataset = None

        # Now create triplets
        self.create_sets(jpg_paths, dataset)

        # First create network
        if (not os.path.isdir("./models/embeddings_batch_all")):
            print("create network")
            self.create_network()
        else:
            self.resnet_feature_extractor = tf.keras.models.load_model('./models/embeddings_batch_all')
            print("feature extractor loaded")

        self.get_reference(jpg_paths)


    def create_network(self):
        # Model creation with resnet as input and dense layers as output
        base_cnn = resnet.ResNet50(
            weights="imagenet", input_shape=target_shape + (3,), include_top=False, pooling="avg"
        )

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(2048, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(2048, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(2048)(dense2)

        # FIXME: embedding if now the feature extractor
        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable
        

        embedding = Model(base_cnn.input, output, name="Embedding")
        
        input = layers.Input(name="anchor", shape=target_shape + (3,))

        distances = DistanceLayer()(
            embedding(resnet.preprocess_input(input)),
        )

        siamese_network = Model(
            inputs=[input], outputs=distances
        )
        
        siamese_model = SiameseModel(siamese_network)
        siamese_model.compile(optimizer=tf.optimizers.Adam(0.0001))

        # TODO: we need to cache this
        siamese_model.fit(self.train_dataset, epochs=5, validation_data=self.val_dataset)

        # Save model
        embedding.save("./models/embeddings_batch_all")
        print("feature extractor saved")
        self.resnet_feature_extractor = embedding

    def create_sets(self, jpg_paths, dataset):
        self.jpg_paths = jpg_paths
       
        # load GT
        PATH_TO_GT = self.dataset_path + "/GT.json"
        gt_data = None
        with open(PATH_TO_GT, 'r') as in_gt:
            gt_data = json.load(in_gt)
        
        # Create path lists
        test_x_path, test_y, train_x_path, train_y = utils.create_sets_from_gt(gt_data, nb_queries = 1, dataset = utils.Dataset.INRIA)
    
        image_count = len(train_x_path)

        # Create dataset from path lists
        anchor_dataset = tf.data.Dataset.from_tensor_slices(train_x_path)
        labels_set = tf.data.Dataset.from_tensor_slices(train_y)

        dataset = tf.data.Dataset.zip((anchor_dataset, labels_set))

        dataset = dataset.shuffle(buffer_size=image_count)
        dataset = dataset.map(self.preprocess_image)

        # Let's now split our dataset in train and validation.
        train_dataset = dataset.take(round(image_count * 0.8))
        val_dataset = dataset.skip(round(image_count * 0.8))

        self.train_dataset = train_dataset.batch(256, drop_remainder=False)
        self.train_dataset.prefetch(8)

        self.val_dataset = val_dataset.batch(256, drop_remainder=False)
        self.val_dataset.prefetch(8)

    def preprocess_image(self, filename, label):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """
        print(filename, label)
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_shape)
        return (image, label)

    def preprocess_image_for_triplets(self, filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_shape)
        return image


    def preprocess_triplets(self, anchor, positive, negative):
        """
        Given the filenames corresponding to the three images, load and
        preprocess them.
        """

        return (
            self.preprocess_image_for_triplets(anchor),
            self.preprocess_image_for_triplets(positive),
            self.preprocess_image_for_triplets(negative),
        )
    
    def preprocess_image_for_embedding(self, filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """
        image = tf.keras.utils.load_img(filename)
        image = image.convert('RGB')
        image = image.resize((target_shape))
        x = np.array(image)
        x = preprocess_input(x)
        return x

    def get_reference(self, jpg_paths):
        self.jpg_paths = jpg_paths
        self.ref_embedding = self.get_embeddings(jpg_paths, 16, self.dataset_path + "/cache.npy")
        self.search_engine = NearestNeighbors()
        self.search_engine.fit(self.ref_embedding)

    def get_embeddings(self, jpg_paths, batch_size, cache=None):
        ref = None
        if (cache != None and os.path.isfile(cache)):
            with open(cache, "rb") as f:
                ref = np.load(f)
        else:
            ref = np.array([self.preprocess_image_for_embedding(path) for path in jpg_paths])
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

        if (distances is None or indices is None):
            print("\n   generating ...")
            query_vectors = self.get_embeddings(paths, 16, cache=None)
            distances, indices = self.search_engine.kneighbors(query_vectors, nb_neigh)
           
            with open(self.dataset_path + "/distances.npy", "wb") as f:
                np.save(f, distances)

            with open(self.dataset_path + "/indices.npy", "wb") as f:
                np.save(f, indices)
        
        return (distances, indices)

    def mAp_resnet(self, results, queries, reference, nb_neigh):
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
        #     print("query:", qname)
            # collect the positive results in the dataset
            # the positives have the same prefix as the query image
            print(gt_data[IMG_IDS[qimno]])
            positive_results = [imgid_to_index[img_id] for img_id in gt_data[IMG_IDS[qimno]]]
        #     print("positive_results:", positive_results)
        #     print("qres:", qres)
            #
            # ranks of positives. We skip the result #0, assumed to be the query image
            ranks = [i for i, res in enumerate(qres[1:]) if res in positive_results]
        #     print("ranks:", ranks)
            #
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
            aps.append(ap)

        print("mean AP = %.3f" % np.mean(aps))  # FIXME mean average precision

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask