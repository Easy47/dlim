import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
import utils
import os


target_shape = (224, 224)
class resnetdlim:
    def __init__(self):
        self.ResNet = resnet.ResNet50(
            weights="imagenet", include_top=False,  input_shape=target_shape + (3,), pooling="avg"
        )
        output = layers.Flatten()(self.ResNet.output)
        self.resnet_feature_extractor = Model(self.ResNet.input, output, name="Embedding")

    def get_reference(self, paths):
        self.paths = paths
        self.ref_embedding = self.get_embeddings(paths, 32, "INRIA_Holidays/cache.npy")
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

    def get_embeddings(self, paths, batch_size, cache=None):
        ref = None
        if (cache != None and os.path.isfile(cache)):
            with open(cache, "rb") as f:
                ref = np.load(f)
        else:
            ref = np.array([self.preprocess_image(path) for path in paths])
        if (cache != None and not os.path.isfile(cache)):
            with open(cache, "wb") as f:
                np.save(f, ref)
        train_dataset = tf.data.Dataset.from_tensor_slices((ref))
        train_dataset = train_dataset.batch(batch_size)
        ref_embedding = self.resnet_feature_extractor.predict(train_dataset)
        return ref_embedding

    def execute_query(self, paths):
        query_vectors = self.get_embeddings(paths, 32, None)
        distances, indices = self.search_engine.kneighbors(query_vectors, 9)
        return (distances, indices)