import numpy as np
import tensorflow as tf
import sklearn
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications.resnet import preprocess_input
import os
import json
import glob
import utils
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras import constraints, initializers, regularizers
import tensorflow_addons as tfa


class ProxyAnchor(Layer):
    """
    Custom Keras Layer to calculate similarity between 
    """
    def __init__(self,
                 units,
                 **kwargs):
        super(ProxyAnchor, self).__init__(**kwargs)

        self.units = units
        self.kernel_init = initializers.get("he_normal")

    def build(self, input_shape):
        self.kernel = self.add_weight(
            dtype=tf.float32,
            shape=[input_shape[1], self.units],
            initializer=self.kernel_init, trainable=True)
        super(ProxyAnchor, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(tf.nn.l2_normalize(inputs, axis=1), tf.nn.l2_normalize(self.kernel, axis=0))


def proxy_anchor_loss(y_true, y_pred, margin=0.1, alpha=32):
    """
    Loss function use by this model
    """
    pos_one_hot = tf.one_hot(indices=y_true, depth=11, on_value=None, off_value=None)
    pos_one_hot = tf.squeeze(pos_one_hot, [1])
    neg_one_hot = 1.0 - pos_one_hot
    pos_exp = tf.exp(-alpha * (y_pred - margin))
    neg_exp = tf.exp(alpha * (y_pred + margin))
    pos_sum = tf.reduce_sum(pos_exp * pos_one_hot, axis=0)
    neg_sum = tf.reduce_sum(neg_exp * neg_one_hot, axis=0)
    pos_term = tf.reduce_sum(tf.math.log(1.0 + pos_sum)) / tf.math.count_nonzero(tf.reduce_sum(pos_one_hot, axis=0), dtype=tf.dtypes.float32)
    neg_term = tf.reduce_sum(tf.math.log(1.0 + neg_sum)) / 11
    return pos_term + neg_term

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image = tf.keras.utils.load_img(filename)
    image = image.convert('RGB')
    image = image.resize(((224, 224)))
    return preprocess_input(np.array(image))

class ResNet50_ProxyAnchor(tf.keras.Model):
    def __init__(self, nb_classes):
        super(ResNet50_ProxyAnchor, self).__init__()
        self.ResNet = resnet.ResNet50(
            weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling="avg"
        )
        trainable = False
        for layer in self.ResNet.layers:
            layer.trainable = trainable
        self.flat = layers.Flatten()
        self.fc = layers.Dense(2048, activation=None)
        self.proxy = ProxyAnchor(units=nb_classes)

    def call(self, inputs):
        x = self.ResNet(inputs)
        x = self.flat(x)
        x = self.fc(x)
        return self.proxy(x)

    def training(self, train_dataset, epochs=20, lr=0.0001):
        optimizer_model = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
        self.compile(loss=proxy_anchor_loss, optimizer=optimizer_model)
        self.build(input_shape=(None, 224, 224, 3))
        self.fit(train_dataset, epochs=epochs)
        print("Save model")
        
        flatten = layers.Flatten()(self.ResNet.output)
        output = self.fc(flatten)
        m = Model(self.ResNet.input, output)
        m.save("./models/anchors", overwrite=True, include_optimizer=False)

def mAp_resnet(results, queries, references, nb_neigh=9):
    PATH_TO_GT = "./static/Paris_buildings/GT.json"
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