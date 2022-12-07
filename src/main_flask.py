from flask import Flask, redirect
from flask import render_template, request, url_for
import wtforms
import os
import utils
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input

template_dir = os.path.abspath('templates/')
instance_dir = os.path.abspath('static/upload/')
static_folder=os.path.abspath("static/")

app = Flask(__name__, template_folder=template_dir, instance_path=instance_dir, static_folder=static_folder)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

class IndexForm(FlaskForm):
    dataset = wtforms.fields.RadioField('Dataset', choices=[("holidays", "Inria_Holidays"), ("paris", "Paris6k")], validators=[wtforms.validators.InputRequired()])
    distance = wtforms.fields.RadioField('Distance', choices=[("euclidean", "Euclidean distance"), ("cosine", "Cosine distance")], validators=[wtforms.validators.InputRequired()])
    models = wtforms.fields.RadioField('Models', choices=[("baseline", "Baseline"), ("triplet", "Triplet Loss"), ("anchors", "Proxy Anchors")], validators=[wtforms.validators.InputRequired()])
    nb_queries = wtforms.fields.IntegerField("Number of results", default=10, validators=[wtforms.validators.InputRequired(), wtforms.validators.NumberRange(min=1, max=100)])
    query = wtforms.fields.FileField(validators=[])

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image = tf.keras.utils.load_img(filename)
    image = image.convert('RGB')
    image = image.resize(((224, 224)))
    return preprocess_input(np.array(image))

@app.route('/', methods=['POST', 'GET'])
def index():
    form = IndexForm(request.form)
    if request.method == "POST" and form.validate():
        f = request.files['query']
        path = ""
        filename = secure_filename(f.filename)
        if (filename != "" and filename is not None):
            path = os.path.join(app.instance_path, filename)
            image_data = request.files[form.query.name].read()
            open(str(path), 'wb').write(image_data)
            return redirect(url_for("query", dataset=form.dataset.data, model=form.models.data, nb_queries=form.nb_queries.data, distance=form.distance.data, filename=filename))
        return redirect(url_for("query", dataset=form.dataset.data, model=form.models.data, nb_queries=form.nb_queries.data, distance=form.distance.data))
    return render_template('index.html', form=form)

@app.route('/query/<dataset>/<model>/<int:nb_queries>/<distance>', defaults={'filename': None})
@app.route('/query/<dataset>/<model>/<int:nb_queries>/<distance>/<filename>')
def query(dataset, model, nb_queries, distance, filename):
    jpg_paths, m, cache_path = None, None, None
    if ((distance != "euclidean" and distance != "cosine")):
        redirect(url_for("index"))
    if (dataset == "holidays"):
        cache_path = "./cache/INRIA_paths.npy"
        if (os.path.isfile(cache_path)):
            with open(cache_path, "rb") as f:
                jpg_paths = np.load(f, allow_pickle=True)
        else:
            jpg_paths = utils.collect_INRIA_Holidays_paths("./static/INRIA_Holidays/")
        if (not os.path.isfile(cache_path)):
                with open(cache_path, "wb") as f:
                    np.save(f, np.array(jpg_paths))
        cache_path = "./cache/INRIA.npy"
    elif (dataset == "paris"):
        cache_path = "./cache/PARIS_paths.npy"
        if (os.path.isfile(cache_path)):
            with open(cache_path, "rb") as f:
                jpg_paths = np.load(f, allow_pickle=True)
        else:
            jpg_paths = utils.collect_Paris_buildings_paths("./static/Paris_buildings/")
        if (not os.path.isfile(cache_path)):
                with open(cache_path, "wb") as f:
                    np.save(f, np.array(jpg_paths))
        cache_path = "./cache/PARIS.npy"
    else:
        redirect(url_for("index"))
    if (model != "baseline" and model != "triplet" and model != "anchors"):
        redirect(url_for("index"))
    else:
        m = tf.keras.models.load_model("models/" + model, compile=False)
    # Search in cache for images
    ref = None
    if (os.path.isfile(cache_path)):
        with open(cache_path, "rb") as f:
            ref = np.load(f)
    else:
        ref = np.array([preprocess_image(path) for path in jpg_paths])
    if (not os.path.isfile(cache_path)):
            with open(cache_path, "wb") as f:
                np.save(f, ref)
    # Add preprocess images in cache
    dataset = tf.data.Dataset.from_tensor_slices(ref)
    dataset = dataset.batch(16, drop_remainder=False)
    # Get features (size of 2048)
    embeddings = m.predict(dataset, verbose=0)
    if (filename is None or not os.path.isfile("./static/upload/" + filename)):
        filename = jpg_paths[0]
    else:
        filename = "./static/upload/" + filename
    filename = str(filename)
    # Make a query
    search_engine = NearestNeighbors(metric=distance, algorithm='brute')
    search_engine.fit(embeddings)
    # Preprocess query image
    test_image = np.array([preprocess_image(filename)])
    test_dataset = tf.data.Dataset.from_tensor_slices(test_image)
    test_dataset = test_dataset.batch(16, drop_remainder=False)
    test_embeddings = m.predict(test_dataset, verbose=0)
    distances, indices = search_engine.kneighbors(test_embeddings, nb_queries)
    images_paths = [jpg_paths[i][9:] for i in indices[0]]
    return render_template("query.html", data={
        "images": images_paths, "query": filename[9:], "distances": list(distances[0])
    })


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(400)
def page_not_found(error):
    return render_template('400.html'), 400

@app.errorhandler(500)
def page_not_found(error):
    return render_template('500.html'), 500