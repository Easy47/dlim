from flask import Flask
from flask import render_template
import wtforms
import os
from flask_wtf import FlaskForm

template_dir = os.path.abspath('templates/')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

class IndexForm(FlaskForm):
    dataset = wtforms.fields.RadioField('Dataset', choices=["Inria Holidays", "Paris6k"], validators=[wtforms.validators.DataRequired()])
    distance = wtforms.fields.RadioField('Distance', choices=["Euclidean distance", "Cosine distance"], validators=[wtforms.validators.DataRequired()])
    query = wtforms.fields.FileField(validators=[])
    
@app.route('/')
def index():
    form = IndexForm()
    return render_template('index.html', form=form)

@app.route('/query')
def query():
    
    return 