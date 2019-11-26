from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# import HashingVectorizer from local dir
from vectorizer import vect

app = Flask(__name__)

## Preparing the classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects/classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')


def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = clf.predict_proba(X).max()
    return label[y], proba


def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])


# def sqlite_entry(path, document, y):


class HelloForm(Form):
    sayhello = TextAreaField('', [validators.DataRequired()])


@app.route('/')
def index():
    form = HelloForm(request.form)
    return render_template('index.html', form=form)


@app.route('/hello', methods=['POST'])
def hello():
    form = HelloForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['sayhello']
        return render_template('hello.html', name=name)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)