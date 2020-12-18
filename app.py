# This is a supervised machine learning model trained on tweets to make
# predictions whether a statement passed is a positive or  negative statement.
from flask import Flask, request, render_template, redirect
import gzip
import dill

app = Flask(__name__)


@app.route('/')
def main():
    return redirect("/index")


@app.route('/index', methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return "This is a Machine Learning model which finds predictions of words \n By finding the probability of a statement passed being positive. \n It is trained on tweets and its using the Naive_bayes Classfier."


@app.route('/predict', methods=['GET', "POST"])
def predict():
    if request.method == "GET":
        tweet = request.args.get("tweet")
    else:
        tweet = request.form["text"]

    with gzip.open('sentimental_mode.dill.gz', "rb") as f:
        model = dill.load(f)

    proba = model.predict_proba([tweet])[0, 1]

    return "Positive Sentiment: {}".format(proba)


if __name__ == '__main__':
    app.run()
