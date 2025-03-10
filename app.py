from flask import Flask, render_template, request, jsonify
from retrieval_bot import ChatBot
import warnings

warnings.filterwarnings("ignore") 

retirieval_chat = ChatBot()
retirieval_chat.load()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_chat_response(input)


def get_chat_response(text):

    reply = retirieval_chat.generate_response(text)
    return reply


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
