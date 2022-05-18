from flask import Flask, render_template, g, jsonify, request, redirect, url_for, session, flash
#from bs4 import BeautifulSoup
import os
import flask
#import jyserver.Flask as jsf
from flask_cors import CORS
from flask import jsonify
from flask import request
from flask import current_app
import time
import json

import backend_src.pino as src

#input_text = ""
#output_text = {"text":"gasdas"}

app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")
#CORS(app)
#@jsf.use(app) # Connect Flask object to JyServer

@app.route('/')
def home():
    #if first_time == 1:
    #    print('pulisco')
    return render_template("index.html")

#@app.route('/get', methods =('GET',))
#def summarize():
    #print()
    #data = request.json
    #output_text=jsonify(data)
#    return(output_text)

@app.route('/set', methods =('POST',))
def send_text():
    data = request.get_json()
    #global output_text 
    #output_text=jsonify(data)
    #text_summary = src.pegasus(data['text'])
    html_ner = src.render_ner_analysis(data['text'])
    html_sentiment = src.render_sentiment_analysis(data['text'])
    text_summary = src.summarize(data['text'],data['model'])
    return({'text':text_summary, "html_ner":html_ner, "html_sentiment":html_sentiment})
    
#text_summary = src.render_pino(data['text'],data['model'])#


port = os.getenv('PORT', '5006')
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=int(port),debug=True)
    #serve(app, url_scheme='http', threads=4, port=int(port))