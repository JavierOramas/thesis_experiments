from webui.main import app
from flask import render_template

def get_collections():
    ...

@app.route('/')
def home():
    return render_template('home.html', collections=get_collections())

