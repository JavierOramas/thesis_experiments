from webui.main import app
from flask import render_template

@app.route('/collection/<int:collection_id>')
def view_collection(collection_id):
    
    # TODO Find the collection
    collection = next((c for c in [] if c['id'] == collection_id), None)
    if collection is not None:
        return render_template('collection.html', collection=collection)
    else:
        return "Collection not found."