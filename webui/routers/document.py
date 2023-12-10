from flask import render_template, redirect, url_for
from webui.main import app  # Import the Flask app object

@app.route('/document/<int:document_id>')
def view_document(document_id):
    # Find the document by ID
    document = next((doc for doc in [] if doc['id'] == document_id), None)
    if document is not None:
        return render_template('document.html', document=document)
    else:
        return "Document not found."

