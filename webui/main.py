from flask import Flask

app = Flask(__name__)

# Import routes from separate files
from routers import home, collection

if __name__ == '__main__':
    app.run(debug=True)