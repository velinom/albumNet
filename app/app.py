import flask
from PIL import Image

# Create the application.
APP = flask.Flask(__name__)


@APP.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return flask.render_template('index.html')


@APP.route('/generate')
def generate():
    return flask.send_file('static/CMBR.jpg', 'image/jpeg')


if __name__ == '__main__':
    APP.debug=True
    APP.run()