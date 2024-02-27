from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import test
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file_path = os.path.abspath(filename)
        file.save(file_path)
        
        plot_buf, text = test.generate_plot(file_path)
        plot_buf.seek(0)

        # Then read its contents
        image_bytes = plot_buf.read()

        # Encode these bytes in base64
        encoded_image = base64.b64encode(image_bytes)

        # Decode the base64 bytes to string
        encoded_string = encoded_image.decode('utf-8')

        # Now create the appropriate data URL
        data_url = f"data:image/png;base64,{encoded_string}"

        # This is what you should send back to the frontend
        response = {
            'image': data_url,
            'text': text
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
