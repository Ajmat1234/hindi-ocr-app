from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR
import io
from PIL import Image
import os
from tempfile import NamedTemporaryFile

app = Flask(__name__)

# Initialize OCR (stable 2.8.1: lang='hi' for Hindi, use_angle_cls for rotation)
ocr = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False, show_log=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Temp file for OCR (PaddleOCR needs path)
        with NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name
        
        # Run OCR (2.8.1 API: ocr.ocr() returns list of [line, (text, conf)])
        result = ocr.ocr(temp_path, cls=True)
        
        # Extract text and conf
        recognized_text = []
        confidences = []
        if result and result[0]:  # result[0] is list of detections
            for line in result[0]:
                text = line[1][0]  # Recognized text
                conf = line[1][1]  # Confidence
                if text.strip():
                    recognized_text.append(text)
                    confidences.append(conf)
        
        full_text = ' '.join(recognized_text)
        avg_conf = sum(confidences) / max(1, len(confidences)) if confidences else 0
        
        # Cleanup
        os.unlink(temp_path)
        
        return jsonify({
            'recognized_text': full_text,
            'confidence': f'{avg_conf:.2f}' if avg_conf else '0.00'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
