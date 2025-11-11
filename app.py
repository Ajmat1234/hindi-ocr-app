from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR
import io
from PIL import Image

app = Flask(__name__)

# Initialize OCR once (Hindi lang, angle classification for rotated text)
# First run will download models (~50MB total, one-time)
ocr = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False)  # 'hi' for Hindi/Devanagari

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
        # Read image as bytes
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temp for OCR (PaddleOCR needs file path or np array)
        from tempfile import NamedTemporaryFile
        import os
        with NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name
        
        # Run OCR
        result = ocr.ocr(temp_path, cls=True)
        
        # Extract text
        recognized_text = []
        for line in result:
            if line:  # Check if not empty
                for word_info in line:
                    recognized_text.append(word_info[1][0])  # Text part
        
        full_text = ' '.join(recognized_text)
        confidence = sum([word_info[1][1] for line in result if line for word_info in line]) / max(1, len(recognized_text)) if recognized_text else 0
        
        # Cleanup temp file
        os.unlink(temp_path)
        
        return jsonify({
            'recognized_text': full_text,
            'confidence': f'{confidence:.2f}' if confidence else '0.00'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
