from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR
import io
from PIL import Image
import os
from tempfile import NamedTemporaryFile

app = Flask(__name__)

# Initialize OCR (3.0 params: lang='hi' same, use_textline_orientation=True for angle cls)
# Models auto-download on first run (~100MB one-time, Hindi supported in multilingual)
ocr = PaddleOCR(
    use_angle_cls=True,  # Still works, but in 3.0 it's use_textline_orientation internally
    lang='hi',  # Hindi/Devanagari
    use_gpu=False,  # CPU for free tier
    show_log=False  # Quiet mode (3.0 mein logging changed)
)

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
        
        # Temp file for OCR
        with NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name
        
        # Run OCR (3.0: use predict() instead of ocr())
        result = ocr.predict(temp_path)
        
        # Extract text from new format (list of res objects, each with .res['rec_texts'] and .res['rec_scores'])
        recognized_text = []
        confidences = []
        for res_obj in result:
            res = res_obj.res  # Dict with 'rec_texts', 'rec_scores'
            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            recognized_text.extend(texts)
            confidences.extend(scores)
        
        full_text = ' '.join([t for t in recognized_text if t.strip()])  # Clean empty
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
