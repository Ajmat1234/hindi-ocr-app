from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR
import io
from PIL import Image
import os
from tempfile import NamedTemporaryFile
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Initialize OCR with Hindi support
try:
    app.logger.info("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False, show_log=False)
    app.logger.info("PaddleOCR initialized successfully.")
except Exception as init_err:
    app.logger.error(f"PaddleOCR init failed: {init_err}")
    ocr = None  # Will handle in predict

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if ocr is None:
        return jsonify({'error': 'OCR not initialized. Please restart the app.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        app.logger.info(f"Processing file: {file.filename}")
        
        # Read image bytes
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty file uploaded'}), 400
        
        app.logger.info("Image loaded successfully.")
        image = Image.open(io.BytesIO(image_bytes))
        
        # Create temp file for OCR (PaddleOCR requires file path)
        temp_path = None
        with NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG', quality=95)
            temp_path = temp_file.name
            app.logger.info(f"Temp file created: {temp_path}")
        
        # Run OCR
        app.logger.info("Running OCR...")
        result = ocr.ocr(temp_path, cls=True)
        app.logger.info(f"OCR result: {result}")
        
        # Parse results
        recognized_text = []
        confidences = []
        if result and result[0]:  # List of detections
            for line in result[0]:
                if line and len(line) > 1 and line[1]:
                    text = line[1][0]  # Text
                    conf = line[1][1] if len(line[1]) > 1 else 0.0  # Confidence
                    if text and text.strip():
                        recognized_text.append(text.strip())
                        confidences.append(conf)
        
        full_text = ' '.join(recognized_text)
        avg_conf = sum(confidences) / max(1, len(confidences)) if confidences else 0.0
        
        app.logger.info(f"Extracted text: {full_text}")
        app.logger.info(f"Average confidence: {avg_conf:.2f}")
        
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            app.logger.info("Temp file cleaned up.")
        
        return jsonify({
            'recognized_text': full_text,
            'confidence': f'{avg_conf:.2f}'
        })
        
    except Exception as e:
        app.logger.error(f"Predict error: {str(e)}", exc_info=True)
        # Ensure cleanup even on error
        if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
