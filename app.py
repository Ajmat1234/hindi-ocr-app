from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR
import io
from PIL import Image
import os
from tempfile import NamedTemporaryFile
import logging
import threading  # For init lock

# Minimal logging (INFO for key events only)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Global OCR with lazy init lock
ocr = None
ocr_lock = threading.Lock()

def init_ocr():
    global ocr
    with ocr_lock:
        if ocr is None:
            try:
                logger.info("Lazy-loading PaddleOCR (first request)...")
                ocr = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False, show_log=False)
                logger.info("PaddleOCR loaded successfully.")
            except Exception as init_err:
                logger.error(f"OCR init failed: {str(init_err)}", exc_info=True)
                ocr = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lazy init on first call
    init_ocr()
    if ocr is None:
        return jsonify({'error': 'OCR setup failed (memory issue?). Try again or check logs.'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    temp_path = None
    try:
        logger.info(f"Processing: {file.filename}")
        
        # Read & resize image (key for mem savings)
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty image'}), 400
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize to max 800px (aspect preserved, reduces mem 50-70%)
        max_dim = 800
        if image.width > max_dim or image.height > max_dim:
            ratio = min(max_dim / image.width, max_dim / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized to: {new_size}")
        
        logger.info(f"Image ready: {image.size}")
        
        # Temp file
        with NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG', quality=85)  # Lower quality for mem
            temp_path = temp_file.name
        
        # OCR run
        logger.info("Running OCR...")
        result = ocr.ocr(temp_path, cls=True)
        logger.info(f"Detections: {len(result[0]) if result and result[0] else 0}")
        
        # Parse
        recognized_text = []
        confidences = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) > 1 and line[1]:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.0
                    if text.strip():
                        recognized_text.append(text.strip())
                        confidences.append(float(conf))
        
        full_text = ' '.join(recognized_text)
        avg_conf = sum(confidences) / max(1, len(confidences)) if confidences else 0.0
        
        logger.info(f"Result: '{full_text[:50]}...' (conf: {avg_conf:.2f})")
        
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return jsonify({
            'recognized_text': full_text,
            'confidence': f'{avg_conf:.2f}',
            'num_lines': len(recognized_text)
        })
        
    except MemoryError:
        logger.error("Out of memory during OCR")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': 'Low memory - try smaller image or upgrade plan.'}), 503
    except Exception as e:
        logger.error(f"Predict error: {str(e)}", exc_info=True)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': f'Failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
