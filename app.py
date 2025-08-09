from flask import Flask, request, render_template, url_for
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf  # Core TensorFlow
import keras
from typing import Optional, Protocol, Any, runtime_checkable, cast
from werkzeug.utils import secure_filename
import time

@runtime_checkable
class PredictModel(Protocol):
    def predict(self, x: np.ndarray) -> Any: ...

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model: Optional[PredictModel] = None


############################## MODEL LOADING ##############################################


# Attention block function
def attention_block(features, depth):
    attn = keras.layers.Conv2D(256, (1, 1), padding='same', activation='relu')(keras.layers.Dropout(0.5)(features))
    attn = keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(attn)
    attn = keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(attn)
    attn = keras.layers.Conv2D(1, (1, 1), padding='valid', activation='sigmoid')(attn)

    up = keras.layers.Conv2D(depth, (1, 1), padding='same', activation='linear', use_bias=False)
    up_w = np.ones((1, 1, 1, depth), dtype=np.float32) # Initialize with ones
    up.build((None, None, None, 1)) # Build with dummy input shape
    up.set_weights([up_w]) # Set weights
    up.trainable = True # Make trainable

    attn = up(attn) # Expand attention map to match feature dimensions
    masked = keras.layers.Multiply()([attn, features]) # Apply attention by element-wise multiplication

    gap_feat = keras.layers.GlobalAveragePooling2D()(masked)
    gap_mask = keras.layers.GlobalAveragePooling2D()(attn)
    gap = keras.layers.Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_feat, gap_mask])
    return gap




############################################################################################


# Path configurations
inputpath='static/images/uploadedimage'
inputimagepath = './static/images/uploadedimage/input_image.jpeg'
outputimagepath = './static/images/uploadedimage/output_image.jpeg'
image_size = 128
default_image_size =  tuple((128, 128))

def is_allowed_file(filename: str) -> bool:
    if not filename or '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resolve_label_name(index: int) -> str:
    try:
        if 'class_labels' in globals() and class_labels is not None:
            if hasattr(class_labels, 'classes_'):
                classes = list(class_labels.classes_)
                return str(classes[index])
            if isinstance(class_labels, (list, tuple, np.ndarray, pd.Series)):
                classes = list(class_labels)
                return str(classes[index])
    except Exception as e:
        print(f"[LABEL_RESOLVE_ERROR] {e}")
    # Fallback ordering commonly used in this project
    fallback = ['Fake', 'Real']
    if 0 <= index < len(fallback):
        return fallback[index]
    return str(index)

try:
    labels_path = os.path.join('model', 'label_transform.pkl')
    if not os.path.exists(labels_path):
        labels_path = os.path.join(os.path.dirname(__file__), 'model', 'label_transform.pkl')
    class_labels = pd.read_pickle(labels_path)
    print(getattr(class_labels, 'classes_', None))
except Exception as e:
    print(f"Error loading labels: {e}")
    class_labels = None
try:
    # model=load_model('model/my_model_plain.h5')

        # Input size
    # inputShape = (128, 128, 3)

    # # Load EfficientNetB7 base
    # base_model = EfficientNetB7(include_top=False, weights=None, input_shape=inputShape) # Load EfficientNetB7 without top layer
    # base_model.trainable = True # Make base model trainable

    # # Attention-enhanced features
    # features = base_model.output # Extract output features
    # bn_features = BatchNormalization()(features) # Apply batch normalization
    # pt_depth = base_model.output_shape[-1] # Get depth of feature maps
    # gap = attention_block(bn_features, pt_depth) # Apply attention block

    # # Classification head (regularized)
    # x = Dropout(0.5)(gap) # Dropout for regularization
    # x = Dense(64, activation='relu', kernel_regularizer=l2(0.00001))(x) # Dense layer with L2 regularization
    # x = Dropout(0.25)(x) # Dropout
    # output = Dense(2, activation='softmax')(x) # Output layer for binary classification

    # # Final model
    # model = keras.Model(inputs=base_model.input, outputs=output) # Create final model
    # model.load_weights(r'model\best_model_effatt.h5')
    # model.compile(
    #     loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer-labeled classification
    #     optimizer='adam',                         # Use Adam optimizer for adaptive learning rate optimization
    #     metrics=['accuracy']                      # Evaluate model performance based on accuracy
    # )
    model_path = os.path.join('model', 'best_model_effatt.h5')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model_effatt.h5')

    # Define Lambda replacement at top-level to avoid marshal issues across Python versions
    def RescaleGAP(tensors):
        return tensors[0] / tensors[1]

    # Prefer tf.keras loader for legacy H5 models; provide custom_objects to bypass marshaled code
    model = cast(PredictModel, tf.keras.models.load_model(
        model_path,
        custom_objects={
            'attention_block': attention_block,
            'RescaleGAP': RescaleGAP
        },
        compile=False
    ))
    print('Model Loaded successfully via keras!')

except Exception as e:
    print(f"Error loading Model: {e}")
    model = None


def load_image_array(filepath: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(filepath)
        if img is None:
            return None
        resized = cv2.resize(img, default_image_size)
        # Keras utils img_to_array expects HWC and returns float32 array
        arr = keras.utils.img_to_array(resized)
        return arr
    except Exception as e:
        print(f"Error - {e}")
        return None

           
@app.route('/')
def index(): 
  return render_template('index.html')

@app.route('/home')
def home(): 
  return render_template('index.html')

@app.route('/service')
def service():
  return render_template('service.html',outputvalues='nonoutput')

@app.route('/deepfake', methods=['POST'])
def deep_fake():
    if 'file' not in request.files:
        return render_template('service.html', error="No file uploaded")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('service.html', error="No file selected")
    filename_for_check = file.filename or ""
    if not is_allowed_file(filename_for_check):
        return render_template('service.html', error="Unsupported file type. Please upload a PNG or JPG image.")
    
    
    if model is None:
        return render_template('service.html', error="Model not loaded")
    
    if class_labels is None:
        return render_template('service.html', error="Class labels not loaded")
    
    try:
        # inputimagepath = os.path.join(inputimagepath)
        filename: str = secure_filename(file.filename or "uploaded.jpeg")
        # Always save to fixed analysis path
        abs_save_path = os.path.abspath(inputimagepath)
        os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)
        file.save(abs_save_path)
        file_size_kb = 0
        try:
            file_size_kb = int(os.path.getsize(abs_save_path) / 1024)
        except Exception:
            pass
        print(f"[UPLOAD] Saved to: {abs_save_path} ({file_size_kb} KB)")

        img = load_image_array(abs_save_path)
        if img is None:
            return render_template('service.html', error="Invalid image uploaded")
        print(f"[IMAGE] shape={getattr(img, 'shape', None)} dtype={getattr(img, 'dtype', None)}")
        if hasattr(img, 'ndim') and img.ndim == 3:
            img_batch = np.expand_dims(img, axis=0)
        else:
            img_batch = img
        prediction = model.predict(img_batch)
        print(f"[PREDICT] raw={prediction}")
        probs = np.array(prediction).squeeze()
        if isinstance(probs, list):
            probs = np.array(probs)
        probs = probs.astype(float)
        if probs.ndim > 1:
            # Take first row if batch dimension present
            probs = probs[0]
        answer_idx = int(np.argmax(probs))
        pred_label = resolve_label_name(answer_idx)
        confidence = float(probs[answer_idx])
        print(f"Predicted: {pred_label} ({confidence*100:.1f}%)")
        
        
        return render_template(
            'service.html',
            pred_class=pred_label,
            confidence=f"{confidence*100:.1f}%",
            image_url=url_for('static', filename='images/uploadedimage/input_image.jpeg') + f"?t={int(time.time())}",
            outputvalues='output'
        )
    
    except Exception as e:
      print(f"[ERROR] {e}")
      return render_template('service.html', error=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(inputimagepath), exist_ok=True)
    # Disable Flask reloader to avoid watching .venv leading to infinite reloads
    app.run(debug=True, use_reloader=False)