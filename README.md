<p align="right"><a href="https://www.ncirl.ie/">National College of Ireland</a></p>

<p align="right">
  <a href="https://www.ncirl.ie/agent-portal/international-brand-guidelines">
    <img src="https://www.ncirl.ie/Portals/0/International/Marketing/NCI_Logo_colour.png" alt="National College of Ireland logo" width="180" />
  </a>
  
</p>


## Deepfake Detection Web Application (Flask)

A production-ready Flask web application for image-based deepfake detection developed as part of a thesis project. The app provides a simple web UI and a form-data HTTP endpoint to classify an input image as Real or Fake using a TensorFlow/Keras model with an attention mechanism.

### Key Features
- Image upload via UI (`/service`) or HTTP POST (`/deepfake`)
- Input validation (PNG/JPG/JPEG), automatic resizing to 128×128
- TensorFlow/Keras inference with custom attention block and GAP rescaling
- Label mapping via `model/label_transform.pkl` (fallback to ["Fake", "Real"]) 
- Clean error handling and deterministic save path for uploads

### Repository Structure
```
FlaskWebApp/
  app.py                     # Flask server and inference logic
  requirements.txt           # Python dependencies
  model/                     # Place model files here (not tracked)
    best_model_effatt.h5     # Trained model weights (required)
    label_transform.pkl      # Label encoder (required)
  static/                    # Frontend assets (CSS/JS/images)
    images/uploadedimage/    # Upload destination (gitignored)
  templates/                 # Jinja2 templates (index.html, service.html)
  .gitignore                 # Excludes datasets, models, envs, large binaries
```

### Prerequisites
- macOS 12+ (Apple Silicon supported)
- Python 3.11.x
- Recommended: virtual environment

### Setup
```bash
# 1) Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Add model assets (required to run inference)
# Place files under ./model/
#  - best_model_effatt.h5
#  - label_transform.pkl

# 4) Run the application
python app.py
# App will start at http://127.0.0.1:5000/
```

### Usage
- Web UI: Open `http://127.0.0.1:5000/service`, choose an image (PNG/JPG/JPEG), and submit.
- Programmatic (HTTP):
```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://127.0.0.1:5000/deepfake
```
This returns the rendered `service.html` view. The app stores the uploaded image under `static/images/uploadedimage/` with a unique filename (e.g., `upload_<timestamp>_<uuid>.jpeg`) and displays the predicted label with confidence.

### Inference Details
- Preprocessing: OpenCV read → resize to 128×128 → convert to float32 array
- Model: TensorFlow/Keras model (H5) loaded with `custom_objects` including:
  - `attention_block(features, depth)`
  - `RescaleGAP([gap_feat, gap_mask]) = gap_feat / gap_mask`
- Labels: Loaded from `model/label_transform.pkl` when available. Fallback order: `["Fake", "Real"]`.

### Notes
- Large artifacts like datasets, model binaries, `.venv`, and upload outputs are excluded via `.gitignore`.
- If model or label files are missing, the app will start but will not be able to produce predictions.
- For Apple Silicon: the pinned TensorFlow (2.18.0) supports macOS 12+. If you face install issues, ensure you are on Python 3.11 and latest pip.

### Troubleshooting (Setup/Run)

- Python version: use Python 3.11.x only. Create a clean venv in the `FlaskWebApp/` folder.
  ```bash
  cd FlaskWebApp
  python3.11 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- VS Code cannot resolve imports (Pylance):
  - Command Palette → "Python: Select Interpreter" → choose `FlaskWebApp/.venv/bin/python`
  - Command Palette → "Python: Restart Language Server"
  - If needed, reload window: "Developer: Reload Window"
- Model fails to load with "bad marshal data (unknown type code)":
  - Likely wrong Python version or mismatched TF/Keras. Recreate venv with Python 3.11 and reinstall requirements. The app first tries to load the full H5; if it fails it rebuilds the architecture and loads only the weights automatically.
- Port 5000 already in use:
  ```bash
  lsof -ti :5000 | xargs kill -9  # free the port on macOS
  # or run on another port:
  python -c "from app import app; app.run(debug=True, use_reloader=False, port=5001)"
  ```
- Conda conflicts with venv: deactivate conda before activating `.venv`.
  ```bash
  conda deactivate || true
  source FlaskWebApp/.venv/bin/activate
  ```
- Image preview shows an old image: the app saves with unique filenames and adds a cache-busting query param; hard refresh (Cmd+Shift+R) if needed.

### Academic Use
If you reference this application in your thesis or publications, consider citing your model and training methodology (e.g., EfficientNet variant + attention) and include a short description of preprocessing and evaluation protocol.

---

## Derin Sahte (Deepfake) Tespit Web Uygulaması (Flask)

Bu depo, bir tez çalışması kapsamında geliştirilen, görüntü tabanlı derin sahte tespiti yapan üretim seviyesinde bir Flask web uygulamasını içerir. Uygulama, basit bir web arayüzü ve form-data HTTP uç noktası ile bir görüntüyü Gerçek (Real) veya Sahte (Fake) olarak sınıflandırır. Model, TensorFlow/Keras ve dikkat (attention) mekanizması kullanır.

### Öne Çıkan Özellikler
- Arayüz (`/service`) veya HTTP POST (`/deepfake`) ile görüntü yükleme
- Girdi doğrulama (PNG/JPG/JPEG), otomatik 128×128 yeniden boyutlandırma
- Dikkat bloğu ve GAP yeniden ölçekleme ile TensorFlow/Keras çıkarım
- `model/label_transform.pkl` ile etiket eşleme (["Fake", "Real"] geri dönüş değeri)
- Temiz hata yönetimi ve belirli bir yükleme kayıt yolu

### Depo Yapısı
```
FlaskWebApp/
  app.py                     # Flask sunucu ve çıkarım mantığı
  requirements.txt           # Python bağımlılıkları
  model/                     # Model dosyaları burada (takip dışı)
    best_model_effatt.h5     # Eğitilmiş model ağırlıkları (gerekli)
    label_transform.pkl      # Etiket kodlayıcı (gerekli)
  static/                    # Arayüz varlıkları (CSS/JS/görseller)
    images/uploadedimage/    # Yükleme klasörü (gitignore)
  templates/                 # Jinja2 şablonları (index.html, service.html)
  .gitignore                 # Veri, model, ortam ve büyük dosyaları hariç tutar
```

### Gereksinimler
- macOS 12+ (Apple Silicon desteklenir)
- Python 3.11.x
- Öneri: sanal ortam kullanımı

### Kurulum
```bash
# 1) Sanal ortam oluşturma ve etkinleştirme
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Bağımlılıkların kurulumu
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Model dosyalarını ekleme (çıkarım için zorunlu)
# ./model/ altına kopyalayın:
#  - best_model_effatt.h5
#  - label_transform.pkl

# 4) Uygulamayı çalıştırma
python app.py
# Uygulama http://127.0.0.1:5000/ adresinde çalışır
```

### Kullanım
- Web Arayüzü: `http://127.0.0.1:5000/service` adresini açın, bir resim (PNG/JPG/JPEG) seçin ve gönderin.
- Programatik (HTTP):
```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://127.0.0.1:5000/deepfake
```
Bu istek `service.html` sayfasını döndürür. Yüklenen görüntü `static/images/uploadedimage/` altında benzersiz bir adla (örn. `upload_<timestamp>_<uuid>.jpeg`) kaydedilir; tahmin sınıfı ve güven değeri ekranda gösterilir.

### Çıkarım Ayrıntıları
- Ön İşleme: OpenCV okuma → 128×128 yeniden boyutlandırma → float32 diziye çevirme
- Model: TensorFlow/Keras (H5) + `custom_objects`:
  - `attention_block(features, depth)`
  - `RescaleGAP([gap_feat, gap_mask]) = gap_feat / gap_mask`
- Etiketler: Varsayılan olarak `["Fake", "Real"]`; mevcutsa `model/label_transform.pkl` kullanılır.

### Notlar
- Veri setleri, model ikili dosyaları, `.venv` ve yükleme çıktıları `.gitignore` ile hariç tutulmuştur.
- Model veya etiket dosyaları eksikse uygulama tahmin üretemez.
- Apple Silicon için: Sabitlenmiş TensorFlow (2.18.0) macOS 12+ ile uyumludur. Kurulum hatası yaşarsanız Python 3.11 ve güncel pip kullandığınızdan emin olun.

### Akademik Kullanım
Bu uygulamayı tezinizde veya yayınlarınızda referans gösterecekseniz, model ve eğitim metodolojinizi (örn. EfficientNet türevi + attention) ve ön işleme/evaluasyon ayrıntılarınızı kısaca belirtmeniz önerilir.
