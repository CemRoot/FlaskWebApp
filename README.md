<p align="right"><a href="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.eduopinions.com%2Funiversities%2Funiversities-in-ireland%2Fnational-college-of-ireland%2F&psig=AOvVaw01SkHEeJDZ5G7fzDJPBY84&ust=1754854894165000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCMDO8Ia-_o4DFQAAAAAdAAAAABAM">National College of Ireland</a></p>

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
This returns the rendered `service.html` view. The app stores the uploaded image at `static/images/uploadedimage/input_image.jpeg` and displays the predicted label with confidence.

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
Bu istek `service.html` sayfasını döndürür. Yüklenen görüntü `static/images/uploadedimage/input_image.jpeg` olarak kaydedilir; tahmin sonucu ve güven değeri ekranda gösterilir.

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
