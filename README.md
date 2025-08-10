<p align="right"><a href="https://www.ncirl.ie/">National College of Ireland</a></p>

<p align="right">
  <a href="https://www.ncirl.ie/agent-portal/international-brand-guidelines">
    <img src="https://www.ncirl.ie/Portals/0/International/Marketing/NCI_Logo_colour.png" alt="National College of Ireland logo" width="180" />
  </a>
  
</p>


## Deepfake Detection Web Application (Flask)

This repository contains a Flask web app for classifying images as Real or Fake using a TensorFlow/Keras model with an attention block. It offers a simple web UI and an HTTP endpoint.

### TL;DR Quick Start
```bash
# 0) Prereqs: macOS 12+, Python 3.11.x

# 1) Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Add model assets (required to predict)
mkdir -p model
# Copy your trained files into ./model/
#  - best_model_effatt.h5
#  - label_transform.pkl

# 4) Run the app
python app.py  # opens http://127.0.0.1:5000/
```

### What it does
- Upload an image from the UI (`/service`) or send an HTTP POST (`/deepfake`).
- Validates file type (PNG/JPG/JPEG) and resizes to 128×128.
- Runs inference with a Keras model that includes an attention block and GAP rescaling.
- Maps predicted index to label using `model/label_transform.pkl` (falls back to ["Fake", "Real"]).
- Saves uploads under `static/images/uploadedimage/` with unique filenames.

### Repository Structure
```
FlaskWebApp/
  app.py                      # Flask server and inference logic
  requirements.txt            # Python dependencies
  model/                      # Place model files here (not tracked)
    best_model_effatt.h5      # Trained model weights (required)
    label_transform.pkl       # Label encoder (required)
  notebooks/                  # Jupyter notebooks used in the thesis (read-only)
  static/                     # Frontend assets (CSS/JS/images)
    images/uploadedimage/     # Upload destination (gitignored)
  templates/                  # Jinja2 templates (index.html, service.html)
  .gitignore                  # Excludes datasets, models, envs, upload outputs
```

### Usage
- Web UI: open `http://127.0.0.1:5000/service`, choose an image (PNG/JPG/JPEG), and submit.
- Programmatic (HTTP):
```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://127.0.0.1:5000/deepfake
```
This returns the rendered `service.html` view. The app stores the uploaded image under `static/images/uploadedimage/` with a unique filename (e.g., `upload_<timestamp>_<uuid>.jpeg`) and displays the predicted label with confidence.

### How inference works
- Preprocessing: OpenCV read → resize to 128×128 → float32 array
- Model: H5 model loaded with `custom_objects`:
  - `attention_block(features, depth)`
  - `RescaleGAP([gap_feat, gap_mask]) = gap_feat / gap_mask`
- Labels: from `model/label_transform.pkl` if present; otherwise `["Fake", "Real"]`.

### Notes
- Datasets, models, `.venv`, and upload outputs are ignored by git.
- If model/labels are missing, the server runs but cannot predict.
- On Apple Silicon, TensorFlow 2.18.0 requires macOS 12+ and Python 3.11.

### Troubleshooting (quick)

- Python must be 3.11.x. Create the venv inside `FlaskWebApp/`:
  ```bash
  cd FlaskWebApp
  python3.11 -m venv .venv && source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- VS Code import errors (Pylance):
  - Cmd+Shift+P → Python: Select Interpreter → `FlaskWebApp/.venv/bin/python`
  - Cmd+Shift+P → Python: Restart Language Server → (if needed) Developer: Reload Window
- Model load error “bad marshal data (unknown type code)”:
  - Likely wrong Python version/dependency mismatch. Recreate the venv with Python 3.11. The app first attempts to load the full H5 model; if that fails, it rebuilds the architecture in code and loads weights only.
- Port 5000 in use:
  ```bash
  lsof -ti :5000 | xargs kill -9
  # or run on another port
  python -c "from app import app; app.run(debug=True, use_reloader=False, port=5001)"
  ```
- Conda conflicts:
  ```bash
  conda deactivate || true
  source FlaskWebApp/.venv/bin/activate
  ```
- Same image appears repeatedly:
  - The app saves with unique filenames and appends a cache-busting query param. If needed, hard-refresh the page (Cmd+Shift+R).

### Notebooks
The `notebooks/` folder contains the Jupyter notebooks used during experimentation and thesis work. They are not required to run the Flask app but are included for reference and reproducibility.

### Academic Use
If you reference this application in your thesis or publications, consider citing your model and training methodology (e.g., EfficientNet variant + attention) and include a short description of preprocessing and evaluation protocol.

---

## Türkçe Bölüm (Özet ve Kurulum)

Bu bölüm, uygulamanın Türkçe özeti ve hızlı kurulum talimatlarını içerir.

### Hızlı Başlangıç
```bash
# 1) Sanal ortam oluşturun ve etkinleştirin
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Bağımlılıkları yükleyin
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Model dosyalarını ekleyin (./model/ altına)
mkdir -p model
#  - best_model_effatt.h5
#  - label_transform.pkl

# 4) Uygulamayı çalıştırın
python app.py  # http://127.0.0.1:5000/
```

### Ne Yapar?
- Arayüzden (`/service`) ya da HTTP POST (`/deepfake`) ile görüntü yüklenir.
- Dosya türü doğrulanır (PNG/JPG/JPEG) ve 128×128 boyutuna getirilir.
- Dikkat (attention) bloğu içeren Keras modeli ile tahmin yapılır.
- `model/label_transform.pkl` varsa etiket eşleme buradan yapılır; yoksa varsayılan sıra ["Fake", "Real"] kullanılır.
- Yüklenen görseller `static/images/uploadedimage/` altına benzersiz adlarla kaydedilir.

### Depo Yapısı
```
FlaskWebApp/
  app.py
  requirements.txt
  model/
    best_model_effatt.h5
    label_transform.pkl
  notebooks/
  static/images/uploadedimage/
  templates/
```

### Çıkarım Ayrıntıları
- OpenCV okuma → 128×128 yeniden boyutlandırma → float32 dizi
- Model H5 formatında yüklenir; `custom_objects` içinde:
  - `attention_block(features, depth)`
  - `RescaleGAP([gap_feat, gap_mask]) = gap_feat / gap_mask`

### Sorun Giderme (Kısa)
- Python sürümü 3.11 olmalı. Venv’i bu sürümle oluşturun ve `requirements.txt` kurun.
- VS Code import hataları: Cmd+Shift+P → Python: Select Interpreter → `FlaskWebApp/.venv/bin/python`, ardından Python: Restart Language Server.
- "bad marshal data" hatası: Yanlış Python/bağımlılık. Venv’i 3.11 ile yeniden kurun. Uygulama, tam modeli yükleyemezse mimariyi koddan kurup sadece ağırlıkları yükler.
- 5000 portu dolu: `lsof -ti :5000 | xargs kill -9` ya da farklı portta çalıştırın.
- Aynı görsel sorunu: Uygulama benzersiz adlarla kaydediyor; yine de gerekirse sert yenileme (Cmd+Shift+R).

### References
1. Agrawal, D.R., Haneef, F., 2025. Eye Blinking Feature Processing Using Convolutional Generative Adversarial Network for Deep Fake Video Detection. Trans. Emerg. Telecommun. Technol. 36, e70083. https://doi.org/10.1002/ett.70083
2. Al-Khazraji, S.H., Saleh, H.H., Khalid, A.I., Mishkhal, I.A., 2023. Impact of Deepfake Technology on Social Media: Detection, Misinformation and Societal Implications. Eurasia Proc. Sci. Technol. Eng. Math. 23, 429–441. https://doi.org/10.55549/epstem.1371792
3. Borji, A., 2023. Qualitative Failures of Image Generation Models and Their Application in Detecting Deepfakes. https://doi.org/10.48550/ARXIV.2304.06470
4. Goceri, E., 2023. Medical image data augmentation: techniques, comparisons and interpretations. Artif. Intell. Rev. 56, 12561–12605. https://doi.org/10.1007/s10462-023-10453-z
5. Heidari, A., Jafari Navimipour, N., Dag, H., Unal, M., 2024. Deepfake detection using deep learning methods: A systematic and comprehensive review. WIREs Data Min. Knowl. Discov. 14, e1520. https://doi.org/10.1002/widm.1520
6. Hernandez Aros, L., Bustamante Molano, L.X., Gutierrez Portela, F., Moreno Hernández, J.J., Rodríguez Barrero, M.S., 2023. Detection of financial fraud by applying ML techniques a RSL. https://doi.org/10.7910/DVN/CM8NVY
7. Jung, T., Kim, S., Kim, K., 2020. DeepVision: Deepfakes Detection Using Human Eye Blinking Pattern. IEEE Access 8, 83144–83154. https://doi.org/10.1109/ACCESS.2020.2988660
8. Lal, K., Saini, M.L., 2023. A study on deep fake identification techniques using deep learning. Presented at the RECENT ADVANCES IN SCIENCES, ENGINEERING, INFORMATION TECHNOLOGY & MANAGEMENT, Jaipur, India, p. 020155. https://doi.org/10.1063/5.0154828
9. Malik, A., Kuribayashi, M., Abdullahi, S.M., Khan, A.N., 2022. DeepFake Detection for Human Face Images and Videos: A Survey. IEEE Access 10, 18757–18775. https://doi.org/10.1109/ACCESS.2022.3151186
10. Masood, M., Nawaz, M., Malik, K.M., Javed, A., Irtaza, A., 2021. Deepfakes Generation and Detection: State-of-the-art, open challenges, countermeasures, and way forward. https://doi.org/10.48550/ARXIV.2103.00484
11. Mujawar, S.M., -, H.M.K., -, Y.S.P., -, K.S.K., -, A.P.T., 2025. Exploring AI/ML Techniques for Deepfake Detection: A Comprehensive Review. Int. J. Sci. Technol. 16, 3843. https://doi.org/10.71097/IJSAT.v16.i2.3843
12. Pishori, A., Rollins, B., Houten, N. van, Chatwani, N., Uraimov, O., 2020. Detecting Deepfake Videos: An Analysis of Three Techniques. https://doi.org/10.48550/arXiv.2007.08517
13. Rana, M.S., Nobi, M.N., Murali, B., Sung, A.H., 2022. Deepfake Detection: A Systematic Literature Review. IEEE Access 10, 25494–25513. https://doi.org/10.1109/ACCESS.2022.3154404
14. Stankov, I.S., Dulgerov, E.E., 2024. Detection of Deepfake Images and Videos Using SVM, CNN, and Hybrid Approaches, in: 2024 XXXIII International Scientific Conference Electronics (ET). Presented at the 2024 XXXIII International Scientific Conference Electronics (ET), IEEE, Sozopol, Bulgaria, pp. 1–5. https://doi.org/10.1109/ET63133.2024.10721497
15. Stroebel, L., Llewellyn, M., Hartley, T., Ip, T.S., Ahmed, M., 2023. A systematic literature review on the effectiveness of deepfake detection techniques. J. Cyber Secur. Technol. 7, 83–113. https://doi.org/10.1080/23742917.2023.2192888
