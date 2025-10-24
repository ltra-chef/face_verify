# DeepFace CLI – Face Verification & Face Search

A Python CLI tool built on [DeepFace](https://github.com/serengil/deepface) for:

* ✅ **Face verification** – check if two images are of the same person
* 🔍 **Face search** – find similar faces in a local image database

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/ltra-chef/face_verify.git
cd face_verify
```

### 2. Create and activate a Python virtual environment

```bash
# Create venv (Python 3.9+ recommended)
python3 -m venv .

# Activate on macOS/Linux
source bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## 🚀 Usage
# Note on first run a large model file will be downloaded 

### ✅ Verify Mode (Two-Image Comparison)

Check if two images belong to the same person.

```bash
python face_verify.py verify img1.jpg img2.jpg
```


**Example Output:**

```
--- DeepFace Verification Result ---
✅ The two images are of the same person.
Verified: True
Distance: 0.6727
Threshold: 0.6800
Model: VGG-Face
Detector: opencv
Metric: cosine
```

**Optional arguments:**

```bash
--model VGG-Face|Facenet|Facenet512|OpenFace|DeepFace|ArcFace
--detector opencv|ssd|dlib|mtcnn|retinaface|mediapipe
--metric cosine|euclidean|euclidean_l2
```

**Example:**

```bash
python face_verify.py verify img1.jpg img2.jpg --model ArcFace --detector mtcnn
```

---

### 🔍 Find Mode (Search in a Database)

Search for similar faces within a local folder (the "database").

```bash
python face_verify.py find query.jpg ./faces_db
```

**Example Output:**

```
--- DeepFace Find Results ---
               identity  VGG-Face_cosine
 ./faces_db/john_1.jpg         0.203145
 ./faces_db/john_2.jpg         0.214567
 ./faces_db/jane_1.jpg         0.625891
-----------------------------------
Annotated image saved to out.jpg
```

An annotated image (`out.jpg`) is generated with:

* Bounding boxes for detected faces
* Labels showing the top match

---

## 🧰 Building Your Face Database

DeepFace CLI expects a folder of images to serve as your face database.

### **Option 1 – Flat Structure**

All images in one folder:

```
faces_db/
├── john_1.jpg
├── john_2.jpg
├── mary_1.jpg
└── alex_1.jpg
```

### **Option 2 – Folder per Person (Recommended)**

Subfolders grouped by person name:

```
faces_db/
├── John/
│   ├── 1.jpg
│   └── 2.jpg
├── Mary/
│   ├── 1.jpg
│   └── 2.jpg
└── Alex/
    └── 1.jpg
```

DeepFace will automatically create embeddings for each image.
These embeddings are cached (see below) for faster lookups.

### **Tips for Accuracy**

* Use clear, frontal, well-lit photos.
* Include 2–3 images per person.
* Consistent file naming helps interpret labels (e.g., `John/1.jpg`).

---

## ⚙️ Embedding Caching & Performance Optimization

When you first run:

```bash
python face_verify.py find query.jpg ./faces_db
```

DeepFace will compute embeddings for all faces in `./faces_db`.
This can take time for large datasets (hundreds or thousands of images).

To speed up future searches, DeepFace **automatically caches** these embeddings in a file:

```
./faces_db/representations_<model>_<metric>.pkl
```

On subsequent runs, it reuses this cache to instantly perform face matching.

### 🧩 To Pre-Build the Cache

You can  precompute and store embeddings by running:

```bash
python face_verify.py build ./faces_db
```
Where the faces_db is the path to the database

This command creates the cache file without performing a search, which is useful for preparing large databases.

### 🧼 To Reset the Cache

Simply delete the `.pkl` file in your database directory:

```bash
rm ./faces_db/representations_*.pkl
```

---

## 🧪 Running Tests

Run automated tests to verify installation and functionality.

### 1. Install Pytest

```bash
pip install pytest
```

### 2. Run Tests

```bash
pytest -v
```

**Example Output:**

```
============================= test session starts =============================
collected 3 items

tests/test_face_verify.py::test_same_person PASSED                       [ 33%]
tests/test_face_verify.py::test_different_person PASSED                   [ 66%]
tests/test_face_verify.py::test_with_different_models[ArcFace] PASSED     [100%]
============================== 3 passed in 16.5s =============================
```

---


## 💡 Example Workflow

```bash
# 1. Setup
git clone https://github.com/<yourusername>/deepface-cli.git
cd deepface-cli
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Verify two images
python face_verify.py img1.jpg img2.jpg

# 3. Search for a match in your local face database
python face_verify.py find query.jpg ./faces_db

# 4. View annotated result
open find_output.jpg   # macOS
xdg-open find_output.jpg  # Linux
start find_output.jpg     # Windows
```

---

## 🧠 Supported Models

DeepFace CLI supports the following face recognition models:

| Model          | Speed     | Accuracy                 | Notes         |
| -------------- | --------- | ------------------------ | ------------- |
| **VGG-Face**   | ⚡ Fast    | ✅ Good                   | Default model |
| **Facenet**    | 🐢 Slower | 🔍 Very accurate         |               |
| **Facenet512** | 🐢 Slower | 🔍 High accuracy         |               |
| **ArcFace**    | ⚡ Fast    | ✅ Excellent accuracy     |               |
| **OpenFace**   | ⚡ Fast    | 🟡 Lightweight           |               |
| **DeepFace**   | ⚡ Fast    | 🟡 Basic reference model |               |


