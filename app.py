import os
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Buat folder jika belum ada
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model('model/modelcabai.h5')

# (Opsional) Kompilasi ulang model untuk menghindari warning metrics kosong
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Nama kelas
classes = ['Antraknosa', 'Busuk_Buah', 'Lalat_Buah']

# Deskripsi penyakit cabai
penyakit_cabai_descriptions = {
    'Antraknosa': 'Antraknosa adalah penyakit cabai setengah busuk menuju busuk.',
    'Busuk_Buah': 'Busuk Buah merupakan penyakit buah cabai dengan tingkat kebusukan 100%.',
    'Lalat_Buah': 'Lalat Buah merupakan penyakit buah cabai yang berbentuk rongga bolong-bolong.'
}

# Halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    description = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'

        file = request.files['image']
        if file.filename == '':
            return 'No selected file'

        # Validasi ekstensi file gambar
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return 'File harus berupa gambar (jpg/jpeg/png)'

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_path = filepath

            # Preprocessing gambar
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds[0])
            prediction = classes[predicted_class]

            # Ambil deskripsi penyakit
            description = penyakit_cabai_descriptions.get(prediction, "Deskripsi tidak tersedia.")

    return render_template('index.html', prediction=prediction, image_path=image_path, description=description)

# Jalankan server
if __name__ == '__main__':
    app.run(debug=True)