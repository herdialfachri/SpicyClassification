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
    'Antraknosa': 'Antraknosa adalah penyakit pada cabai yang disebabkan oleh jamur Colletotrichum spp.. Penyakit ini menyerang buah cabai dan menimbulkan bercak cekung berwarna coklat kehitaman. Jamur ini berkembang pesat pada kondisi lembap dan menyebabkan buah membusuk hingga rontok.',
    'Busuk_Buah': 'Busuk cabai adalah kondisi pembusukan buah yang disebabkan oleh jamur atau bakteri, seperti Phytophthora capsici. Penyakit ini muncul terutama pada kondisi lembap, membuat buah menjadi lunak, basah, dan berwarna hitam, sehingga tidak layak konsumsi.',
    'Lalat_Buah': 'Lalat buah merupakan hama dari jenis serangga, biasanya dari genus Bactrocera, yang meletakkan telurnya di dalam buah cabai. Setelah menetas, larva memakan daging buah dari dalam hingga buah membusuk, berlubang, dan jatuh sebelum matang.'
}

# Halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    description = None
    confidence = None

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

            # Tambahkan di route index setelah memprediksi
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds[0])
            prediction = classes[predicted_class]

            # Hitung persentase prediksi
            confidence = float(preds[0][predicted_class]) * 100  # Ubah ke persen

            # Ambil deskripsi penyakit
            description = penyakit_cabai_descriptions.get(prediction, "Deskripsi tidak tersedia.")

    return render_template('index.html', prediction=prediction, image_path=image_path, description=description, confidence=confidence)

# Jalankan server
if __name__ == '__main__':
    app.run(debug=True)