from flask import Flask, request, jsonify, render_template
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import numpy as np
import io
import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import nest_asyncio
from pyngrok import ngrok
from datetime import datetime

app = Flask(__name__)

# If you want to start from the pretrained model, load the checkpoint with `VisionEncoderDecoderModel`
processor = TrOCRProcessor.from_pretrained('ziyadazz/OCR-PLAT-NOMOR-INDONESIA')

# TrOCR is a decoder model and should be used within a VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_pretrained('ziyadazz/OCR-PLAT-NOMOR-INDONESIA')

model_driver = YOLO("best.pt")
model_object = YOLO("best2.pt")  # Ganti path dengan model yang sesuai

@app.route('/')
def index_view():
    return render_template('tengah.html')

@app.route('/predict', methods=['POST'])
def detect_object():
    # Get the image file from the request
    file = request.files['image']

    # Save the file temporarily (optional)
    image_path = 'Pengendara.jpg'
    file.save(image_path)

    results = model_driver.predict(image_path)

    all_box_list = []
    all_conf_list = []
    all_cls_list = []
    cropped_image_paths = []  # Menyimpan path hasil cropping

    for idx, result in enumerate(results):
        boxes = result.boxes
        box_list = []
        conf_list = []
        cls_list = []

        for box in boxes:
            conf = round(float(box.conf), 2)
            cls = int(box.cls)

            if conf >= 0.5:
                box_data = [int(x) for x in box.xyxy[0].tolist()]
                box_list.append(box_data)
                conf_list.append(conf)
                cls_list.append(cls)

        all_box_list.append(box_list)
        all_conf_list.append(conf_list)
        all_cls_list.append(cls_list)

        # Menyimpan hasil cropping dalam format JPG
        img = Image.open(image_path)
        cropped_img = img.crop(box_list[0])  # Ambil kotak pertama
        cropped_image_path = f'static/images/cropped_image_{idx}.jpg'
        cropped_img.save(cropped_image_path)
        cropped_image_paths.append(cropped_image_path)

    data = {
        'image_original': [file.filename] * len(all_box_list),
        'boxes': all_box_list,
        'confidence': all_conf_list,
        'classes': all_cls_list,
        'cropped_image_paths': cropped_image_paths  # Menambahkan path hasil cropping ke dalam data
    }

    df = pd.DataFrame(data)

    results = model_object.predict(list(df['cropped_image_paths']),save=True)

    all_box_list = []
    all_conf_list = []
    all_cls_list = []
    rows = []

    for result in results:
        boxes = result.boxes
        cls_list = []
        box_list = []
        conf_list = []

        max_confidence_box_0 = None
        max_confidence_box_2 = None
        max_confidence_0 = 0
        max_confidence_2 = 0

        for box in boxes:
            conf = round(float(box.conf), 2)
            cls = round(float(box.cls), 2)

            if conf >= 0.2:
                if cls == 0 and conf > max_confidence_0:
                    max_confidence_0 = conf
                    max_confidence_box_0 = box
                elif cls == 2 and conf > max_confidence_2:
                    max_confidence_2 = conf
                    max_confidence_box_2 = box
                elif cls in [1, 3]:
                    cls_list.append(cls)
                    conf_list.append(conf)
                    box_data = box.data[0][:4]
                    box_data = [int(x) for x in box_data]
                    box_list.append(box_data)

        if max_confidence_box_0 is not None:
            box_data = max_confidence_box_0.data[0][:4]
            box_data = [int(x) for x in box_data]
            cls_list.append(0)
            conf_list.append(max_confidence_0)
            box_list.append(box_data)

        if max_confidence_box_2 is not None:
            box_data = max_confidence_box_2.data[0][:4]
            box_data = [int(x) for x in box_data]
            cls_list.append(2)
            conf_list.append(max_confidence_2)
            box_list.append(box_data)

        all_box_list.append(box_list)
        all_conf_list.append(conf_list)
        all_cls_list.append(cls_list)

    df["pred_box"] = all_box_list
    df["confidence"] = all_conf_list
    df['cls'] = all_cls_list

    rows = []
    for idx, row in df.iterrows():
        image_path = row['cropped_image_paths']
        pred_boxes = row['pred_box']
        confidences = row['confidence']
        classes = row['cls']

        # Loop untuk setiap prediksi dalam satu baris
        for i in range(len(pred_boxes)):
            rows.append({
                "cropped_image_paths": image_path,
                "pred_box": pred_boxes[i],
                "confidence": confidences[i],
                "cls": classes[i],
                "image_path":results[0].save_dir
            })
    new_df = pd.DataFrame(rows)

    def crop_and_save_image(row):
        img = cv2.imread(row['cropped_image_paths'])
        pred_box = row['pred_box']
        cropped_img = img[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2]]

        # Mendapatkan label untuk nama file
        label_mapping = {0: 'exp-date', 1: 'helm', 2: 'licence-plate', 3: 'no-helm'}
        label = label_mapping[row['cls']]

        # Buat folder sesuai dengan label jika belum ada
        folder_name = f'static/images/{label}'  # Ganti 'images' dengan nama folder yang diinginkan
        os.makedirs(folder_name, exist_ok=True)

        # Resize semua gambar menjadi 384x384
        cropped_img = cv2.resize(cropped_img, (384, 384), interpolation=cv2.INTER_AREA)

        # Simpan gambar yang sudah dipotong dalam format JPG sesuai dengan label dan folder
        cropped_image_path = f'{folder_name}/cropped_{label}_{row.name}.jpg'
        cv2.imwrite(cropped_image_path, cropped_img)

        return cropped_image_path

    new_df['cropped_image_saved_path'] = new_df.apply(crop_and_save_image, axis=1)
    directory = new_df['image_path'].iloc[0]  # Ganti dengan path lengkap ke direktori Anda

    # Mendapatkan daftar file dalam direktori
    files = os.listdir(directory)

    # Filter hanya file gambar (jpg/png)
    image_files = [file for file in files if file.lower().endswith(('jpg', 'png'))]

    # Buka dan tampilkan setiap gambar dalam direktori
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        new_df['image_path']=image_path
    def check_driver_eligibility(new_df):
        if (new_df['cls'] == 3).any():
            return "Pengendara tidak layak"
        else:
            return "Pengendara layak di jalan"
    
    hasil_pengecekan = check_driver_eligibility(new_df)
    
    def helm_deteksi(new_df):
        if (new_df['cls'] == 3).any():
            return "Pengendara tidak menggunakan helm"
        else:
            return "Pengendara menggunakan helm"

    deteksi_helm = helm_deteksi(new_df)
    filtered_df = new_df[new_df['cls'].isin([0.0, 2.0])]
    filtered_df = filtered_df.groupby('cls').apply(lambda x: x.loc[x['confidence'].idxmax()]).reset_index(drop=True)

    if filtered_df.empty:
        new_df['kelayakan']='Pengendara tidak bisa diidentifikasi'
        return render_template('prediction.html', new_df=new_df)
    else:
        pred = []

        for imges_path in filtered_df['cropped_image_saved_path']:
            image = Image.open(imges_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred.append(generated_text)

        filtered_df['cls'] = pred

        for index, row in filtered_df.iterrows():
            img_path = row['cropped_image_saved_path']
            new_value = row['cls']  # Mengambil nilai 'cls' dari filtered_df
            new_df.loc[new_df['cropped_image_saved_path'] == img_path, 'cls'] = new_value

        new_df.loc[new_df['cropped_image_saved_path'].str.contains('static/images/exp-date/'), 'cls']= new_df.loc[new_df['cropped_image_saved_path'].str.contains('static/images/exp-date/'), 'cls'].str.replace(' ', '')

        # Definisikan kembali fungsi balik_prediksi
        def balik_prediksi(prediksi):
            if isinstance(prediksi, str) and prediksi.isdigit() and len(prediksi) == 4:
                return prediksi[2:] + prediksi[:2]
            else:
                return prediksi

        # Mengaplikasikan fungsi pada kolom 'Prediksi' dan membuat kolom baru 'Prediksi_Balik'
        tanggal = new_df.loc[new_df['cropped_image_saved_path'].str.contains('static/images/exp-date/'), 'cls'].apply(balik_prediksi)
        tanggal = tanggal.iloc[0]
        today_date = datetime.now().date()
        formatted_date = today_date.strftime("%y%m")

        def pajak(tanggal, formatted_date):
            if (tanggal > formatted_date):
                return 'Pengendara layak di jalan'
            else:
                return 'Pengendara tidak layak'

        exp_date=pajak(tanggal, formatted_date)

        def pajak_motor(tanggal, formatted_date):
            if (tanggal > formatted_date):
                return 'Pajak motor hidup'
            else:
                return 'Pajak motor mati'

        keterangan=pajak_motor(tanggal, formatted_date)

        def kelayakan(exp_date, hasil_pengecekan):
            if exp_date=='Pengendara layak di jalan' and hasil_pengecekan == 'Pengendara layak di jalan':
                return 'Pengendara Layak'
            elif exp_date=='Pengendara tidak layak' and hasil_pengecekan == 'Pengendara layak di jalan':
                return 'Pengendara tidak layak'
            elif exp_date=='Pengendara layak di jalan' and hasil_pengecekan == 'Pengendara tidak layak':
                return 'Pengendara tidak layak'
            elif exp_date=='Pengendara tidak layak' and hasil_pengecekan == 'Pengendara tidak layak':
                return 'Pengendara tidak layak'
            else:
                return 'Kondisi tidak tau'

        def buat_kesimpulan(deteksi_helm, keterangan):
            if deteksi_helm == "Pengendara tidak menggunakan helm" and keterangan == "Pajak motor mati":
                return "Pengendara tidak menggunakan helm dan pajak motor mati"
            elif deteksi_helm == "Pengendara tidak menggunakan helm" and keterangan == "Pajak motor hidup":
                return "Pengendara tidak menggunakan helm"
            elif deteksi_helm == "Pengendara menggunakan helm" and keterangan == "Pajak motor mati":
                return "Pajak motor mati"
            elif deteksi_helm == "Pengendara menggunakan helm" and keterangan == "Pajak motor hidup":
                return "Pengendara tidak melanggar aturan lalu lintas"
            else:
                return "Kondisi tidak terdefinisi"
            
        new_df['jenis_pelanggaran']=buat_kesimpulan(deteksi_helm, keterangan)                    
        new_df['kelayakan']=kelayakan(exp_date, hasil_pengecekan)                
        new_df=new_df

    # Return the DataFrame as a JSON response
    return render_template('web 2.html', new_df=new_df, df=df)

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
app.run(host="0.0.0.0", port=8000)
    
