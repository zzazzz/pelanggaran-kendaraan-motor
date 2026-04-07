import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TENSORFLOW"] = "1"
os.environ["USE_TORCH"] = "1"

from flask import Flask, request, render_template
from PIL import Image
import pandas as pd
import cv2
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = STATIC_DIR / "uploads"
IMAGES_DIR = STATIC_DIR / "images"
RUNS_DIR = STATIC_DIR / "runs" / "detect"
HISTORY_FILE = STATIC_DIR / "history.json"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# ── Load Models ───────────────────────────────────────────────────────────────
print("Loading TrOCR processor & model...")
processor = TrOCRProcessor.from_pretrained("ziyadazz/OCR-PLAT-NOMOR-INDONESIA")
model = VisionEncoderDecoderModel.from_pretrained("ziyadazz/OCR-PLAT-NOMOR-INDONESIA")

print("Loading YOLO models...")
model_driver = YOLO("best.pt")
model_object = YOLO("best2.pt")

LABEL_MAP = {
    0: "exp-date",
    1: "helm",
    2: "licence-plate",
    3: "no-helm",
}

LABEL_ORDER = ["exp-date", "helm", "licence-plate", "no-helm"]


# ── Helpers ──────────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    for p in [UPLOAD_DIR, IMAGES_DIR, RUNS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def rel_static(path: str) -> str:
    """Return a path relative to /static/ from either an absolute or relative input."""
    if not path:
        return path

    path = str(path).replace("\\", "/")
    marker = "/static/"
    idx = path.lower().rfind(marker)
    if idx != -1:
        return path[idx + len(marker):]
    if path.lower().startswith("static/"):
        return path[len("static/"):]
    if path.lower().startswith("./static/"):
        return path[len("./static/"):]
    return path.lstrip("/")
    if path.startswith("./static/"):
        return path[len("./static/"):]
    return path.lstrip("/")


def safe_read_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def safe_write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history(limit: int = 12) -> List[Dict[str, Any]]:
    history = safe_read_json(HISTORY_FILE)
    history = history[-limit:] if len(history) > limit else history
    history.reverse()
    return history


def append_history(entry: Dict[str, Any], max_items: int = 50) -> None:
    history = safe_read_json(HISTORY_FILE)
    history.append(entry)
    history = history[-max_items:]
    safe_write_json(HISTORY_FILE, history)


def check_driver_eligibility(df: pd.DataFrame) -> str:
    return "Pengendara tidak layak" if (df["cls_raw"] == 3).any() else "Pengendara layak di jalan"


def helm_deteksi(df: pd.DataFrame) -> str:
    return "Pengendara tidak menggunakan helm" if (df["cls_raw"] == 3).any() else "Pengendara menggunakan helm"


def balik_prediksi(prediksi):
    if isinstance(prediksi, str):
        cleaned = prediksi.replace(" ", "")
        if cleaned.isdigit() and len(cleaned) == 4:
            return cleaned[2:] + cleaned[:2]
    return prediksi


def cek_pajak(tanggal, formatted_date):
    return tanggal > formatted_date


def buat_kesimpulan(deteksi_helm, keterangan):
    no_helm = deteksi_helm == "Pengendara tidak menggunakan helm"
    pajak_mati = keterangan == "Pajak motor mati"
    if no_helm and pajak_mati:
        return "Tidak menggunakan helm & pajak motor mati"
    if no_helm:
        return "Tidak menggunakan helm"
    if pajak_mati:
        return "Pajak motor mati"
    return "Tidak melanggar aturan lalu lintas"


def kelayakan(pajak_ok, helm_ok):
    return "Pengendara Layak" if pajak_ok and helm_ok else "Pengendara Tidak Layak"


def class_label_from_row(row):
    cls = int(row["cls_raw"])
    return LABEL_MAP.get(cls, str(cls))


def class_text_from_label(label):
    return {
        "helm": "🪖 Helm",
        "no-helm": "⚠ No-Helm",
        "licence-plate": "🔢 Plat Nomor",
        "exp-date": "📅 Exp-Date",
    }.get(label, label)


def confidence_class(conf: float) -> str:
    if conf >= 0.7:
        return "success"
    if conf >= 0.5:
        return "warning"
    return "danger"


def crop_and_save_image(row, base_folder="static/images"):
    img = cv2.imread(row["cropped_image_paths"])
    if img is None:
        return row["cropped_image_paths"]

    x1, y1, x2, y2 = row["pred_box"]
    h, w = img.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))

    cropped_img = img[y1:y2, x1:x2]
    if cropped_img.size == 0:
        cropped_img = img

    folder = Path(base_folder) / row["group_label"]
    folder.mkdir(parents=True, exist_ok=True)
    cropped_img = cv2.resize(cropped_img, (384, 384), interpolation=cv2.INTER_AREA)
    save_path = folder / f"cropped_{row['group_label']}_{row.name}.jpg"
    cv2.imwrite(str(save_path), cropped_img)
    return str(save_path)


def get_detected_image_path(result, fallback_input_path: str) -> str:
    save_dir = Path(str(result.save_dir))
    fallback_name = Path(fallback_input_path).name

    # YOLO biasanya menyimpan file annotated dengan nama file input yang sama.
    guessed = save_dir / fallback_name
    if guessed.exists():
        return rel_static(str(guessed))

    candidates = []
    if save_dir.exists():
        candidates.extend(sorted(
            [p for p in save_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ))
    if candidates:
        return rel_static(str(candidates[0]))

    return rel_static(str(guessed))


def prepare_template_data(new_df: pd.DataFrame) -> pd.DataFrame:
    new_df = new_df.copy()
    new_df["group_label"] = new_df.apply(class_label_from_row, axis=1)
    new_df["group_text"] = new_df["group_label"].apply(class_text_from_label)
    new_df["cls_display"] = new_df["group_text"]
    new_df["ocr_text"] = "-"
    new_df["confidence_pct"] = (new_df["confidence"] * 100).round().astype(int)
    new_df["confidence_class"] = new_df["confidence"].apply(confidence_class)

    # OCR display: plat nomor dan exp-date
    filtered_df = (
        new_df[new_df["cls_raw"].isin([0, 2])]
        .sort_values("confidence", ascending=False)
        .groupby("cls_raw")
        .first()
        .reset_index()
    )

    if filtered_df.empty:
        new_df["kelayakan"] = "Tidak dapat diidentifikasi"
        new_df["jenis_pelanggaran"] = "-"
        return new_df

    ocr_results = []
    for img_path in filtered_df["cropped_image_saved_path"]:
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            text = "-"
        ocr_results.append(text)

    filtered_df["ocr_text"] = ocr_results

    # Simpan OCR ke new_df untuk plat & exp-date
    for _, row in filtered_df.iterrows():
        mask = new_df["cropped_image_saved_path"] == row["cropped_image_saved_path"]
        new_df.loc[mask, "ocr_text"] = row["ocr_text"]

        if int(row["cls_raw"]) == 2:
            new_df.loc[mask, "cls_display"] = f"🔢 Plat: {row['ocr_text']}"
        elif int(row["cls_raw"]) == 0:
            fixed_text = balik_prediksi(row["ocr_text"].replace(" ", ""))
            new_df.loc[mask, "ocr_text"] = fixed_text
            new_df.loc[mask, "cls_display"] = f"📅 Exp-Date: {fixed_text}"

    exp_mask = new_df["group_label"].eq("exp-date")
    if exp_mask.any():
        tanggal = new_df.loc[exp_mask, "ocr_text"].iloc[0].replace(" ", "")
        formatted_date = datetime.now().strftime("%y%m")
        pajak_ok = cek_pajak(tanggal, formatted_date)
        keterangan = "Pajak motor hidup" if pajak_ok else "Pajak motor mati"
    else:
        pajak_ok = True
        keterangan = "Exp-date tidak terdeteksi"

    hasil_pengecekan = check_driver_eligibility(new_df)
    deteksi_helm_str = helm_deteksi(new_df)
    helm_ok = hasil_pengecekan == "Pengendara layak di jalan"
    status_kelayakan = kelayakan(pajak_ok, helm_ok)
    jenis_pelanggaran = buat_kesimpulan(deteksi_helm_str, keterangan)

    new_df["kelayakan"] = status_kelayakan
    new_df["jenis_pelanggaran"] = jenis_pelanggaran
    return new_df


def get_class_summary_text(label: str, item: Dict[str, Any]) -> str:
    if not item:
        return "-"
    if label == "helm":
        return "Helm terdeteksi"
    if label == "no-helm":
        return "Tidak menggunakan helm"
    if label == "licence-plate":
        return item.get("ocr_text") if item.get("ocr_text") not in (None, "-") else "Plat nomor terdeteksi"
    if label == "exp-date":
        return item.get("cls_display") if item.get("cls_display") not in (None, "-") else item.get("ocr_text", "-")
    return item.get("cls_display") or item.get("ocr_text") or "-"


def build_class_groups(df: pd.DataFrame) -> List[Dict[str, Any]]:
    groups = []
    for label in LABEL_ORDER:
        subset = df[df["group_label"] == label].copy()
        items = []
        for _, row in subset.iterrows():
            items.append({
                "cropped_image_saved_path": rel_static(row["cropped_image_saved_path"]),
                "confidence_pct": int(row["confidence_pct"]),
                "confidence_class": row["confidence_class"],
                "cls_display": row["cls_display"],
                "ocr_text": row["ocr_text"],
                "kelayakan": row["kelayakan"],
                "pred_box": row["pred_box"],
                "group_text": row["group_text"],
            })

        primary_item = items[0] if items else None
        groups.append({
            "label": label,
            "title": class_text_from_label(label),
            "count": len(items),
            "items": items,
            "primary_item": primary_item,
            "summary_text": get_class_summary_text(label, primary_item or {}),
            "summary_confidence": primary_item["confidence_pct"] if primary_item else 0,
            "summary_kelayakan": primary_item["kelayakan"] if primary_item else "-",
            "summary_ocr": primary_item["ocr_text"] if primary_item else "-",
        })
    return groups


def build_summary_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {label: int((df["group_label"] == label).sum()) for label in LABEL_ORDER}


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index_view():
    ensure_dirs()
    return render_template("tengah.html", history=load_history())


@app.route("/predict", methods=["POST"])
def detect_object():
    ensure_dirs()
    try:
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("prediction.html", error="Tidak ada gambar yang diupload.")

        if not allowed_file(file.filename):
            return render_template("prediction.html", error="Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.")

        analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = secure_filename(file.filename)
        if not safe_name:
            safe_name = f"input_{analysis_id}.jpg"
        else:
            stem, ext = os.path.splitext(safe_name)
            safe_name = f"{stem}_{analysis_id}{ext.lower()}"

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        image_path = UPLOAD_DIR / safe_name
        file.save(str(image_path))

        # ── Stage 1: Deteksi pengendara ─────────────────────────────────────
        driver_project = RUNS_DIR
        driver_name = f"predict_{analysis_id}"
        results_driver = model_driver.predict(
            source=str(image_path),
            conf=0.5,
            save=True,
            project=str(driver_project),
            name=driver_name,
            exist_ok=True,
            verbose=False,
        )

        cropped_image_paths = []
        for idx, result in enumerate(results_driver):
            box_list = [
                [int(x) for x in box.xyxy[0].tolist()]
                for box in result.boxes
                if round(float(box.conf), 2) >= 0.5
            ]
            if not box_list:
                continue

            img = Image.open(str(image_path)).convert("RGB")
            cropped_img = img.crop(box_list[0])
            crop_path = IMAGES_DIR / f"cropped_image_{analysis_id}_{idx}.jpg"
            cropped_img.save(str(crop_path))
            cropped_image_paths.append(str(crop_path))

        if not cropped_image_paths:
            return render_template("prediction.html", error="Pengendara tidak terdeteksi dalam gambar.")

        # ── Stage 2: Deteksi objek ──────────────────────────────────────────
        object_project = RUNS_DIR
        object_name = f"object_{analysis_id}"
        results_obj = model_object.predict(
            source=cropped_image_paths,
            conf=0.52,
            save=True,
            project=str(object_project),
            name=object_name,
            exist_ok=True,
            verbose=False,
        )

        rows = []
        for idx, result in enumerate(results_obj):
            boxes = result.boxes
            best_box = {0: (None, 0), 2: (None, 0)}
            others = []

            for box in boxes:
                conf = round(float(box.conf), 2)
                cls = int(box.cls)
                if conf < 0.52:
                    continue

                if cls in [0, 2]:
                    if conf > best_box[cls][1]:
                        best_box[cls] = (box, conf)
                elif cls in [1, 3]:
                    others.append((box, conf, cls))

            all_boxes = others[:]
            for cls_id, (box, conf) in best_box.items():
                if box is not None:
                    all_boxes.append((box, conf, cls_id))

            annotated_path = get_detected_image_path(result, cropped_image_paths[idx])

            for box, conf, cls in all_boxes:
                box_data = [int(x) for x in box.data[0][:4]]
                rows.append(
                    {
                        "cropped_image_paths": cropped_image_paths[idx],
                        "pred_box": box_data,
                        "confidence": conf,
                        "cls_raw": cls,
                        "image_path": annotated_path,
                    }
                )

        if not rows:
            return render_template("prediction.html", error="Tidak ada objek yang terdeteksi.")

        new_df = pd.DataFrame(rows)
        new_df["group_label"] = new_df.apply(class_label_from_row, axis=1)

        # ── Stage 3: Crop & simpan setiap objek ─────────────────────────────
        new_df["cropped_image_saved_path"] = new_df.apply(crop_and_save_image, axis=1)

        # ── Stage 4: OCR & final verdict ───────────────────────────────────
        new_df = prepare_template_data(new_df)

        summary_counts = build_summary_counts(new_df)
        class_groups = build_class_groups(new_df)

        original_rel = rel_static(str(image_path))
        detected_rel = rel_static(new_df["image_path"].iloc[0])

        # ── History ────────────────────────────────────────────────────────
        append_history({
            "analysis_id": analysis_id,
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            "original_image": original_rel,
            "detected_image": detected_rel,
            "kelayakan": str(new_df["kelayakan"].iloc[0]),
            "jenis_pelanggaran": str(new_df["jenis_pelanggaran"].iloc[0]),
            "summary_counts": summary_counts,
            "total_objects": int(len(new_df)),
            "class_groups": class_groups,
        })

        return render_template(
            "web 2.html",
            new_df=new_df,
            original_image_path=original_rel,
            detected_image_path=detected_rel,
            summary_counts=summary_counts,
            class_groups=class_groups,
            analysis_id=analysis_id,
        )

    except Exception as e:
        traceback.print_exc()
        return render_template("prediction.html", error=f"Terjadi kesalahan: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)