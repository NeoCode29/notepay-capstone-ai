"""
Synthetic data generator untuk training CRNN OCR.

Membuat gambar struk sintetis dari teks label — berbeda dari augmentasi
(yang memodifikasi gambar yang sudah ada), script ini merender teks baru
di atas background kertas thermal menggunakan Pillow.

Strategi dual-source:
  1. Label dari labels.csv (re-render dengan variasi baru)
  2. Teks struk baru yang digenerate secara random per kelas

Jalankan di WSL2 dengan venv-tf:
    python ai/synthetic_data.py
    python ai/synthetic_data.py --count 3 --random-count 200
    python ai/synthetic_data.py --src data/ocr_dataset --dst data/ocr_dataset
    python ai/synthetic_data.py --classes line_item total_belanja
    python ai/synthetic_data.py --only-random --random-count 500
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print("ERROR: Pillow tidak terinstall. Jalankan: pip install pillow")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    OCR_DATASET_DIR, OCR_LABELS_CSV,
    CROP_HEIGHT, CROP_WIDTH, CHARACTERS, YOLO_CLASSES,
)

VALID_CHARS = set(CHARACTERS)

# ---------------------------------------------------------------------------
# Pencarian font system
# ---------------------------------------------------------------------------

FONT_SEARCH_PATHS = [
    # WSL2 / Ubuntu
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    # Windows (jika dijalankan native)
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/lucon.ttf",
    "C:/Windows/Fonts/consola.ttf",
]


def find_fonts():
    """Cari semua font monospace yang tersedia di sistem."""
    found = [p for p in FONT_SEARCH_PATHS if os.path.exists(p)]
    return found if found else []


# ---------------------------------------------------------------------------
# Generator teks struk realistis per kelas
# ---------------------------------------------------------------------------

NAMA_TOKO_TEMPLATES = [
    "INDOMARET", "ALFAMART", "ALFAMIDI", "CIRCLE K", "LAWSON",
    "HERO", "GIANT", "SUPERINDO", "HYPERMART", "LOTTEMART",
    "TRANSMART", "CARREFOUR", "ISLAND MARKET", "FRESH MARKET",
    "MINIMARKET {}", "TOKO SEMBAKO {}", "WARUNG MAKAN {}",
    "APOTEK {}", "KLINIK {}", "BENGKEL {}",
    "RESTO {}", "CAFE {}", "BAKSO {}", "WARTEG {}",
]

NAMA_RANDOM = [
    "MAJU JAYA", "SEJAHTERA", "MAKMUR", "SUMBER REJEKI",
    "ABADI", "BERKAH", "SENTOSA", "MANDIRI", "USAHA BARU",
    "KARYA BERSAMA", "MULYA", "CAHAYA", "BINTANG", "HARAPAN",
    "JAYA RAYA", "PUTRA PERDANA", "UTAMA", "PRIMA",
]

SATUAN = ["pcs", "kg", "gr", "ltr", "ml", "btl", "dus", "pak", "bks", "lbr"]

ITEM_TEMPLATES = [
    "{item}",
    "{item} {qty}x",
    "{item} @{price}",
    "{qty} {item}",
    "{item} {qty}{sat}",
    "{item:<20} {price}",
]

ITEM_NAMES = [
    "AQUA 600ML", "INDOMIE GORENG", "POCARI SWEAT", "TEH BOTOL", "SPRITE",
    "COCA COLA", "FANTA", "GOOD DAY", "CHITATO", "PRINGLES",
    "BENG-BENG", "SILVER QUEEN", "KIT KAT", "OREO", "RICHEESE",
    "SABUN LIFEBUOY", "SHAMPO PANTENE", "PASTA GIGI", "TISU PASEO",
    "MIE SEDAAP", "ABC SARDINES", "SARDEN", "BERAS 5KG", "MINYAK GORENG",
    "GULA PASIR", "TELUR AYAM", "SUSU UHT", "KOPI KAPAL API",
    "NESCAFE 3IN1", "ENERGEN", "QUAKER OATS", "MILO",
    "ROTI TAWAR", "KERUPUK", "SNACK", "BISCUIT", "WAFER",
    "KECAP ABC", "SAUS SAMBAL", "TERASI", "GARAM", "MERICA",
    "DETERJEN RINSO", "SOFTENER DOWNY", "VIM", "WIPOL",
    "LILIN", "KOREK API", "BATERAI", "KANTONG PLASTIK",
]

TOTAL_TEMPLATES = [
    "TOTAL           Rp {total}",
    "TOTAL BELANJA   Rp {total}",
    "GRAND TOTAL     Rp {total}",
    "SUBTOTAL        Rp {total}",
    "TOTAL BAYAR     Rp {total}",
    "JUMLAH          Rp {total}",
    "TOTAL :         Rp {total}",
    "TOTAL PEMBAYARAN Rp {total}",
    "TTL             {total}",
    "Total   {total}",
]

TANGGAL_TEMPLATES = [
    "{d:02d}/{m:02d}/{y} {h:02d}:{mn:02d}",
    "{d:02d}-{m:02d}-{y} {h:02d}:{mn:02d}:{s:02d}",
    "{d:02d}/{m:02d}/{y}",
    "TGL: {d:02d}/{m:02d}/{y}",
    "Tgl {d:02d}/{m:02d}/{y} {h:02d}.{mn:02d}",
    "{d:02d}/{m:02d}/{y} Pkl {h:02d}:{mn:02d}",
    "KASIR: {kasir}  {d:02d}/{m:02d}/{y}",
    "{d:02d}/{m:02d}/{y}  {h:02d}:{mn:02d} WIB",
    "No Trx: {trx}  {d:02d}/{m:02d}/{y}",
]

KASIR_NAMES = ["ANI", "BUDI", "CITRA", "DIAN", "EKO", "FANI", "GITA"]


def _fmt_rupiah(n):
    """Format angka ke format harga struk: 12.500 atau 125.000"""
    s = f"{n:,}".replace(",", ".")
    return s


def _random_price(min_val=500, max_val=500_000, step=500):
    val = random.randrange(min_val, max_val + step, step)
    return _fmt_rupiah(val)


def _filter_chars(text):
    """Buang karakter di luar charset CRNN."""
    return "".join(c for c in text if c in VALID_CHARS).strip()


def generate_nama_toko():
    tmpl = random.choice(NAMA_TOKO_TEMPLATES)
    if "{}" in tmpl:
        name = random.choice(NAMA_RANDOM)
        text = tmpl.format(name)
    else:
        text = tmpl
    return _filter_chars(text)


def generate_line_item():
    item = random.choice(ITEM_NAMES)
    tmpl = random.choice(ITEM_TEMPLATES)
    qty = random.randint(1, 5)
    sat = random.choice(SATUAN)
    price = _random_price(500, 50_000)
    text = tmpl.format(item=item, qty=qty, sat=sat, price=price)
    return _filter_chars(text)


def generate_total_belanja():
    tmpl = random.choice(TOTAL_TEMPLATES)
    total = _random_price(5_000, 1_000_000, 1_000)
    text = tmpl.format(total=total)
    return _filter_chars(text)


def generate_tanggal_waktu():
    d  = random.randint(1, 28)
    m  = random.randint(1, 12)
    y  = random.randint(2022, 2026)
    h  = random.randint(6, 22)
    mn = random.randint(0, 59)
    s  = random.randint(0, 59)
    kasir = random.choice(KASIR_NAMES)
    trx = f"{random.randint(1000, 9999)}"
    tmpl = random.choice(TANGGAL_TEMPLATES)
    text = tmpl.format(d=d, m=m, y=y, h=h, mn=mn, s=s, kasir=kasir, trx=trx)
    return _filter_chars(text)


CLASS_GENERATORS = {
    "nama_toko"     : generate_nama_toko,
    "line_item"     : generate_line_item,
    "total_belanja" : generate_total_belanja,
    "tanggal_waktu" : generate_tanggal_waktu,
}


# ---------------------------------------------------------------------------
# Render gambar sintetis
# ---------------------------------------------------------------------------

def _make_thermal_background(h, w):
    """Background kertas thermal: putih + noise sedikit."""
    bg = np.full((h, w), 245, dtype=np.uint8)
    noise = np.random.normal(0, random.uniform(2, 8), (h, w)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 220, 255).astype(np.uint8)
    return bg


def _load_font(font_paths, size):
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_text_image(text, font_paths, out_h=CROP_HEIGHT, out_w=CROP_WIDTH):
    """
    Render `text` ke gambar grayscale (out_h × out_w) seperti crop struk.
    Variasi: ukuran font, posisi vertikal, spacing, warna teks.
    """
    # Pilih ukuran font — thermal receipt biasanya 8–16px pada tinggi 32px
    font_size = random.randint(max(8, out_h - 20), out_h - 4)
    font = _load_font(font_paths, font_size)

    # Buat kanvas sementara lebih besar untuk diukur
    tmp = Image.new("L", (out_w * 4, out_h * 4), 255)
    draw = ImageDraw.Draw(tmp)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Posisi x: kiri / tengah / kanan secara acak
    margin_left = random.randint(2, max(3, (out_w - text_w) // 2 + 5))
    x = max(0, min(margin_left, out_w - text_w - 2))

    # Posisi y: tengahkan vertikal dengan sedikit jitter
    y_center = (out_h - text_h) // 2
    y = max(0, y_center + random.randint(-2, 2))

    # Background thermal
    bg_np = _make_thermal_background(out_h, out_w)
    img = Image.fromarray(bg_np, mode="L")
    draw = ImageDraw.Draw(img)

    # Warna teks: hitam (5–40) — agak pudar untuk thermal faded
    text_color = random.randint(5, 50)
    draw.text((x - bbox[0], y - bbox[1]), text, font=font, fill=text_color)

    # Efek opsional
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    if random.random() < 0.2:
        # Sedikit kemiringan (shear horizontal)
        img_np = np.array(img)
        shear = random.uniform(-0.05, 0.05)
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img_np = cv2.warpAffine(img_np, M, (out_w, out_h),
                                borderMode=cv2.BORDER_REPLICATE)
        img = Image.fromarray(img_np)

    # Resize ke dimensi tepat (jika render sedikit berbeda)
    img = img.resize((out_w, out_h), Image.LANCZOS)

    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helpers I/O
# ---------------------------------------------------------------------------

def _read_labels_csv(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _next_crop_idx(csv_path):
    rows = _read_labels_csv(csv_path)
    indices = []
    for r in rows:
        stem = Path(r["filepath"]).stem
        if stem.startswith("syn_"):
            try:
                indices.append(int(stem.split("_")[1]))
            except ValueError:
                pass
        elif stem.startswith("crop_"):
            try:
                indices.append(int(stem.split("_")[1]))
            except ValueError:
                pass
    return max(indices) + 1 if indices else len(rows)


def _append_csv(csv_path, rows):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "label", "class"],
            quoting=csv.QUOTE_ALL,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_synthetic(
    dst_dir,
    dst_csv,
    font_paths,
    existing_rows,
    class_filter,
    rerenders_per_label,
    random_count_per_class,
    only_random,
):
    crop_idx = _next_crop_idx(dst_csv)
    new_rows = []

    def _save_and_record(img, label, class_name):
        nonlocal crop_idx
        class_dir = os.path.join(dst_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        fname = f"syn_{crop_idx:05d}.jpg"
        cv2.imwrite(os.path.join(class_dir, fname), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        new_rows.append({
            "filepath": f"{class_name}/{fname}",
            "label"   : label,
            "class"   : class_name,
        })
        crop_idx += 1

    # --- 1. Re-render label yang sudah ada ---
    if not only_random and existing_rows:
        classes_to_render = {
            r["class"] for r in existing_rows
            if not class_filter or r["class"] in class_filter
        }
        print(f"\n[Re-render] {rerenders_per_label}x per label dari labels.csv")

        by_class = {}
        for r in existing_rows:
            if class_filter and r["class"] not in class_filter:
                continue
            label = _filter_chars(r["label"])
            if label:
                by_class.setdefault(r["class"], []).append(label)

        for class_name, labels in by_class.items():
            count = 0
            for label in labels:
                for _ in range(rerenders_per_label):
                    if not label:
                        continue
                    img = render_text_image(label, font_paths)
                    _save_and_record(img, label, class_name)
                    count += 1
            print(f"  {class_name}: {count} gambar dari {len(labels)} label unik")

    # --- 2. Generate teks struk baru ---
    target_classes = [c for c in YOLO_CLASSES if not class_filter or c in class_filter]
    print(f"\n[Random generate] {random_count_per_class} per kelas: {target_classes}")

    for class_name in target_classes:
        gen_fn = CLASS_GENERATORS.get(class_name)
        if gen_fn is None:
            print(f"  [SKIP] Tidak ada generator untuk kelas: {class_name}")
            continue

        generated = 0
        skipped   = 0
        for _ in range(random_count_per_class):
            label = gen_fn()
            if not label or len(label) < 2:
                skipped += 1
                continue
            img = render_text_image(label, font_paths)
            _save_and_record(img, label, class_name)
            generated += 1

        print(f"  {class_name}: {generated} gambar baru (skip {skipped})")

    _append_csv(dst_csv, new_rows)
    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OCR images untuk training CRNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # Default: re-render 2x semua label yang ada + 300 teks random per kelas
  python ai/synthetic_data.py

  # Hanya kelas tertentu, lebih banyak random
  python ai/synthetic_data.py --classes line_item total_belanja --random-count 500

  # Hanya generate random, skip re-render label lama
  python ai/synthetic_data.py --only-random --random-count 1000

  # Output ke folder berbeda (bukan ocr_dataset utama)
  python ai/synthetic_data.py --dst data/ocr_syn
        """,
    )
    parser.add_argument("--src", default=OCR_DATASET_DIR,
                        help="Folder sumber labels.csv")
    parser.add_argument("--dst", default=OCR_DATASET_DIR,
                        help="Folder output gambar sintetis (default: sama dengan --src)")
    parser.add_argument("--count", type=int, default=2, dest="rerenders",
                        help="Jumlah re-render per label dari labels.csv (default: 2)")
    parser.add_argument("--random-count", type=int, default=300,
                        help="Jumlah teks baru yang digenerate per kelas (default: 300)")
    parser.add_argument("--classes", nargs="*", default=None,
                        metavar="CLASS",
                        help="Filter kelas: nama_toko line_item tanggal_waktu total_belanja")
    parser.add_argument("--only-random", action="store_true",
                        help="Lewati re-render label lama, hanya generate teks baru")
    args = parser.parse_args()

    src_csv = os.path.join(args.src, "labels.csv")
    dst_csv = os.path.join(args.dst, "labels.csv")

    print("=" * 55)
    print("Synthetic Data Generator — NotePay OCR")
    print("=" * 55)

    font_paths = find_fonts()
    if font_paths:
        print(f"Font ditemukan: {len(font_paths)} font")
        for fp in font_paths:
            print(f"  {fp}")
    else:
        print("PERINGATAN: Tidak ada font TTF ditemukan — menggunakan PIL default bitmap.")
        print("  Install font: sudo apt install fonts-liberation fonts-dejavu")

    existing_rows = _read_labels_csv(src_csv)
    print(f"\nLabel sumber : {src_csv}")
    print(f"Jumlah label : {len(existing_rows)} baris")
    print(f"Output       : {args.dst}")
    print(f"Re-render    : {args.rerenders}x per label {'(dilewati)' if args.only_random else ''}")
    print(f"Random baru  : {args.random_count} per kelas")
    if args.classes:
        print(f"Filter kelas : {args.classes}")
    print()

    total = generate_synthetic(
        dst_dir             = args.dst,
        dst_csv             = dst_csv,
        font_paths          = font_paths,
        existing_rows       = existing_rows,
        class_filter        = set(args.classes) if args.classes else None,
        rerenders_per_label = args.rerenders,
        random_count_per_class = args.random_count,
        only_random         = args.only_random,
    )

    print(f"\nSelesai. {total} gambar sintetis baru ditambahkan ke {dst_csv}")
    print("\nLangkah selanjutnya:")
    print("  Jalankan augment_dataset.py untuk split train/val/test, lalu fase4_train_crnn.py.")


if __name__ == "__main__":
    main()
