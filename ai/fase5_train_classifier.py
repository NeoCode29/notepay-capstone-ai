"""
Fase 5 — Training Expense Category Classifier (TF/Keras)

Arsitektur:
    Input teks (nama item struk)
        → TextVectorization (built-in, vocab bisa di-save dalam model)
        → Embedding(VOCAB_SIZE, EMBED_DIM)
        → GlobalAveragePooling1D
        → Dense(128, relu) → Dropout(0.3)
        → Dense(64, relu)  → Dropout(0.2)
        → Dense(NUM_CATEGORIES, softmax)

Dataset: synthetic — ~3000+ item struk minimarket Indonesia per kategori.
Tidak perlu file eksternal; semua data di-generate di script ini.

Jalankan di WSL2 (venv-tf aktif):
    source ~/venv-tf/bin/activate
    cd "/mnt/c/Users/kresna/Documents/SIB Dicoding/Capstone Project"
    python ai/fase5_train_classifier.py

Atau di Windows (venv-yolo / environment Python apa pun yang punya TF):
    python ai/fase5_train_classifier.py
"""

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    CLASSIFIER_MODEL_DIR, CLASSIFIER_MODEL_PATH,
    EXPENSE_CATEGORIES, NUM_CATEGORIES,
    CLASSIFIER_VOCAB_SIZE, CLASSIFIER_EMBED_DIM, CLASSIFIER_SEQ_LEN,
    CLASSIFIER_EPOCHS, CLASSIFIER_BATCH_SIZE, CLASSIFIER_LR,
    LOGS_DIR,
)

# ---------------------------------------------------------------------------
# Synthetic dataset — item struk minimarket Indonesia
# ---------------------------------------------------------------------------

DATASET: dict[str, list[str]] = {
    "Makanan & Minuman": [
        # Mie instan
        "Indomie Goreng", "Indomie Soto", "Indomie Rebus", "Indomie Kari Ayam",
        "Indomie Rendang", "Mie Sedaap Goreng", "Mie Sedaap Soto", "Supermie Ayam",
        "Sarimi Soto Ayam", "Pop Mie Ayam Bawang", "Pop Mie Goreng",
        "Indomie Jumbo", "Mie Goreng Spesial", "Mie Sedaap White Curry",
        # Air & minuman kemasan
        "Aqua 600ml", "Aqua 1500ml", "Aqua 330ml", "Le Minerale 600ml",
        "Club 600ml", "Vit 600ml", "Cleo 350ml", "Nestle Pure Life",
        "Pocari Sweat 500ml", "Pocari Sweat 350ml", "Mizone Orange",
        "Mizone Apple Blackcurrant", "Mizone Passion Fruit",
        "Teh Botol Sosro 450ml", "Teh Botol Sosro 350ml", "Teh Pucuk Harum",
        "Teh Pucuk 350ml", "Ultra Teh Kotak", "Frestea Green",
        "Coca Cola 390ml", "Sprite 390ml", "Fanta Strawberry", "Fanta Orange",
        "Pepsi Cola", "7 Up 390ml", "Pulpy Orange", "Sari Buah Pulpy",
        "Good Day Cappuccino", "Good Day Original", "Nescafe Classic",
        "Nescafe 3in1", "Kopi Kapal Api Spesial", "Torabika 3in1",
        "Kopiko Brown Coffee", "Luwak White Koffie", "TOP Coffee",
        "ABC Susu", "Milo Activ-Go", "Ovaltine 3in1", "Energen Coklat",
        "Energen Vanilla", "Quaker Instant Oatmeal",
        # Snack & kue
        "Chitato Sapi Panggang", "Chitato Original", "Chitato Lite",
        "Lays Original", "Lays BBQ", "Cheetos Keju", "Cheetos BBQ",
        "Taro Net Jagung Bakar", "Qtela Singkong Keju", "Qtela Tempe",
        "Piattos Keju", "Piattos Rumput Laut", "Richeese Ahh",
        "Oreo Original", "Oreo Coklat", "Roma Kelapa", "Roma Malkist",
        "Monde Butter Cookies", "Slai O'lai Stroberi",
        "Biskuat Coklat", "Biskuat Susu", "Khong Guan Assorted",
        "Wafer Tango Coklat", "Wafer Tango Stroberi", "Gery Cheese",
        "Nabati Wafer Roll Keju", "Roma Sari Gandum",
        "Kacang Garuda Original", "Kacang Dua Kelinci Kulit",
        "Kacang Atom Coklat", "Makaroni Ngehe", "Keripik Kentang",
        "Soyjoy Raisin", "Soyjoy Apple", "Tropicana Slim",
        # Roti
        "Sari Roti Tawar Kupas", "Sari Roti Tawar Gandum",
        "Sari Roti Coklat", "Sari Roti Keju", "Sari Roti Sandwich",
        "Roti Tawar Gardenia", "Roti Gandum",
        # Susu & produk susu
        "Susu UHT Ultramilk Full Cream", "Susu UHT Ultramilk Coklat",
        "Susu UHT Indomilk Stroberi", "Susu UHT Frisian Flag",
        "Susu Bendera Full Cream", "Bear Brand Susu Steril",
        "Susu Kental Manis Indomilk", "Susu Kental Manis Frisian Flag",
        "Yakult 5x65ml", "Cimory Yogurt Drink Stroberi", "Cimory Greek",
        "Keju Kraft Singles", "Keju Prochiz Spready",
        # Bahan masak & bumbu
        "Beras Premium 5kg", "Beras Pandan Wangi 5kg", "Beras Ramos 5kg",
        "Beras Pulen Wangi 10kg", "Tepung Terigu Segitiga Biru 1kg",
        "Tepung Terigu Kunci Biru", "Gula Pasir 1kg", "Gula Aren",
        "Minyak Goreng Bimoli 2L", "Minyak Goreng Sania 2L",
        "Minyak Goreng Tropical", "Minyak Goreng Filma",
        "Margarin Blue Band 200g", "Mentega Wijsman",
        "Kecap Manis Bango 275ml", "Kecap Manis ABC 275ml",
        "Saos Tomat ABC 335ml", "Saos Sambal ABC", "Saus Tiram Saori",
        "Kecap Asin Indofood", "Bumbu Nasi Goreng Indofood",
        "Royco Ayam 250g", "Masako Ayam", "Masako Sapi", "Penyedap Rasa Ajinomoto",
        "Santan Kara 65ml", "Santan Kara 200ml", "Sarden ABC 155g",
        "Kornet Tulip", "Tuna Kaleng ABC",
        "Teh Sariwangi 25tb", "Teh Celup Sosro", "Teh Celup 2 Tang",
        "Kopi Kapal Api 165g", "Teh Javana Melati",
        "Garam Dapur Kapal", "Lada Bumbu Lapangan",
        # Es krim & frozen
        "Es Krim Campina Coklat", "Es Krim Walls Populaire",
        "Es Krim Aice Coklat", "Aice Big Panda",
        "Es Teh Kacang Merah", "Nugget So Good",
        # Buah & sayur kemasan
        "Tomat 500g", "Wortel 500g", "Bayam Ikat",
    ],

    "Kebersihan & Perawatan": [
        # Sabun mandi
        "Sabun Lifebuoy Total 10", "Sabun Lifebuoy Bar 100g",
        "Sabun Dove Moisturizing", "Sabun Dove Original",
        "Sabun Lux Soft Rose", "Sabun Lux Magical Spell",
        "Sabun Dettol Original", "Sabun Dettol Fresh",
        "Sabun Nuvo Family", "Sabun Harmony", "Sabun GIV",
        "Sabun Cair Vaseline", "Sabun Cair Biore Body Foam",
        "Biore Body Foam Bright White", "Shinzui Body Scrub",
        # Shampoo & kondisioner
        "Shampoo Pantene Total Damage Care", "Shampoo Pantene Anti Dandruff",
        "Shampoo Sunsilk Black Shine", "Shampoo Sunsilk Smooth & Manageable",
        "Shampoo Head & Shoulders 2in1", "Shampoo Head & Shoulders Cool Menthol",
        "Shampoo Clear Men", "Shampoo Clear Anti Dandruff",
        "Shampoo Rejoice 3in1", "Shampoo Dove Hairfall Rescue",
        "Shampoo L'Oreal Total Repair", "Shampoo Makarizo",
        "Kondisioner Pantene", "Kondisioner Sunsilk",
        "Serum Rambut Ellips",
        # Pasta gigi & sikat gigi
        "Pasta Gigi Pepsodent Action 123", "Pasta Gigi Pepsodent Whitening",
        "Pasta Gigi Pepsodent Complete 8", "Pasta Gigi Close Up Deep Action",
        "Pasta Gigi Close Up Menthol Fresh", "Pasta Gigi Sensodyne",
        "Pasta Gigi Darlie All Shiny White", "Pasta Gigi Formula",
        "Sikat Gigi Oral-B Soft", "Sikat Gigi Formula",
        "Sikat Gigi Systema", "Obat Kumur Listerine",
        # Deterjen & perawatan pakaian
        "Deterjen Rinso Anti Noda 1kg", "Deterjen Rinso Cair 800ml",
        "Deterjen Attack Easy 900g", "Deterjen Daia Bunga Matahari",
        "Deterjen So Klin Lavender", "Deterjen Surf Semerbak Cinta",
        "Pewangi Molto Ultra Blue", "Pewangi Molto Violet",
        "Pewangi Downy Sunrise Fresh", "Pewangi Softlan",
        "Pemutih Bayclin 1L", "Pemutih So Klin",
        # Pembalut & toiletri wanita
        "Pembalut Softex Classic Maxi Wing", "Pembalut Laurier Active",
        "Pembalut Kotex Natural Soft", "Charm Extra Protection Wing",
        "Pantyliner Softex", "Pantyliner Laurier", "Pantyliner Hers Protex",
        # Tisu & kebersihan
        "Tisu Paseo 250 Sheet", "Tisu Paseo Facial 100 Sheet",
        "Tisu Toples Lovely", "Tisu Basah Baby Wipes",
        "Tisu Basah Fiesta", "Tisu Kering Kleenex",
        "Kapas Rabbit", "Cotton Bud Swisspers",
        # Deodorant & parfum
        "Deodorant Rexona Men Active", "Deodorant Rexona Women Shower Clean",
        "Deodorant Axe Anarchy", "Deodorant Nivea Pearl Beauty",
        "Deodorant Dove Original", "Spray Cologne Casablanca",
        # Pelembab & skincare
        "Pelembab Vaseline Healthy White", "Pelembab Nivea Soft",
        "Pelembab Citra Pearl White", "Losion Citra Hazeline Snow",
        "Sunblock Biore UV Perfect", "Toner Wardah", "Micellar Water Garnier",
        "Facial Wash Biore Oil Control", "Facial Wash Pond's",
        # Popok bayi
        "Popok Pampers Premium Care NB", "Popok Pampers Active Baby M",
        "Popok Mamy Poko Extra Dry M", "Popok Huggies Dry",
    ],

    "Rumah Tangga": [
        # Sabun cuci piring
        "Sabun Piring Sunlight Jeruk Nipis 800ml", "Sabun Piring Sunlight Lemon",
        "Sabun Piring Mama Lemon 800ml", "Sabun Piring Mama Lime",
        "Sabun Piring Extra Clean Wipol",
        # Pembersih lantai & kamar mandi
        "Pembersih Lantai Wipol Pine 800ml", "Pembersih Lantai Wipol Lavender",
        "Supersol Pembersih Lantai", "Ajax Lantai Lemon",
        "Karbol Wangi Bagus", "Pembersih Kamar Mandi Domestos",
        "Toilet Duck Citrus", "Bref Power Activ",
        # Pengharum ruangan
        "Glade Aerosol Lavender", "Glade Electric Jasmine",
        "Air Wick Lavender", "Stella Aerosol Fresh Flower",
        "Baygon Spray Lavender", "HIT Aerosol",
        # Pembasmi serangga
        "Baygon Semprot Aktif", "Hit Elektrik Refill",
        "Antis Hand Sanitizer", "Baygon Listrik",
        "Obat Nyamuk Bakar Fumakilla",
        # Kantong & plastik
        "Kantong Plastik Kresek Putih", "Kantong Sampah Hitam Tebal 60L",
        "Kantong Sampah Medium", "Plastik Wrap Serbaguna",
        "Kotak Makan Plastik",
        # Alat masak & peralatan
        "Spons Cuci Piring Scotch-Brite", "Spons Sabut Serbaguna",
        "Kain Lap Microfiber", "Sikat WC",
        # Bola lampu & listrik
        "Lampu LED Philips 10W", "Lampu LED Osram 7W",
        "Lampu LED Panasonic 9W", "Lampu Bohlam Hemat Energi",
        "Baterai ABC AA 2 pcs", "Baterai ABC AAA", "Baterai Alkaline Duracell",
        "Korek Api Gas 3in1", "Lilin Spiral",
        # Peralatan kecil
        "Lem Alteco", "Lem Fox Kayu", "Lakban Coklat",
        "Staples Joyko", "Pulpen Pilot G2", "Buku Tulis Sidu",
        "Amplop Surat", "Stopmap Plastik",
        # Pewangi & pewarna
        "Bayclin Pemutih Pakaian", "Dylon Pewarna Kain",
        "Kamper Anti Ngengat", "Kapur Barus Kupu-Kupu",
        "Gantungan Baju Plastik",
    ],

    "Kesehatan & Farmasi": [
        # Obat demam & nyeri
        "Paracetamol 500mg 10 tab", "Panadol Biru 10 tab",
        "Bodrex Extra 4 tab", "Bodrexin Sirup", "Oskadon SP",
        "Feminax Extra", "Ponstan Forte", "Ibuprofen 400mg",
        "Aspirin 80mg", "Novalgim",
        # Obat flu & batuk
        "Antangin JRG Cair", "Tolak Angin Sido Muncul",
        "Neo Napacin", "Neozep Forte", "Mixagrip Flu",
        "Decolgen 4 tab", "Actifed", "Rhinofed",
        "Komix OBH Jeruk", "Vicks Formula 44",
        "OBH Combi Batuk Pilek", "Woods Peppermint",
        "Laserin Ekspektoran", "Bisolvon Ekstra",
        # Obat maag & lambung
        "Promag 10 tab", "Mylanta Liquid 150ml", "Antasida Doen",
        "Polysilane", "Lambucid", "Ranitidine 150mg",
        "Omeprazole 20mg", "Nexium 20mg",
        # Obat diare & pencernaan
        "Diapet 4 kap", "Entrostop 4 tab", "Norit 50 tab",
        "Oralit Rasa Jeruk 5 sach", "Imodium 2mg",
        "New Diatabs 12 tab", "Lacto-B Probiotik",
        # Vitamin & suplemen
        "Vitamin C 500mg 10 tab", "Vitamin C Kawasaki", "Redoxon Effervescent",
        "Sangobion Kapsul 10 kap", "Tonikum Bayer Sirup",
        "Scott Emulsion Original 200ml", "Minyak Ikan Kapsul",
        "Suplemen Zinc", "Curcuma Plus Sirup",
        "Hemaviton Action", "Extra Joss Active",
        # Obat luka & antiseptik
        "Betadine Solution 30ml", "Betadine Krim 5g",
        "Rivanol 100ml", "Alkohol 70% 100ml",
        "Minyak Kayu Putih Cap Lang 120ml", "Minyak Kayu Putih Konicare",
        "Balsem Rheumason", "Counterpain Gel",
        "Perban Gulung 5cm", "Plester Luka Hansaplast",
        "Kasa Steril", "Tensoplast Elastis",
        # Obat mata & telinga
        "Rohto Eye Drops", "Insto Regular Eye Drop",
        "Obat Tetes Telinga Otopain",
        # Obat kulit
        "Salep Kulit Enkasari", "Kalpanax Krim",
        "Canesten Krim 10g", "Mycoral Krim",
        "Loratadine 10mg", "CTM 4mg Alergi",
        # Obat khusus
        "Neuralgin 10 tab", "Neurobion 5000", "Voltaren Gel 20g",
    ],

    "Elektronik & Pulsa": [
        # Pulsa & kuota
        "Pulsa Telkomsel 25.000", "Pulsa Telkomsel 50.000",
        "Pulsa Telkomsel 100.000", "Pulsa XL 25.000", "Pulsa XL 50.000",
        "Pulsa Indosat 25.000", "Pulsa Indosat 50.000",
        "Pulsa Tri 25.000", "Pulsa Smartfren 20.000",
        "Kuota Internet Telkomsel 5GB", "Kuota Internet XL 10GB",
        "Paket Data Indosat 3GB", "Kuota Tri 10GB",
        "Paket Flash Unlimited Telkomsel",
        # Token listrik & tagihan
        "Token Listrik 20.000", "Token Listrik 50.000",
        "Token Listrik 100.000", "Bayar Listrik PLN",
        "Token PLN Prabayar 500.000",
        # Aksesoris HP
        "Baterai Lithium Handphone", "Baterai Replacement Samsung",
        "Kabel Data USB Type-C", "Kabel Charger Micro USB",
        "Charger HP 2A Universal", "Adaptor Fast Charging",
        "Headset Earphone In-Ear", "Headset Kabel 3.5mm",
        "Tempered Glass Samsung A", "Screen Guard Anti Gores",
        "Case Softcase HP", "Back Cover Hp",
        "Memory Card 32GB Kingston", "Memory Card 16GB Sandisk",
        "Flash Disk 16GB Toshiba",
        # Baterai umum
        "Baterai 9V ABC", "Baterai Kotak 9V",
        "Batre Remote TV", "Baterai Watch CR2032",
        # Elektronik kecil
        "Stop Kontak 3 Lubang", "Lampu Senter LED Mini",
        "Kaset CD DVD-R Blank",
    ],

    "Pakaian & Aksesori": [
        # Kaos kaki & pakaian dalam
        "Kaos Kaki Pendek Pria", "Kaos Kaki Panjang", "Kaos Kaki Anak",
        "Celana Dalam Pria Rider", "Celana Dalam Wanita GT Man",
        "BH Bra Sorex", "Singlet Pria Daleman",
        # Aksesoris
        "Tali Pinggang Pria", "Belt Karet Anak",
        "Sendal Jepit Swallow", "Sendal Jepit Carvil",
        "Sendal Jepit Anak Hello Kitty",
        "Topi Kain Polos", "Masker Kain 3 Ply",
        "Masker Disposable 50pcs", "Sarung Tangan Kerja",
        # Lainnya
        "Jarum Jahit Set", "Benang Jahit",
        "Kancing Baju Putih", "Peniti Safety Pin",
        "Resleting Baju", "Perekat Velcro",
    ],

    "Lain-lain": [
        # Rokok (tersendiri tapi masuk lain-lain)
        "Rokok Gudang Garam Surya 16", "Rokok Surya 12",
        "Rokok Dji Sam Soe Magnum", "Dji Sam Soe Filter",
        "Marlboro Red 20", "Marlboro Menthol", "Marlboro Ice Burst",
        "Sampoerna Mild 16", "A Mild 20", "U Mild",
        "Rokok Dunhill Filter", "Esse Change",
        "Rokok Gratis 234", "Class Mild",
        # Administrasi & biaya
        "Biaya Administrasi", "Biaya Layanan", "Ongkos Kirim",
        "Administrasi Bank", "Biaya Pengemasan",
        "Kantong Belanja Plastik", "Paper Bag",
        # Fotokopi & ATK
        "Fotokopi A4", "Print Hitam Putih", "Print Warna",
        "Materai 10.000", "Perangko Surat",
        # Minuman keras (convenience store)
        "Bir Bintang 330ml", "Heineken 330ml",
        # Alat tulis
        "Pulpen Ballpoint", "Spidol Board Marker", "Penggaris 30cm",
        "Stabilo Boss Yellow", "Penghapus Faber-Castell",
        # Lain
        "Kotak Tisu", "Hanger Jemuran", "Bungkus Kado",
        "Kartu Ucapan", "Pita Dekorasi",
    ],
}


def _augment_item(item: str) -> list[str]:
    """Buat variasi item name: lowercase, uppercase, tambah angka qty/harga."""
    variants = [item, item.lower(), item.upper()]

    # tambah qty prefix
    for qty in ["1x", "2x", "3 pcs", "1 kg", "1 btl", "500ml"]:
        if random.random() < 0.3:
            variants.append(f"{qty} {item}")
            variants.append(f"{item} {qty}")

    # tambah harga/kode
    if random.random() < 0.2:
        price = random.choice(["5.000", "10.000", "25.000", "50.000"])
        variants.append(f"{item} {price}")

    return variants


def build_dataset() -> tuple[list[str], list[int]]:
    texts, labels = [], []
    cat_to_idx = {cat: i for i, cat in enumerate(EXPENSE_CATEGORIES)}

    for cat, items in DATASET.items():
        idx = cat_to_idx[cat]
        for item in items:
            for variant in _augment_item(item):
                texts.append(variant)
                labels.append(idx)

    # shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_classifier(vocab_size, embed_dim, seq_len, num_categories):
    import tensorflow as tf

    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")

    vectorize = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len,
        name="vectorization",
    )
    x = vectorize(text_input)
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="embedding")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="pooling")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    output = tf.keras.layers.Dense(num_categories, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=text_input, outputs=output, name="expense_classifier")
    return model, vectorize


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    os.makedirs(CLASSIFIER_MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("FASE 5 — Training Expense Category Classifier")
    print("=" * 60)

    # Dataset
    print("\n[1/4] Membangun dataset sintetis...")
    texts, labels = build_dataset()
    labels_arr = np.array(labels)
    total = len(texts)
    print(f"      Total sampel: {total}")
    for i, cat in enumerate(EXPENSE_CATEGORIES):
        count = (labels_arr == i).sum()
        print(f"      {i}. {cat}: {count}")

    # Split train/val
    val_n  = max(1, int(total * 0.15))
    train_texts  = texts[val_n:]
    train_labels = labels_arr[val_n:]
    val_texts    = texts[:val_n]
    val_labels   = labels_arr[:val_n]
    print(f"\n      Train: {len(train_texts)} | Val: {len(val_texts)}")

    # Build model + adapt vectorizer
    print("\n[2/4] Membangun model...")
    model, vectorize = build_classifier(
        CLASSIFIER_VOCAB_SIZE, CLASSIFIER_EMBED_DIM,
        CLASSIFIER_SEQ_LEN, NUM_CATEGORIES,
    )

    train_arr = np.array(train_texts, dtype=object).reshape(-1, 1)
    vectorize.adapt(train_arr)
    print(f"      Vocab ukuran: {len(vectorize.get_vocabulary())}")
    model.summary()

    # Dataset tf.data
    def make_ds(t, l, shuffle=False):
        t_arr = np.array(t, dtype=object).reshape(-1, 1)
        ds = tf.data.Dataset.from_tensor_slices((t_arr, l))
        if shuffle:
            ds = ds.shuffle(buffer_size=2000, seed=42)
        return ds.batch(CLASSIFIER_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_texts, train_labels, shuffle=True)
    val_ds   = make_ds(val_texts,   val_labels)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CLASSIFIER_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CLASSIFIER_MODEL_PATH,
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOGS_DIR, "..", "classifier"),
            histogram_freq=0,
        ),
    ]

    # Train
    print("\n[3/4] Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CLASSIFIER_EPOCHS,
        callbacks=callbacks,
    )

    # Evaluasi akhir
    print("\n[4/4] Evaluasi akhir...")
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"      Best val accuracy : {best_val_acc*100:.2f}%")

    # Simpan jika belum tersimpan oleh checkpoint
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        model.save(CLASSIFIER_MODEL_PATH)
    print(f"      Model tersimpan  : {CLASSIFIER_MODEL_PATH}")

    # Classification report & confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix

    val_arr  = np.array(val_texts, dtype=object).reshape(-1, 1)
    val_prob = model.predict(tf.constant(val_arr), verbose=0)
    val_pred = np.argmax(val_prob, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(
        val_labels, val_pred,
        target_names=EXPENSE_CATEGORIES,
        digits=4,
    ))

    cm = confusion_matrix(val_labels, val_pred)
    print("--- Confusion Matrix ---")
    header = f"{'':>24}" + "".join(f"{c[:6]:>8}" for c in EXPENSE_CATEGORIES)
    print(header)
    for i, row in enumerate(cm):
        label = EXPENSE_CATEGORIES[i][:24]
        print(f"{label:>24}" + "".join(f"{v:>8}" for v in row))

    # Uji cepat
    print("\n--- Quick test ---")
    test_items = [
        "Indomie Goreng",
        "Sabun Lifebuoy",
        "Paracetamol 500mg",
        "Pulsa Telkomsel 50.000",
        "Beras Premium 5kg",
        "Kaos Kaki",
        "Biaya Administrasi",
        "Token Listrik 50.000",
    ]
    test_arr = tf.constant([[item] for item in test_items])
    preds = model.predict(test_arr, verbose=0)
    for item, pred in zip(test_items, preds):
        idx = np.argmax(pred)
        conf = pred[idx]
        print(f"  {item:<35} → {EXPENSE_CATEGORIES[idx]} ({conf*100:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    train()
