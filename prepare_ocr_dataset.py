import os
import csv
import random
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from trdg.generators import GeneratorFromStrings
from PIL import Image, ImageDraw, ImageFont

def prepare_datasets():
    os.makedirs("datasets", exist_ok=True)
    all_data = []  # store image paths and labels

    # -----------------------------
    # 1️⃣ EMNIST (letters + digits)
    # -----------------------------
    print("⬇️ Downloading EMNIST (letters + digits)...")
    emnist = datasets.EMNIST(
        root="datasets/emnist",
        split='byclass',
        download=True,
        transform=transforms.ToTensor()
    )
    print(f"✅ EMNIST downloaded: {len(emnist)} samples")

    # Convert EMNIST samples to PNGs and add to CSV
    print("🧩 Converting EMNIST samples to images...")
    os.makedirs("datasets/emnist_images", exist_ok=True)

    # Limit number of EMNIST samples (you can change to use all)
    limit = 3000  # use fewer to keep it lightweight
    for i in range(limit):
        img, label = emnist[i]
        img = to_pil_image(img)  # Convert tensor -> PIL Image

        # Map EMNIST label to character
        if 0 <= label <= 9:
            char = str(label)
        elif 10 <= label <= 35:
            char = chr(label - 10 + 65)  # A–Z
        elif 36 <= label <= 61:
            char = chr(label - 36 + 97)  # a–z
        else:
            continue  # skip other symbols

        path = f"datasets/emnist_images/{i:05d}.png"
        img.save(path)
        all_data.append([path, char])

    print(f"✅ EMNIST images saved: {limit} samples added")

    # -----------------------------
    # 2️⃣ TRDG synthetic text images
    # -----------------------------
    print("🧮 Generating TRDG samples...")
    os.makedirs("datasets/trdg_custom", exist_ok=True)

    strings = []
    letters = [chr(i) for i in range(65, 91)]  # A–Z
    letters += [chr(i) for i in range(97, 123)]  # a–z
    digits = [str(i) for i in range(10)]
    symbols = list("!@#$%^&*()-_=+[]{};:'\",.<>?/\\|`~")
    all_chars = letters + digits + symbols

    # Create random short strings
    for _ in range(200):  # generate 200 samples
        s = ''.join(random.choices(all_chars, k=random.randint(1, 6)))
        strings.append(s)

    generator = GeneratorFromStrings(strings, count=len(strings), random_blur=True, random_skew=True)
    for i, (img, lbl) in enumerate(generator):
        path = f"datasets/trdg_custom/{i:04d}.png"
        img.save(path)
        all_data.append([path, lbl])
    print(f"✅ TRDG samples generated: {len(strings)}")

    # -----------------------------
    # 3️⃣ Synthetic math symbols
    # -----------------------------
    print("🧮 Creating synthetic math symbols...")
    os.makedirs("datasets/synthetic_math", exist_ok=True)

    symbols = ['+', '-', '=', '×', '÷', '∫', '√', 'π', '∑', '≤', '≥', '≈']
    font_path = "myFonts/DejaVuSans.ttf"  # Ensure this exists!
    if not os.path.exists(font_path):
        print("⚠️ Font not found. Please download DejaVuSans.ttf and place it in a 'myFonts/' folder.")
        return
    font = ImageFont.truetype(font_path, 32)

    for s in symbols:
        img = Image.new('L', (64, 64), color=0)
        d = ImageDraw.Draw(img)
        w, h = d.textsize(s, font=font)
        d.text(((64 - w) / 2, (64 - h) / 2), s, fill=255, font=font)
        path = f"datasets/synthetic_math/{ord(s)}.png"
        img.save(path)
        all_data.append([path, s])
    print(f"✅ Math symbols generated: {len(symbols)}")

    # -----------------------------
    # 4️⃣ Save all data to CSV
    # -----------------------------
    csv_path = "datasets/data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(all_data)

    print(f"📄 Data summary saved to: {csv_path}")
    print(f"✅ Total labeled samples: {len(all_data)}")

# ------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting dataset preparation...")
    prepare_datasets()
    print("✅ All datasets ready!")

# python prepare_ocr_dataset.py