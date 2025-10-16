import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# CRNN Model Definition
class CRNN(nn.Module):
    def __init__(self, img_h=32, nc=1, nclass=80, nh=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU()
        )
        self.rnn1 = nn.LSTM(512, nh, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(nh*2, nh, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0,2,1)
        recurrent, _ = self.rnn1(conv)
        recurrent, _ = self.rnn2(recurrent)
        output = self.embedding(recurrent)
        output = output.permute(1,0,2)
        return output

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("ocr_crnn_model/crnn.pth", map_location=device)
idx_to_char = checkpoint['idx_to_char']

model = CRNN(nclass=len(idx_to_char)+1).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Helper Functions
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0)
    return img.to(device)

def ctc_greedy_decode(preds, idx_to_char):
    preds = preds.permute(1,0,2)
    preds = preds.argmax(2).cpu().numpy()
    texts = []
    for p in preds:
        chars = []
        prev = -1
        for idx in p:
            if idx != prev and idx != 0:
                chars.append(idx_to_char.get(idx, ""))
            prev = idx
        texts.append("".join(chars))
    return texts

# Load CSV and Compare
data_csv = pd.read_csv("datasets/data.csv")

# pick a few random samples to check predictions
for i in range(2):  # change number if you want more
    image_path = data_csv.iloc[i]["image_path"]
    ground_truth = data_csv.iloc[i]["label"]

    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_path}")
        continue

    img = preprocess_image(image_path)
    with torch.no_grad():
        pred = model(img)
        text = ctc_greedy_decode(pred, idx_to_char)[0]

    print(f"\nImage: {image_path}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Predicted   : {text}")

def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    idx_to_char = checkpoint["idx_to_char"]
    model = CRNN(nclass=len(idx_to_char)+1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, idx_to_char, device


# python inference_crnn.py
