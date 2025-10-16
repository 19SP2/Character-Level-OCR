import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

# -----------------------------
# 1️⃣ Dataset Class
# -----------------------------
class OCRDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Preprocessing: resize height=32, width=128
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor()
        ])
        # Define charset including letters, digits, symbols
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{};:'\",.<>?/\\|`~∑∫√π≤≥≈"
        self.char_to_idx = {c:i+1 for i,c in enumerate(chars)}  # 0 = blank for CTC
        self.idx_to_char = {i:c for c,i in self.char_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def encode_label(self, label):
        return [self.char_to_idx[c] for c in label if c in self.char_to_idx]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")
        img = self.transform(img)
        label = torch.tensor(self.encode_label(row["label"]), dtype=torch.long)
        return img, label

# -----------------------------
# 2️⃣ Collate function for variable-length labels
# -----------------------------
def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return imgs, labels_concat, label_lengths

# -----------------------------
# 3️⃣ CRNN Model
# -----------------------------
class CRNN(nn.Module):
    def __init__(self, img_h=32, nc=1, nclass=80, nh=256):
        super(CRNN, self).__init__()
        # CNN layers (unchanged)
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
        # RNN layers (call manually)
        self.rnn1 = nn.LSTM(512, nh, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(nh*2, nh, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "Height after CNN must be 1"
        conv = conv.squeeze(2).permute(0,2,1)  # (batch, width, channels)
        recurrent, _ = self.rnn1(conv)
        recurrent, _ = self.rnn2(recurrent)
        output = self.embedding(recurrent)  # (batch, width, nclass)
        output = output.permute(1,0,2)  # (seq_len, batch, nclass) for CTC
        return output

# -----------------------------
# 4️⃣ CTC Decoding
# -----------------------------
def ctc_greedy_decode(preds, idx_to_char):
    preds = preds.permute(1,0,2)  # (batch, seq_len, nclass)
    preds = preds.argmax(2).cpu().numpy()
    texts = []
    for p in preds:
        chars = []
        prev = -1
        for idx in p:
            if idx != prev and idx != 0:
                chars.append(idx_to_char.get(idx,""))
            prev = idx
        texts.append("".join(chars))
    return texts

# -----------------------------
# 5️⃣ Training Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = OCRDataset("datasets/data.csv")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

model = CRNN(nclass=len(dataset.char_to_idx)+1).to(device)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct_chars = 0
    total_chars = 0

    for imgs, labels_concat, label_lengths in dataloader:
        imgs = imgs.to(device)
        labels_concat = labels_concat.to(device)
        label_lengths = label_lengths.to(device)

        # Forward pass
        preds = model(imgs)
        preds_log_softmax = preds.log_softmax(2)
        preds_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(device)

        # Compute CTC loss
        loss = criterion(preds_log_softmax, labels_concat, preds_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # ---- Accuracy calculation ----
        # Greedy decode for batch
        with torch.no_grad():
            preds_decoded = preds.permute(1,0,2).argmax(2)  # (batch, seq_len)
            start_idx = 0
            for i, l in enumerate(label_lengths):
                gt = labels_concat[start_idx:start_idx+l].cpu().numpy()
                pred_seq = preds_decoded[i].cpu().numpy()
                # Collapse repeated chars & remove blanks (0)
                pred_text = []
                prev = -1
                for idx in pred_seq:
                    if idx != prev and idx != 0:
                        pred_text.append(idx)
                    prev = idx
                # Count character-level matches
                matches = sum([p==g for p,g in zip(pred_text, gt)])
                correct_chars += matches
                total_chars += len(gt)
                start_idx += l

    epoch_acc = correct_chars / total_chars if total_chars > 0 else 0
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {epoch_loss/len(dataloader):.4f} — Accuracy: {epoch_acc*100:.2f}%")

# -----------------------------
# 6️⃣ Save Model
# -----------------------------
os.makedirs("ocr_crnn_model", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': dataset.char_to_idx,
    'idx_to_char': dataset.idx_to_char
}, "ocr_crnn_model/crnn.pth")
print("✅ Model saved to ocr_crnn_model/crnn.pth")

# -----------------------------
# 7️⃣ Example Inference
# -----------------------------
model.eval()
with torch.no_grad():
    sample_img, sample_label = dataset[0]
    pred = model(sample_img.unsqueeze(0).to(device))
    text = ctc_greedy_decode(pred, dataset.idx_to_char)[0]
print(f"Ground truth: {''.join([dataset.idx_to_char[i.item()] for i in sample_label])}")
print(f"Prediction  : {text}")

# thon train_model.pypy