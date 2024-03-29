from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import random

print("Model yükleniyor... -Turkish Tiny BERT")
# Model ve tokenizer yükleme
model_name = "ytu-ce-cosmos/turkish-tiny-bert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

m = 1

print("Veri kümesi yükleniyor...")
file_path = "dataset/soru_cevap.xlsx"
df = pd.read_excel(file_path)

# CUDA kontrolü
if torch.cuda.is_available():
    print("Model GPU'ya taşınıyor...")
    model.to('cuda')

# Soruların vektör temsillerini elde etme fonksiyonu


def get_vector_representation(question):
    global m
    print(
        f"{m}/{len(df['soru'].tolist())} soru vektör temsiline dönüştürülüyor...")
    inputs = tokenizer(question, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    m += 1
    return outputs.pooler_output.cpu().numpy()


# Gerçek soru listesi
sorular = df['soru'].tolist()

print("Soruların vektör temsilleri alınıyor...")
# Tüm soruların vektör temsillerini hesapla
vec_temsiller = np.array([get_vector_representation(soru)[0]
                         for soru in sorular])

# Rastgele 100 soru seç ve benzerlikleri hesapla
secilen_sorular_indices = list(range(9950, min(10050, len(sorular))))
for idx in secilen_sorular_indices:
    secilen_soru_vec = vec_temsiller[idx].reshape(1, -1)
    benzerlikler = cosine_similarity(secilen_soru_vec, vec_temsiller)[0]
    # Kendi kendine olan benzerliği çıkarmak için -11
    en_benzer_10 = np.argsort(benzerlikler)[-11:-1]
    print(f"Soru: {sorular[idx]}")
    print("En benzer 10 soru:")
    for benzer_idx in en_benzer_10.reverse():
        print(
            f"- {sorular[benzer_idx]} (Benzerlik: {benzerlikler[benzer_idx]:.4f})")
    print("\n---\n")
