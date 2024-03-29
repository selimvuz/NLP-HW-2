import pandas as pd

# Excel dosyasını oku
file_path = 'dataset/soru_cevap.xlsx'
df = pd.read_excel(file_path)

# Tercih sütununa göre filtrele ve say
insan_cevabi_sayisi = (df['tercih'] == 1).sum()
makine_cevabi_sayisi = (df['tercih'] == 2).sum()
ikisi_de_iyi_sayisi = (df['tercih'] == 3).sum()

# Sonuçları yazdır
print("İnsan cevabının tercih edildiği sayı:", insan_cevabi_sayisi)
print("Makine cevabının tercih edildiği sayı:", makine_cevabi_sayisi)
print("Her iki cevabın da iyi olduğu durumların sayısı:", ikisi_de_iyi_sayisi)
