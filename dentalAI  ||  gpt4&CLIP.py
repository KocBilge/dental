import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
import openai
import torch
import warnings
import gc
from evaluate import load

# Bellek ve GPU önbelleği temizleme
gc.collect()
torch.cuda.empty_cache()

# Cihazı CPU olarak ayarlama
device = torch.device("cpu")

# Uyarıları filtreleme
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be .* leaked semaphore objects")
warnings.filterwarnings("ignore", category=UserWarning, message="Current model requires .* bytes of buffer")
warnings.filterwarnings("ignore", category=UserWarning, message="for model.layers.*")

# Hugging Face zaman aşımı ve chunk boyutu ayarları
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "14400"
os.environ["HF_HUB_CHUNK_SIZE"] = "1048576"  # 1 MB

torch.multiprocessing.set_start_method("spawn", force=True)

# OpenAI API anahtarını tanımla
openai.api_key = ""

if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please set 'OPENAI_API_KEY' as an environment variable.")

class DentalAnalysis:

    def __init__(self, data_path, image_folder, output_path):
        self.data_path = data_path
        self.image_folder = image_folder
        self.output_path = output_path
        self.data = None
        self.data_cleaned = None
        self.alignment_scores = []
        self.missing_images = []
        self.treatment_suggestions = []
        self.bleu_score = None
        self.rouge_score = None

    def load_and_clean_data(self):
        self.data = pd.read_excel(self.data_path, header=1)
        self.data_cleaned = self.data.dropna(subset=['Comment'])
        print(f"\nTemizlenmiş Veri Seti Boyutu: {self.data_cleaned.shape[0]} satır")

    def check_images(self):
        print("\nGörseller kontrol ediliyor...")
        for image_name in self.data_cleaned['Image']:
            full_path = os.path.join(self.image_folder, image_name)
            if not os.path.exists(full_path):
                print(f"Eksik Görsel: {image_name}")
                self.missing_images.append(image_name)
        if self.missing_images:
            print(f"\nEksik Görseller Listesi: {self.missing_images}")

    def preprocess_images(self):
        print("\nGörseller işleniyor...")
        for image_name in self.data_cleaned['Image']:
            if image_name in self.missing_images:
                continue
            try:
                full_path = os.path.join(self.image_folder, image_name)
                image = Image.open(full_path).convert("RGB")
                resized_image = image.resize((384, 384))
                resized_image.save(full_path)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"Görsel İşleme Hatası ({image_name}): {e}")
                self.missing_images.append(image_name)

    def analyze_alignment(self):
        print("\nGörsel-Metin Uyum Analizi Başlıyor...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        for image_name, comment in zip(self.data_cleaned['Image'], self.data_cleaned['Comment']):
            if image_name in self.missing_images:
                self.alignment_scores.append(None)
                continue
            try:
                full_path = os.path.join(self.image_folder, image_name)
                image = Image.open(full_path)
                inputs = processor(text=[comment[:75]], images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).max().item()
                self.alignment_scores.append(probs)
                print(f"Görsel: {image_name} - Uyum Skoru: {probs:.4f}")
            except Exception as e:
                print(f"Görsel Yükleme Hatası ({image_name}): {e}")
                self.alignment_scores.append(None)

    def generate_treatment_suggestions(self):
        print("\nTedavi Önerileri Oluşturuluyor...")

        few_shot_examples = [
            {"role": "user", "content": "Diş ağrısı ve hassasiyet var."},
            {"role": "assistant", "content": "Diş ağrısını azaltmak için sıcak-soğuk hassasiyetini azaltan diş macunu kullanın."}
        ]

        for i, comment in enumerate(self.data_cleaned['Comment']):
            try:
                messages = [
                    {"role": "system", "content": "Sen bir diş hekimi asistanısın ve hastalara tedavi önerileri sunuyorsun."}
                ]

                if i % 5 == 0:  # Few-Shot her 5 yorumda bir kullanılır
                    messages.extend(few_shot_examples)

                messages.append({"role": "user", "content": f"Hasta şikayet: {comment}\nTedavi önerisi:"})

                response = openai.chat.completions.create(  # Doğru yöntem
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Sen bir diş hekimi asistanısın ve hastalara tedavi önerileri sunuyorsun."},
                        {"role": "user", "content": f"Hasta şikayet: {comment}\nTedavi önerisi:"}
                    ],
                    max_tokens=100,
                    temperature=0.3
                )

                suggestion = response['choices'][0]['message']['content'].strip()
                self.treatment_suggestions.append(suggestion)
                print(f"Tedavi Önerisi: {suggestion}")
            except Exception as e:
                print(f"Tedavi Önerisi Hatası: {e}")
                self.treatment_suggestions.append("Bilgi yetersiz")

    def save_results(self):
        self.data_cleaned['Alignment Score'] = self.alignment_scores
        self.data_cleaned['Treatment Suggestion'] = self.treatment_suggestions
        self.data_cleaned.to_excel(self.output_path, index=False)
        print(f"\nSonuçlar başarıyla kaydedildi: {self.output_path}")

    def run(self):
        self.load_and_clean_data()
        self.check_images()
        self.preprocess_images()
        self.analyze_alignment()
        self.generate_treatment_suggestions()
        self.save_results()


if __name__ == "__main__":
    analysis = DentalAnalysis(
        '/Users/bilge/Desktop/dental-csv-excel.xlsx',
        "/Users/bilge/Desktop/dental_project.v1i.yolov11/train/images",
        "/Users/bilge/Desktop/cleaned_data_with_results.xlsx"
    )
    analysis.run()
