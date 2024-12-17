import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, pipeline, AutoTokenizer, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# Hugging Face zaman aşımı ayarı
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "7200"

class DentalAnalysis:
    def __init__(self, data_path, image_folder, output_path, token):
        """
        DentalAnalysis sınıfı, dental analiz sürecini yönetir.
        
        Args:
            data_path (str): Veri dosyasının yolu.
            image_folder (str): Görsellerin bulunduğu klasörün yolu.
            output_path (str): Analiz sonuçlarının kaydedileceği dosya yolu.
            token (str): Hugging Face API tokeni.
        """
        self.data_path = data_path
        self.image_folder = image_folder
        self.output_path = output_path
        self.token = token
        self.data = None
        self.data_cleaned = None
        self.alignment_scores = []
        self.missing_images = []
        self.treatment_suggestions = []
        self.treatment_alignment_scores = []

    def load_and_clean_data(self):
        """Veri setini yükle ve temizle."""
        self.data = pd.read_excel(self.data_path, header=1)  # Excel dosyasını yükle
        self.data_cleaned = self.data.dropna()  # Eksik verileri temizle
        print(f"\nTemizlenmiş Veri Seti Boyutu: {self.data_cleaned.shape[0]} satır")

    def check_images(self):
        """Görselleri kontrol et."""
        print("\nGörseller kontrol ediliyor...")
        for image_name in self.data_cleaned['Image']:
            full_path = os.path.join(self.image_folder, image_name)
            if not os.path.exists(full_path):
                print(f"Eksik Görsel: {image_name}")
                self.missing_images.append(image_name)
        if self.missing_images:
            print(f"\nEksik Görseller Listesi: {self.missing_images}")

    def preprocess_images(self):
        """Görselleri boyutlandır."""
        print("\nGörseller işleniyor...")
        for image_name in self.data_cleaned['Image']:
            if image_name in self.missing_images:
                continue  # Eksik görselleri atla
            try:
                full_path = os.path.join(self.image_folder, image_name)
                image = Image.open(full_path).convert("RGB")
                resized_image = image.resize((384, 384))  # Görselleri yeniden boyutlandır
                resized_image.save(full_path)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"Görsel İşleme Hatası ({image_name}): {e}")
                self.missing_images.append(image_name)

    def analyze_alignment(self):
        """CLIP modeliyle görsel-metin uyum analizi."""
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
        """Aya veya LLaMA modeli ile tedavi önerileri oluştur."""
        print("\nTedavi Önerileri Oluşturuluyor...")
        model_id = "CohereForAI/aya-expanse-32b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        few_shot_example = (
            "Hasta şikayet: Diş ağrısı ve sızı.\nTedavi önerisi: Diş kontrolü yapılmalı, gerekli durumlarda dolgu uygulanmalı.\n\n"
        )

        for comment in self.data_cleaned['Comment']:
            try:
                messages = [
                    {"role": "user", "content": f"{few_shot_example}Hasta şikayet: {comment}\nTedavi önerisi:"}
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                gen_tokens = model.generate(
                    input_ids, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    temperature=0.3,
                )
                gen_text = tokenizer.decode(gen_tokens[0])
                self.treatment_suggestions.append(gen_text.split("Tedavi önerisi:")[-1].strip())
            except Exception as e:
                print(f"Tedavi Önerisi Hatası: {e}")
                self.treatment_suggestions.append("Bilgi yetersiz")

    def analyze_treatment_alignment(self):
        """Tedavi önerileri ile uzman planlarının uyumunu analiz et."""
        print("\nTedavi Önerileri ile Uzman Planlarının Uyumu Analiz Ediliyor...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        for suggestion, comment in zip(self.treatment_suggestions, self.data_cleaned['Comment']):
            try:
                suggestion_embedding = model.encode(suggestion, normalize_embeddings=True)
                comment_embedding = model.encode(comment, normalize_embeddings=True)
                similarity = cosine_similarity([suggestion_embedding], [comment_embedding])[0][0]
                self.treatment_alignment_scores.append(similarity)
            except Exception as e:
                self.treatment_alignment_scores.append(None)

    def save_results(self):
        """Sonuçları Excel dosyasına kaydet."""
        self.data_cleaned['Alignment Score'] = self.alignment_scores
        self.data_cleaned['Treatment Suggestion'] = self.treatment_suggestions
        self.data_cleaned['Treatment Alignment Score'] = self.treatment_alignment_scores
        self.data_cleaned.to_excel(self.output_path, index=False)
        print(f"\nSonuçlar başarıyla kaydedildi: {self.output_path}")

    def run_analysis(self):
        self.load_and_clean_data()
        self.check_images()
        self.preprocess_images()
        self.analyze_alignment()
        self.generate_treatment_suggestions()
        self.analyze_treatment_alignment()
        self.save_results()

# Ana program
if __name__ == "__main__":
    data_path = '/Users/bilge/Desktop/dental-csv-excel.xlsx'
    image_folder = "/Users/bilge/Desktop/dental_project.v1i.yolov11/train/images"
    output_path = "/Users/bilge/Desktop/cleaned_data_with_results.xlsx"
    token = "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"

    analysis = DentalAnalysis(data_path, image_folder, output_path, token)
    analysis.run_analysis()