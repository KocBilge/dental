import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class DentalAnalysis:
    def __init__(self, data_path, image_folder, output_path):
        self.data_path = data_path
        self.image_folder = image_folder
        self.output_path = output_path
        self.data = None
        self.data_cleaned = None
        self.alignment_scores = []
        self.generated_captions = []
        self.similarity_scores = []
        self.treatment_suggestions = []
        self.treatment_alignment_scores = []

    def load_and_clean_data(self):
        """Veri setini yükle ve temizle."""
        self.data = pd.read_excel(self.data_path, header=1)
        self.data_cleaned = self.data.dropna()
        print(f"\nTemizlenmiş Veri Seti Boyutu: {self.data_cleaned.shape[0]} satır")

    def display_cleaned_data(self, num_rows=5):
        """Temizlenmiş verinin ilk birkaç satırını görüntüle."""
        print("\nTemizlenmiş Veri Seti İlk Satırlar:")
        for i, row in self.data_cleaned.head(num_rows).iterrows():
            print(f"{i+1}. Görsel: {row['Image']}, Etiket: {row['Label']}, Yorum: {row['Comment']}")

    def preprocess_images(self):
        """Görselleri boyutlandır ve hazırlık yap."""
        print("\nGörseller Boyutlandırılıyor...")
        for image_path in self.data_cleaned['Image']:
            try:
                full_path = os.path.join(self.image_folder, image_path)
                image = Image.open(full_path).convert("RGB")
                resized_image = image.resize((224, 224))
                resized_image.save(full_path)
            except Exception as e:
                print(f"Görsel Boyutlandırma Hatası ({image_path}): {e}")

    def analyze_alignment(self):
        """CLIP modeliyle görsel-metin uyum analizi yap."""
        print("\nGörsel-Metin Uyum Analizi Başlıyor...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for image_path, comment in zip(self.data_cleaned['Image'], self.data_cleaned['Comment']):
            try:
                full_image_path = os.path.join(self.image_folder, image_path)
                image = Image.open(full_image_path)
                inputs = processor(text=[comment], images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                self.alignment_scores.append(probs.item())
                print(f"Görsel: {image_path} - Uyum Skoru: {probs.item():.4f}")
            except Exception as e:
                print(f"Görsel Yükleme Hatası ({image_path}): {e}")
                self.alignment_scores.append(None)

    def generate_treatment_suggestions(self):
        """LLama 3.1 ile alternatif tedavi önerileri oluştur."""
        print("\nTedavi Önerileri LLama 3.1 ile Oluşturuluyor...")
        generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
        for comment in self.data_cleaned['Comment']:
            try:
                suggestion = generator(f"Yorum: {comment}\nTedavi Önerisi:", max_length=150, num_return_sequences=1)
                self.treatment_suggestions.append(suggestion[0]['generated_text'])
                print(f"Yorum: {comment}\nÖnerilen Tedavi: {suggestion[0]['generated_text']}")
            except Exception as e:
                print(f"Tedavi Önerisi Hatası: {e}")
                self.treatment_suggestions.append(None)
    
    def compare_blip_with_comments(self):
        """BLIP ile oluşturulan metinleri uzman yorumlarıyla kıyasla."""
        print("\nBLIP ile Uzman Yorumları Kıyaslanıyor...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        for caption, comment in zip(self.generated_captions, self.data_cleaned['Comment']):
            try:
                caption_embedding = model.encode(caption)
                comment_embedding = model.encode(comment)
                similarity = cosine_similarity([caption_embedding], [comment_embedding])[0][0]
                self.similarity_scores.append(similarity)
                print(f"BLIP Metni: {caption} -> Uzman Yorumu: {comment}\nBenzerlik Skoru: {similarity:.4f}")
            except Exception as e:
                print(f"Kıyaslama Hatası: {e}")
                self.similarity_scores.append(None)
            

    def analyze_treatment_alignment(self):
        """Tedavi önerileri ile uzman planlarının uyumunu analiz et."""
        print("\nTedavi Önerileri ile Uzman Planlarının Uyumu Analiz Ediliyor...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        for suggestion, comment in zip(self.treatment_suggestions, self.data_cleaned['Comment']):
            try:
                suggestion_embedding = model.encode(suggestion)
                comment_embedding = model.encode(comment)
                similarity = cosine_similarity([suggestion_embedding], [comment_embedding])[0][0]
                self.treatment_alignment_scores.append(similarity)
                print(f"Tedavi Önerisi: {suggestion} -> Uzman Yorumu: {comment}\nUyum Skoru: {similarity:.4f}")
            except Exception as e:
                print(f"Uyum Analizi Hatası: {e}")
                self.treatment_alignment_scores.append(None)

    def save_results(self):
        """Sonuçları Excel dosyasına kaydet."""
        self.data_cleaned['Alignment Score'] = self.alignment_scores
        self.data_cleaned['Generated Caption'] = self.generated_captions
        self.data_cleaned['Similarity Score'] = self.similarity_scores
        self.data_cleaned['Treatment Suggestion'] = self.treatment_suggestions
        self.data_cleaned['Treatment Alignment Score'] = self.treatment_alignment_scores
        self.data_cleaned.to_excel(self.output_path, index=False)
        print(f"Sonuçlar başarıyla kaydedildi: {self.output_path}")

    def run_analysis(self):
        """Tüm analiz sürecini çalıştır."""
        self.load_and_clean_data()
        self.display_cleaned_data()
        self.preprocess_images()
        self.analyze_alignment()
        self.generate_treatment_suggestions()
        self.analyze_treatment_alignment()
        self.save_results()


# Ana program
if __name__ == "__main__":
    data_path = '/Users/bilge/Desktop/dental-csv-excel.xlsx'
    image_folder = "/Users/bilge/Desktop/dental_project.v1i.yolov11/train"
    output_path = "/Users/bilge/Desktop/cleaned_data_with_results.xlsx"

    analysis = DentalAnalysis(data_path, image_folder, output_path)
    analysis.run_analysis()