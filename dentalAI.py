import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt


class DentalAnalysis:
    def __init__(self, data_path, image_folder, output_path):
        self.data_path = data_path
        self.image_folder = image_folder
        self.output_path = output_path
        self.data = None
        self.data_cleaned = None
        self.alignment_scores = []
        self.summarized_comments = []
        self.generated_captions = []
        self.treatment_suggestions = []

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

    def summarize_comments(self):
        """BART modeliyle yorumları özetle."""
        print("\nTedavi Önerisi ve Özetleme Başlıyor...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        for comment in self.data_cleaned['Comment']:
            try:
                summary = summarizer(comment, max_length=100, min_length=30, do_sample=False)
                self.summarized_comments.append(summary[0]['summary_text'])
                print(f"Orijinal Yorum: {comment}\nÖzet: {summary[0]['summary_text']}\n")
            except Exception as e:
                print(f"Özetleme Hatası: {e}")
                self.summarized_comments.append(None)

    def generate_captions(self):
        """BLIP modeliyle görsellerden otomatik metin oluştur."""
        print("\nGörsellerden Otomatik Metin Oluşturma Başlıyor...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        for image_path in self.data_cleaned['Image']:
            try:
                full_path = os.path.join(self.image_folder, image_path)
                image = Image.open(full_path).convert("RGB")
                
                inputs = processor(images=image, return_tensors="pt")
                outputs = model.generate(**inputs)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                
                self.generated_captions.append(caption)
                print(f"Görsel: {image_path} -> Tahmin Edilen Metin: {caption}")
            except Exception as e:
                print(f"Görsel Metin Üretim Hatası ({image_path}): {e}")
                self.generated_captions.append(None)

    def generate_treatment_suggestions(self):
        """Few-shot ve zero-shot analizlerle alternatif tedavi önerileri oluştur."""
        print("\nAlternatif Tedavi Önerileri Oluşturuluyor...")
        generator = pipeline("text-generation", model="gpt2")  # Hugging Face GPT modeli kullanılıyor
        
        for comment in self.data_cleaned['Comment']:
            try:
                suggestion = generator(f"Yorum: {comment}\nTedavi Önerisi:", max_length=100, num_return_sequences=1)
                self.treatment_suggestions.append(suggestion[0]['generated_text'])
                print(f"Orijinal Yorum: {comment}\nÖnerilen Tedavi: {suggestion[0]['generated_text']}\n")
            except Exception as e:
                print(f"Tedavi Önerisi Oluşturma Hatası: {e}")
                self.treatment_suggestions.append(None)

    def visualize_alignment_scores(self):
        """Görsel-metin uyum skorlarının dağılımını görselleştir."""
        print("\nGörsel-Metin Uyum Skorlarının Dağılımı...")
        scores_cleaned = [score for score in self.alignment_scores if score is not None]
        plt.hist(scores_cleaned, bins=10, color='blue', alpha=0.7)
        plt.title("Görsel-Metin Uyum Skor Dağılımı")
        plt.xlabel("Uyum Skoru")
        plt.ylabel("Frekans")
        plt.show()

    def save_results(self):
        """Sonuçları Excel dosyasına kaydet."""
        self.data_cleaned['Alignment Score'] = self.alignment_scores
        self.data_cleaned['Summary'] = self.summarized_comments
        self.data_cleaned['Generated Caption'] = self.generated_captions
        self.data_cleaned['Treatment Suggestion'] = self.treatment_suggestions
        self.data_cleaned.to_excel(self.output_path, index=False)
        print(f"Sonuçlar başarıyla kaydedildi: {self.output_path}")

    def run_analysis(self):
        """Tüm analiz sürecini çalıştır."""
        self.load_and_clean_data()
        self.display_cleaned_data()
        self.analyze_alignment()
        self.summarize_comments()
        self.generate_captions()
        self.generate_treatment_suggestions()
        self.visualize_alignment_scores()
        self.save_results()


# Ana program
if __name__ == "__main__":
    data_path = '/Users/bilge/Desktop/dental-csv-excel.xlsx'
    image_folder = "/Users/bilge/Desktop/dental_project.v1i.yolov11/train"
    output_path = "/Users/bilge/Desktop/cleaned_data_with_results.xlsx"

    analysis = DentalAnalysis(data_path, image_folder, output_path)
    analysis.run_analysis()