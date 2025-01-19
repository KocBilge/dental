import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
import openai
import torch
import warnings
import gc
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein
from bert_score import score as bert_score
import psutil
import matplotlib.pyplot as plt

# === Bellek Temizleme ve Performans İzleme ===
def cleanup():
    """Bellek temizleme ve GPU önbelleği boşaltma."""
    gc.collect()
    torch.cuda.empty_cache()

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Anlık Bellek Kullanımı: {memory_info.rss / (1024 * 1024):.2f} MB")

# === Metrik Hesaplama Sınıfı ===
class Metrics:
    @staticmethod
    def calculate_bleu(reference, generated):
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)

    @staticmethod
    def calculate_rouge(reference, generated):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    @staticmethod
    def calculate_meteor(reference, generated):
        return meteor_score([reference], generated)

    @staticmethod
    def calculate_cosine_similarity(reference, generated):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        ref_embedding = model.encode([reference])
        gen_embedding = model.encode([generated])
        return cosine_similarity(ref_embedding, gen_embedding)[0][0]

    @staticmethod
    def calculate_edit_distance(reference, generated):
        return Levenshtein.distance(reference, generated)

    @staticmethod
    def calculate_bert_score(reference, generated):
        P, R, F1 = bert_score([generated], [reference], model_type='bert-base-uncased', lang='en', verbose=False)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }

# === Dental Analiz Sınıfı ===
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
        self.metrics_results = []

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
            except Exception as e:
                print(f"Görsel Yükleme Hatası ({image_name}): {e}")
                self.alignment_scores.append(None)

    def generate_treatment_suggestions(self):
        print("\nTedavi Önerileri ve Metrik Hesaplamaları Başlıyor...")
        for i, row in self.data_cleaned.iterrows():
            comment = row['Comment']
            reference = row.get('Expected Output', "Referans Yok")
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Sen bir diş hekimi asistanısın."},
                        {"role": "user", "content": f"Hasta şikayet: {comment}\nTedavi önerisi:"}
                    ],
                    max_tokens=100,
                    temperature=0.3
                )
                generated = response['choices'][0]['message']['content'].strip()

                # Metrik hesaplamaları
                bleu_score = Metrics.calculate_bleu(reference, generated)
                rouge_scores = Metrics.calculate_rouge(reference, generated)
                meteor = Metrics.calculate_meteor(reference, generated)
                cosine_sim = Metrics.calculate_cosine_similarity(reference, generated)
                edit_distance = Metrics.calculate_edit_distance(reference, generated)
                bert_scores = Metrics.calculate_bert_score(reference, generated)

                self.treatment_suggestions.append(generated)
                self.metrics_results.append({
                    'BLEU': bleu_score,
                    'ROUGE-1': rouge_scores['rouge1'],
                    'ROUGE-2': rouge_scores['rouge2'],
                    'ROUGE-L': rouge_scores['rougeL'],
                    'METEOR': meteor,
                    'Cosine Similarity': cosine_sim,
                    'Edit Distance': edit_distance,
                    'BERT Precision': bert_scores['precision'],
                    'BERT Recall': bert_scores['recall'],
                    'BERT F1': bert_scores['f1']
                })
            except Exception as e:
                print(f"Tedavi Önerisi Hatası: {e}")
                self.treatment_suggestions.append(None)
                self.metrics_results.append(None)

    def save_results(self):
        print("\nSonuçlar Kaydediliyor...")
        self.data_cleaned['Treatment Suggestion'] = self.treatment_suggestions
        self.data_cleaned = pd.concat([self.data_cleaned, pd.DataFrame(self.metrics_results)], axis=1)
        self.data_cleaned.to_excel(self.output_path, index=False)
        print(f"Sonuçlar başarıyla kaydedildi: {self.output_path}")

    def visualize_metrics(self):
        metrics_df = pd.DataFrame(self.metrics_results).dropna()
        metrics_df.plot(kind='box', figsize=(12, 8))
        plt.title("Metrik Dağılımları")
        plt.ylabel("Değerler")
        plt.show()

    def run(self):
        self.load_and_clean_data()
        self.check_images()
        self.preprocess_images()
        self.analyze_alignment()
        self.generate_treatment_suggestions()
        self.save_results()
        self.visualize_metrics()
        cleanup()

if __name__ == "__main__":
    analysis = DentalAnalysis(
        '/Users/bilge/Desktop/dental-csv-excel.xlsx',
        "/Users/bilge/Desktop/dental_project.v1i.yolov11/train/images",
        "/Users/bilge/Desktop/cleaned_data_with_results.xlsx"
    )
    analysis.run()
