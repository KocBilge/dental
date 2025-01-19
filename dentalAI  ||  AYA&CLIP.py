import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
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

# === Genel Temizlik ve Performans İzleme ===
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

    def __init__(self, data_path, image_folder, output_path, token):
        self.data_path = data_path
        self.image_folder = image_folder
        self.output_path = output_path
        self.token = token
        self.data = None
        self.data_cleaned = None
        self.treatment_suggestions = []
        self.metrics_results = []

    def load_and_clean_data(self):
        print("\nVeri seti yükleniyor ve temizleniyor...")
        log_memory_usage()
        self.data = pd.read_excel(self.data_path, header=1)
        self.data_cleaned = self.data.dropna()
        print(f"Temizlenmiş Veri Seti Boyutu: {self.data_cleaned.shape[0]} satır")

    def preprocess_images(self):
        print("\nGörseller işleniyor...")
        log_memory_usage()
        for image_name in self.data_cleaned['Image']:
            full_path = os.path.join(self.image_folder, image_name)
            try:
                image = Image.open(full_path).convert("RGB")
                resized_image = image.resize((384, 384))
                resized_image.save(full_path)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"Görsel İşleme Hatası ({image_name}): {e}")

    def generate_treatment_suggestions(self):
        print("\nTedavi Önerileri Oluşturuluyor...")
        model_name = "CohereForAI/aya-expanse-32b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

        for _, row in self.data_cleaned.iterrows():
            comment = row['Comment']
            reference = row['Expected Output']
            try:
                input_text = f"Hasta şikayet: {comment}\nTedavi önerisi:"
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
                gen_tokens = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7)
                generated = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).split("Tedavi önerisi:")[-1].strip()

                # Metrikleri hesapla
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
                print(f"Tedavi önerisi oluşturulamadı: {e}")
                self.treatment_suggestions.append(None)
                self.metrics_results.append(None)
            cleanup()

    def save_results(self):
        print("\nSonuçlar kaydediliyor...")
        self.data_cleaned['Treatment Suggestion'] = self.treatment_suggestions
        self.data_cleaned = pd.concat([self.data_cleaned, pd.DataFrame(self.metrics_results)], axis=1)
        self.data_cleaned.to_excel(self.output_path, index=False, engine='openpyxl')
        print(f"\nSonuçlar başarıyla kaydedildi: {self.output_path}")

    def visualize_metrics(self):
        metrics_df = pd.DataFrame(self.metrics_results).dropna()
        metrics_df.plot(kind='box', figsize=(12, 8))
        plt.title("Metrik Dağılımları")
        plt.ylabel("Değerler")
        plt.show()

    def run(self):
        self.load_and_clean_data()
        self.preprocess_images()
        self.generate_treatment_suggestions()
        self.save_results()
        self.visualize_metrics()
        cleanup()

if __name__ == "__main__":
    TOKEN = "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"
    analysis = DentalAnalysis(
        data_path='/Users/bilge/Desktop/dental-csv-excel.xlsx',
        image_folder="/Users/bilge/Downloads/dental_project/train/images",
        output_path="/Users/bilge/Desktop/cleaned_data_with_results.xlsx",
        token=TOKEN
    )

    analysis.run()
