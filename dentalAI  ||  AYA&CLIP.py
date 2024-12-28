import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import torch
import warnings
import gc
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

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


class DentalAnalysis:
    def __init__(self, data_path, image_folder, output_path, token):
        self.data_path = data_path
        self.image_folder = image_folder
        self.output_path = output_path
        self.token = token
        self.data = None
        self.data_cleaned = None
        self.alignment_scores = []
        self.missing_images = []
        self.treatment_suggestions = []
        self.bleu_scores = []
        self.rouge_scores = []

    def load_and_clean_data(self):
        #Veri setini yükle ve temizle
        self.data = pd.read_excel(self.data_path, header=1)
        self.data_cleaned = self.data.dropna()
        print(f"\nTemizlenmiş Veri Seti Boyutu: {self.data_cleaned.shape[0]} satır")

    def check_images(self):
        #Görselleri kontrol et
        print("\nGörseller kontrol ediliyor...")
        for image_name in self.data_cleaned['Image']:
            full_path = os.path.join(self.image_folder, image_name)
            if not os.path.exists(full_path):
                print(f"Eksik Görsel: {image_name}")
                self.missing_images.append(image_name)
        if self.missing_images:
            print(f"\nEksik Görseller Listesi: {self.missing_images}")

    def preprocess_images(self):
        #Görselleri boyutlandır
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
        #CLIP modeliyle görsel-metin uyum analizi
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
        #Aya-Expanse-32B ile tedavi önerileri oluştur
        print("\nTedavi Önerileri Oluşturuluyor...")
        model_path = "/Users/bilge/Downloads/aya-expanse-32b"
        if not os.path.exists(model_path):
            print("Model yerel olarak bulunamadı. Lütfen modeli manuel olarak indirin.")
            return

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

        for comment in self.data_cleaned['Comment']:
            try:
                input_text = f"Hasta şikayet: {comment}\nTedavi önerisi:"
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                gen_tokens = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.3)
                gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                self.treatment_suggestions.append(gen_text.split("Tedavi önerisi:")[-1].strip())
            except Exception as e:
                print(f"Tedavi Önerisi Hatası: {e}")
                self.treatment_suggestions.append("Bilgi yetersiz")

    def calculate_bleu_and_rouge(self):
        #BLEU ve ROUGE skorlarını hesapla
        print("\nBLEU ve ROUGE Skorları Hesaplanıyor...")
        rouge = Rouge()
        for comment, suggestion in zip(self.data_cleaned['Comment'], self.treatment_suggestions):
            bleu_score = sentence_bleu([comment.split()], suggestion.split(), smoothing_function=SmoothingFunction().method1)
            rouge_score = rouge.get_scores(suggestion, comment)[0]['rouge-l']['f']
            self.bleu_scores.append(bleu_score)
            self.rouge_scores.append(rouge_score)

    def save_results(self):
        #Sonuçları Excel dosyasına kaydet
        self.data_cleaned['Alignment Score'] = self.alignment_scores
        self.data_cleaned['Treatment Suggestion'] = self.treatment_suggestions
        self.data_cleaned['BLEU Score'] = self.bleu_scores
        self.data_cleaned['ROUGE Score'] = self.rouge_scores
        self.data_cleaned.to_excel(self.output_path, index=False)
        print(f"\nSonuçlar başarıyla kaydedildi: {self.output_path}")

    def run(self):
        self.load_and_clean_data()
        self.check_images()
        self.preprocess_images()
        self.analyze_alignment()
        self.generate_treatment_suggestions()
        self.calculate_bleu_and_rouge()
        self.save_results()
