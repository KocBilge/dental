import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
import torch
import warnings
import gc
import psutil
import time
import torch.multiprocessing as mp
from accelerate import load_checkpoint_and_dispatch

# Bellek ve GPU önbelleği temizleme
gc.collect()
torch.cuda.empty_cache()

# TensorFlow bilgi mesajlarını kapama
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Multiprocessing başlangıç yöntemi
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
mp.set_start_method("spawn", force=True)

# GPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")

warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be .* leaked semaphore objects")
warnings.filterwarnings("ignore", category=UserWarning, message="Current model requires .* bytes of buffer")
warnings.filterwarnings("ignore", category=UserWarning, message="for model.layers.*")

# Hugging Face zaman aşımı ve chunk boyutu ayarları
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "14400"
os.environ["HF_HUB_CHUNK_SIZE"] = "1048576"  # 1 MB

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Anlık Bellek Kullanımı: {memory_info.rss / (1024 * 1024):.2f} MB")

def monitor_resources():
    process = psutil.Process()
    while True:
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"Anlık Bellek Kullanımı: {memory_usage:.2f} MB | CPU Kullanımı: {cpu_usage}%")
        time.sleep(2)

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
        self.few_shot_count = 0
        self.zero_shot_count = 0

    def load_and_clean_data(self):
        print("\nVeri seti yükleniyor ve temizleniyor...")
        log_memory_usage()
        self.data = pd.read_excel(self.data_path, header=1)
        self.data_cleaned = self.data.dropna()
        print(f"Temizlenmiş Veri Seti Boyutu: {self.data_cleaned.shape[0]} satır")

    def check_images(self):
        print("\nGörseller kontrol ediliyor...")
        for image_name in self.data_cleaned['Image']:
            full_path = os.path.join(self.image_folder, image_name)
            if not os.path.exists(full_path):
                print(f"Eksik Görsel: {image_name}")
                self.missing_images.append(image_name)
        if self.missing_images:
            print(f" Eksik Görseller Listesi: {self.missing_images}")

    def preprocess_images(self):
        print("\nGörseller işleniyor...")
        log_memory_usage()
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
        log_memory_usage()
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
        model_path = "C:\\Users\\User\\Downloads\\aya-expanse-32b"
        if not os.path.exists(model_path):
            print("Model yerel olarak bulunamadı. Lütfen modeli manuel olarak indirin.")
            self.treatment_suggestions = ["Model Bulunamadı"] * len(self.data_cleaned)
            return

        print("Tokenizer yükleniyor...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Tokenizer başarıyla yüklendi.")

        print("Model yüklenmeye başlanıyor (disk_offload kullanılarak)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True
        )
        print("Model yüklemesi tamamlandı.")

        batch_size = 10
        for i in range(0, len(self.data_cleaned), batch_size):
            batch = self.data_cleaned.iloc[i:i + batch_size]
            for idx, comment in enumerate(batch['Comment']):
                try:
                    print(f"[{i+idx+1}/{len(self.data_cleaned)}] Şu anda işlenen yorum: {comment[:50]}...")
                    input_text = f"Hasta şikayet: {comment}\nTedavi önerisi:"
                    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
                    print(f"Input IDs: {input_ids}")

                    start_time = time.time()
                    gen_tokens = model.generate(
                        input_ids,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7
                    )
                    gen_time = time.time() - start_time
                    print(f"Generate Time: {gen_time:.2f} seconds")
                    print(f"Generated Tokens: {gen_tokens}")

                    suggestion = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                    print(f"Generated Suggestion: {suggestion}")
                    self.treatment_suggestions.append(suggestion.split("Tedavi önerisi:")[-1].strip())
                except Exception as e:
                    print(f"Tedavi önerisi oluşturulamadı: {e}")
                    self.treatment_suggestions.append("Hata")
            cleanup()

    def save_results(self):
        print("\nSonuçlar kaydediliyor...")
        assert len(self.data_cleaned) == len(self.treatment_suggestions), "Veri uzunlukları eşleşmiyor!"
        self.data_cleaned['Alignment Score'] = self.alignment_scores
        self.data_cleaned['Treatment Suggestion'] = self.treatment_suggestions
        self.data_cleaned.to_excel(self.output_path, index=False, engine='openpyxl')
        print(f"\nSonuçlar başarıyla kaydedildi: {self.output_path}")

    def run(self):
        self.load_and_clean_data()
        self.check_images()
        self.preprocess_images()
        self.analyze_alignment()
        self.generate_treatment_suggestions()
        self.save_results()
        cleanup()

if __name__ == "__main__":
    TOKEN = "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"
    analysis = DentalAnalysis(
        data_path="C:\\Users\\User\\Downloads\\test.xlsx",
        image_folder="C:\\Users\\User\\Downloads\\dental_project.v1i.yolov11\\train\\images",
        output_path="C:\\Users\\User\\Downloads\\cleaned_data_with_results.xlsx",
        token=TOKEN
    )

    monitor_process = mp.Process(target=monitor_resources)
    monitor_process.start()

    analysis.run()

    monitor_process.terminate()
    cleanup()
