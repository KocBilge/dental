DentalAI: Diş Sağlığı Analiz ve Tedavi Öneri Sistemi

DentalAI projesi, diş sağlığıyla ilgili görsel ve metinsel verilerin analiz edilmesi, yorumlanması ve tedavi önerilerinin oluşturulması amacıyla geliştirilmiş bir yapay zeka çözümüdür. Bu proje, CLIP, GPT-4, ve AYA modellerini kullanarak diş sağlığı verilerini işler, analiz eder ve öneriler sunar.

* Projenin Amacı
  
    - Diş imageleri üzerinden analiz yapmak.
    - Hasta yorumları ve şikayetleri üzerinden tedavi önerileri oluşturmak.
    - Yapay zeka modelleriyle hızlı ve doğru tedavi önerileri sunmak.
    - Excel veri setlerini analiz edip anlamlı çıktılar sağlamak.

* Kullanılan Teknolojiler ve Modeller

  1. CLIP (openai/clip-vit-large-patch14-336)
    - Görsel ve metin uyumu analiz edilir.
    - Hastaların yorumları ve ilgili diş görselleri eşleştirilir.
  2. GPT-4 (OpenAI)
    - Hasta şikayetleri metin olarak analiz edilir.
    - GPT-4 modeli tedavi önerileri sunar.
  3. AYA (aya-expanse-32b)
    - Alternatif olarak AYA modeliyle de tedavi önerileri oluşturulur.
  4. Pandas & OpenPyXL
    - Excel veri setlerinin okunması, temizlenmesi ve çıktıların kaydedilmesi.
  5. PyTorch & Transformers
    - Modellerin yüklenmesi ve çalıştırılması için PyTorch ve Transformers kütüphaneleri kullanıldı.
  6. Pillow (PIL)
    - Görsellerin işlenmesi ve boyutlandırılması sağlandı.

* Veri Seti (Excel)
  
  - Image: Diş görselinin adı.
  - Comment: Hastanın şikayet açıklaması.
  - Alignment Score: Görsel ve metin eşleşme uyumu.
  - Treatment Suggestion: Yapay zeka tarafından önerilen tedavi yöntemi.
    
⚙ Çalışma Akışı

Veri Seti Yükleme: 
  - Excel verileri yüklenir ve temizlenir.
  - Görsel Kontrol: Eksik görseller tespit edilir ve raporlanır.
  - Görsel İşleme: Görseller yeniden boyutlandırılır ve analiz için hazırlanır.
  - CLIP Analizi: Görseller ve yorumlar eşleştirilir ve uyum skorları hesaplanır.

Tedavi Önerileri:
  - GPT-4 ile tedavi önerileri oluşturulur.
  - Alternatif olarak AYA modeli de kullanılabilir.
  - Sonuçların Kaydedilmesi: Excel dosyası olarak çıktı alınır.

* Çalıştırma Talimatları

  1. Gerekli Paketleri Yükleyin:
    pip install -r requirements.txt
  2. Ortam Değişkenlerini Ayarlayın:
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
  3. Proje Betiklerini Çalıştırın:
    GPT-4 ve CLIP:
    python dentalAI-GPT-4.py
    AYA ve CLIP:
    python dentalAI-AYA.py
  4. Sonuçları İnceleyin:
    Sonuçlar cleaned_data.xlsx dosyasında bulunur.

* Sonuç

DentalAI, diş sağlığı verilerini analiz ederek modern yapay zeka araçlarıyla anlamlı çıktılar üretir. Tedavi önerileri ve görsel analizler, diş sağlığı uzmanlarına değerli içgörüler sağlar.
