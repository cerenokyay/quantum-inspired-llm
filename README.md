

# ⚛️ QILLM Studio: Quantum-Inspired LLM Optimization Platform

QILLM (Quantum-Inspired Large Language Model) Studio, devasa yapay zeka modellerini (LLM) yüksek performanslı C++ motoru ile sıkıştıran, analiz eden ve akıllı stratejilerle iyileştiren uçtan uca bir model optimizasyon platformudur.

Ağır matematiksel tensör işlemlerini, kullanıcı dostu ve sezgisel bir arayüzle birleştirerek yapay zeka mühendisliğini erişilebilir bir ürün deneyimine dönüştürür.

---

## 🚀 Ürün Vizyonu ve Temel Özellikler

* **Evrensel Model Desteği (Agnostic Wrapper):** HuggingFace ekosistemindeki hazır modelleri (Örn: Llama-3, Qwen, Mistral) veya bilgisayarınızdaki yerel `.pt` / `.safetensors` dosyalarını tek tıkla sisteme entegre edebilirsiniz.
* **Akıllı İyileştirme Motoru (Smart Recovery):** Sıkıştırma sonrası modelin zeka (PPL) kaybını analiz eder. Budama oranına ve modelin parametre boyutuna göre kullanıcıya en uygun iyileştirme yöntemini (LoRA, QLoRA veya Knowledge Distillation) otomatik olarak önerir.
* **Canlı Çıkarım Test Alanı (Playground):** Orijinal model ile optimize edilmiş modelin üretim hızını (token/sn) ve çıktı kalitesini yan yana, gerçek zamanlı olarak test etmenize olanak tanır.
* **Anlık Metrik Karşılaştırması:** VRAM tasarrufu, bellek ayak izi ve Perplexity (PPL) değerlerini görsel kartlar halinde sunarak maliyet/performans (trade-off) kararlarını veriye dayalı hale getirir.
* **Yüksek Performanslı Çekirdek:** Arka planda Apple Accelerate framework'ü ile entegre çalışan, donanıma optimize edilmiş özel bir C++ SVD (Singular Value Decomposition) matris ayrıştırma motoru kullanır.

---

## 🏗️ Mimari ve Teknoloji Yığını

* **Kullanıcı Arayüzü (Frontend):** Streamlit (İnteraktif Dashboard, Dinamik Veri Akışı)
* **API ve Sunucu (Backend):** FastAPI, Uvicorn, Pydantic (Asenkron İstek Yönetimi, Dosya Yükleme)
* **Yapay Zeka ve Matematik Motoru:** PyTorch, Transformers, C++ (PyBind11)
* **Donanım Optimizasyonu:** Apple Silicon (M-Series) için Accelerate Framework entegrasyonu

---

## 💻 Kurulum ve Çalıştırma

Platformu kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

**1. Depoyu Klonlayın ve Sanal Ortamı Aktif Edin**

```bash
git clone https://github.com/KULLANICI_ADIN/quantum-inspired-llm.git
cd quantum-inspired-llm
source venv/bin/activate

```

**2. Gerekli Kütüphaneleri Yükleyin**

```bash
pip install -r requirements.txt

```

*(Not: Proje içinde fastapi, uvicorn, streamlit, python-multipart ve torch kütüphanelerinin kurulu olduğundan emin olun).*

**3. C++ Motorunu Derleyin**

```bash
python setup.py install

```

**4. Sistemi Ayağa Kaldırın**
QILLM Studio iki ayrı süreç olarak çalışır. Terminalinizde iki farklı sekme açın:

* **Sekme 1 (Arka Plan API Sunucusu):**

```bash
python app.py

```

* **Sekme 2 (Görsel Arayüz):**

```bash
streamlit run dashboard.py

```

Tarayıcınızda açılan `http://localhost:8501` adresi üzerinden platformu hemen kullanmaya başlayabilirsiniz.

---

## 🧪 Sıkıştırma ve İyileştirme Akışı

Sistem, kullanıcıları şu adımlarla yönlendirir:

1. **Analiz:** Seçilen modelin boyutu ve hedeflenen $r$ (sıkıştırma oranı) değerlendirilir. Gerekirse risk uyarısı yapılır.
2. **Budama (SVD):** HuggingFace mimarisindeki Attention ve MLP katmanları C++ motoruna gönderilerek matris boyutları küçültülür.
3. **Kurtarma (Recovery):** Bozulan ağırlıklar, sistemin önerdiği strateji ile (LoRA vb.) hızlıca yeniden eğitilerek zeka kaybı (PPL) geri kazanılır.
4. **Dışa Aktarma:** Optimize edilen nihai model bilgisayara indirilmeye hazır hale gelir.

---

## 👤 Geliştirici

**Ceren Okyay**

Yazılım ve Ürün Geliştirme Süreçleri | C++ & Python
*(Bu proje, ağır mühendislik çözümlerinin kullanıcı odaklı, ölçeklenebilir ürünlere nasıl dönüştürülebileceğini göstermek amacıyla geliştirilmiştir.)*