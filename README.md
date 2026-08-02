
```markdown
# ⚛️ Quantum-Inspired LLM Architecture (`qillm`)
*A hybrid C++ / Python deep learning library leveraging Quantum Tensor Networks for LLM compression.*
*(LLM sıkıştırması için Kuantum Tensör Ağlarını kullanan hibrit C++ / Python derin öğrenme kütüphanesi.)*

---

## 🇬🇧 English Documentation

A hybrid C++ / Python deep learning library that leverages **Quantum Tensor Networks (Matrix Product States)** and **Singular Value Decomposition (SVD)** to compress attention mechanisms in Large Language Models. Accelerated natively on **Apple Silicon (M-Series MPS / Accelerate Framework)**.

### 🌟 Key Features
* **C++ Native Extension**: Core Tensor Contraction and SVD Truncation implemented in C++ via PyBind11.
* **Hardware Acceleration**: Deep integration with Apple's `Accelerate` framework for M-Series architecture.
* **Configurable Truncation Rate**: Dynamic quantum compression ratio ($0.0 < r \le 1.0$) to balance memory footprint and model expressivity.
* **End-to-End Pipeline**: Includes custom Tokenizer, Training Loop, Autoregressive Inference, and Checkpointing (`.pt`).

### 📊 Benchmark & Trade-offs
| Attention Layer | Compression Ratio | Primary Advantage |
| :--- | :---: | :--- |
| **Standard PyTorch Attention** | 0% | Fast execution for small sequences. |
| **C++ Quantum SVD Motor** | **50%** | **Massive memory reduction & dense state compression.** |

---

## 🇹🇷 Türkçe Dökümantasyon

Büyük Dil Modellerindeki (LLM) "Attention" (Dikkat) mekanizmalarını sıkıştırmak için **Kuantum Tensör Ağları (Matrix Product States)** ve **Tekil Değer Ayrışımı (SVD)** kullanan hibrit bir C++ / Python derin öğrenme kütüphanesi. **Apple Silicon (M-Serisi MPS / Accelerate Framework)** üzerinde donanımsal olarak hızlandırılmıştır.

### 🌟 Öne Çıkan Özellikler
* **C++ Motoru**: Temel Tensör Büzülmesi (Contraction) ve SVD Budama işlemleri PyBind11 aracılığıyla doğrudan C++ ile yazılmıştır.
* **Donanım Hızlandırması**: M-Serisi mimarisi için Apple'ın yerleşik `Accelerate` kütüphanesiyle derin entegrasyon.
* **Dinamik Sıkıştırma Oranı**: Bellek ayak izi ile model doğruluğunu dengelemek için dışarıdan ayarlanabilir kuantum sıkıştırma oranı ($0.0 < r \le 1.0$).
* **Uçtan Uca Mimari**: Özel Tokenizer, Eğitim Döngüsü, Otoregresif Metin Üretimi ve Ağırlık Kaydetme (`.pt`) süreçlerini içerir.

### 📊 Performans ve Mühendislik Kazanımları
| Katman Tipi | Sıkıştırma Oranı | Temel Avantaj |
| :--- | :---: | :--- |
| **Standart PyTorch Attention** | %0 | Küçük metin uzunluklarında (sequence) yüksek hız. |
| **C++ Kuantum SVD Motoru** | **%50** | **Devasa bellek tasarrufu ve kuantum durum sıkıştırması.** |

---

## 🏗️ Architecture Overview / Mimari Genel Bakış

```text
[ Input Text ] ──> [ Custom Tokenizer ] ──> [ Embedding + Positional Encoding ]
                                                       │
                                                       ▼
                                     [ Quantum-Inspired Attention (C++) ]
                                      ├── Q, K, V Linear Projection
                                      ├── M-Series Accelerate SVD (S_truncated)
                                      └── Tensor Contraction (Compress Matrix)
                                                       │
                                                       ▼
                                     [ Feed Forward (FFN) + LM Head ] ──> [ Next Token ]

```

---

## 🚀 Quick Start / Hızlı Başlangıç

### 1. Installation / Kurulum

Clone the repository and build the C++ extension in editable mode:
*(Projeyi klonlayın ve C++ eklentisini geliştirici modunda derleyin:)*

```bash
git clone [https://github.com/cerenokyay/quantum-inspired-llm.git](https://github.com/cerenokyay/quantum-inspired-llm.git)
cd quantum-inspired-llm

# Create and activate virtual environment (Sanal ortam oluşturma ve aktivasyon)
python3 -m venv venv
source venv/bin/activate

# Build C++ extension (C++ motorunu derleme)
pip install -e .

```

### 2. Training / Model Eğitimi

Train the Quantum-LLM on custom text data:
*(Kuantum-LLM modelini özel veri seti üzerinde eğitin:)*

```bash
python train.py

```

### 3. Inference / Metin Üretimi

Generate text autoregressively using the saved checkpoint:
*(Kaydedilmiş ağırlıkları kullanarak otoregresif metin üretin:)*

```bash
python generate.py

```

### 4. Benchmarking / Performans Testi

Compare execution metrics between standard PyTorch and C++ Quantum SVD:
*(Standart PyTorch ile C++ Kuantum SVD arasındaki performans metriklerini karşılaştırın:)*

```bash
python benchmark.py

```

---

## 🛠️ Tech Stack / Teknoloji Yığını

* **Languages:** C++17, Python 3.10+
* **Frameworks:** PyTorch, PyBind11, Apple Accelerate Framework
* **Tools:** Setuptools, CMake/Make

```

```