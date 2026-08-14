# dashboard.py
import streamlit as st
import requests

st.set_page_config(page_title="QILLM Studio", layout="wide", page_icon="⚛️")

if 'stage' not in st.session_state:
    st.session_state.stage = 'setup'
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

st.title("⚛️ QILLM Studio: Model Optimizasyon Platformu")
st.markdown("Kendi modellerinizi getirin, küçültün, farklı yöntemlerle iyileştirin ve canlı test edin.")
st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ Parametreler ve Model")
    
    model_source = st.radio("Model Kaynağını Seçin:", ["Hazır Liste", "HuggingFace ID", "Bilgisayardan Yükle"])
    
    model_name = ""
    is_ready_to_compress = True

    if model_source == "Hazır Liste":
        model_name = st.selectbox("Hedef Model", ["Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3-8B", "mistralai/Mistral-7B-v0.1"])
    elif model_source == "HuggingFace ID":
        model_name = st.text_input("HuggingFace Model ID", placeholder="Örn: google/gemma-2b")
        if not model_name: is_ready_to_compress = False
    else:
        uploaded_file = st.file_uploader("Model Ağırlıklarını Yükle (.pt, .safetensors)", type=["pt", "safetensors", "bin"])
        if uploaded_file:
            model_name = uploaded_file.name
            st.success(f"{model_name} belleğe alındı.")
        else:
            is_ready_to_compress = False

    ratio = st.slider("Sıkıştırma Oranı (r)", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
    
    if st.session_state.stage == 'setup':
        if is_ready_to_compress:
            analyze_res = requests.post("http://127.0.0.1:8000/analyze", json={"ratio": ratio, "model_name": model_name}).json()
            
            if analyze_res.get("is_small") and ratio < 0.5:
                st.warning(analyze_res.get("warning_message", "Küçük modellerde dikkatli olun."))
                risk_accepted = st.checkbox("Riski anlıyorum, sıkıştırmaya devam et.")
            else:
                risk_accepted = True
                
            if st.button("🚀 Modeli Sıkıştır", disabled=not risk_accepted):
                with st.spinner(f"{model_name} QILLM C++ Motoru ile Sıkıştırılıyor..."):
                    if model_source == "Bilgisayardan Yükle" and uploaded_file:
                        requests.post("http://127.0.0.1:8000/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue())})
                    
                    res = requests.post("http://127.0.0.1:8000/compress", json={"ratio": ratio, "model_name": model_name})
                    if res.status_code == 200:
                        st.session_state.metrics = res.json()
                        st.session_state.stage = 'compressed'
                        st.rerun()
                    else:
                        st.error("API Hatası: Sunucuya ulaşılamadı.")

    elif st.session_state.stage == 'compressed':
        ppl_val = st.session_state.metrics.get('compressed_ppl', 'Bilinmiyor')
        st.error(f"⚠️ PPL değeri {ppl_val} seviyesine çıktı.")
        
        dl_res = requests.get("http://127.0.0.1:8000/download")
        st.download_button("📥 Sıkıştırılmış Modeli İndir (.pt)", data=dl_res.content, file_name=f"qillm_{model_name.replace('/', '_')}_comp.pt")
        
        st.divider()
        
        st.subheader("🛠️ İyileştirme (Recovery) Stratejisi")
        
        with st.expander("❓ İyileştirme Yöntemleri Ne İşe Yarar? (Açıklamalar)"):
            st.markdown("""
            * **LoRA (Hızlı İnce Ayar):** Sıkıştırılmış modelin dondurulmuş (frozen) ağırlıklarının yanına çok küçük, eğitilebilir 'ek beyin modülleri' takar. Hızlıdır ve donanım bütçesi olan standart durumlar için en idealidir.
            * **QLoRA (Kuantize Düşük Bellek):** Modeli 4-bit seviyesine indirgeyerek LoRA uygular. 7B ve 8B gibi devasa modelleri tek bir standart GPU'da eğitmek (RAM çökmesini önlemek) için zorunludur.
            * **Knowledge Distillation (Öğretmen-Öğrenci):** Orijinal (Öğretmen) modelin, hasar görmüş (Öğrenci) modele ders vermesidir. %70'ten fazla budama (r < 0.3) yapılan ağır hasarlı modelleri toparlamak için en kaliteli, ancak en yavaş yöntemdir.
            """)

        recommended_method = "LoRA (Hızlı İnce Ayar)" 
        
        if ratio <= 0.3:
            recommended_method = "Knowledge Distillation (Öğretmen-Öğrenci)"
        elif "7B" in model_name or "8B" in model_name:
            recommended_method = "QLoRA (Kuantize Düşük Bellek)"

        methods = [
            "LoRA (Hızlı İnce Ayar)", 
            "QLoRA (Kuantize Düşük Bellek)", 
            "Knowledge Distillation (Öğretmen-Öğrenci)"
        ]
        
        display_options = []
        for m in methods:
            if m == recommended_method:
                display_options.append(f"{m} ⭐ (Sistem Önerisi)")
            else:
                display_options.append(m)

        default_idx = display_options.index(f"{recommended_method} ⭐ (Sistem Önerisi)")
        selected_display = st.selectbox("Durumunuza Uygun Yöntemi Seçin", display_options, index=default_idx)
        
        rec_method_clean = selected_display.replace(" ⭐ (Sistem Önerisi)", "")
        
        if st.button("🧠 Modeli İyileştir"):
            with st.spinner(f"{rec_method_clean} uygulanıyor..."):
                rec_res = requests.post("http://127.0.0.1:8000/recover", json={"model_name": model_name, "method": rec_method_clean})
                if rec_res.status_code == 200:
                    st.session_state.metrics.update(rec_res.json())
                    st.session_state.stage = 'recovered'
                    st.rerun()

    elif st.session_state.stage == 'recovered':
        method_used = st.session_state.metrics.get('method_used', 'Seçilen Yöntem')
        st.success(f"✅ Model başarıyla sıkıştırıldı ve {method_used} ile toparlandı!")
        
        dl_res = requests.get("http://127.0.0.1:8000/download")
        st.download_button("📥 Nihai Modeli İndir (.pt)", data=dl_res.content, file_name=f"qillm_{model_name.replace('/', '_')}_final.pt", type="primary")
        
        if st.button("🔄 Yeni Bir Model Optimize Et"):
            st.session_state.stage = 'setup'
            st.rerun()

with col_right:
    st.subheader("📊 Canlı Metrik Karşılaştırması")
    
    if st.session_state.stage == 'setup':
        st.info("Sıkıştırma işlemi henüz başlamadı. Parametreleri seçip ilerleyin.")
        
    else:
        c1, c2, c3 = st.columns(3)
        m = st.session_state.metrics
        
        c1.metric("1. Orijinal Model", f"{m.get('original_vram_mb', 0)} MB", f"PPL: {m.get('original_ppl', 0)}")
        c2.metric("2. Sıkıştırılmış", f"{m.get('compressed_vram_mb', 0)} MB", f"PPL: {m.get('compressed_ppl', 0)}", delta_color="inverse")
        
        if st.session_state.stage == 'recovered':
            c3.metric(f"3. {m.get('method_used', 'İyileştirilmiş')}", f"{m.get('recovered_vram_mb', 0)} MB", f"PPL: {m.get('recovered_ppl', 0)}")
            
        st.divider()
        
        st.subheader("🧪 Canlı Çıkarım Testi (Playground)")
        prompt = st.text_area("Modele bir soru sorun:", value="Kuantum mekaniği ve yapay zeka")
        
        if st.button("Üret (Generate)"):
            with st.spinner("Model yanıt üretiyor..."):
                current_state = "recovered" if st.session_state.stage == "recovered" else "compressed"
                
                res_orig = requests.post("http://127.0.0.1:8000/inference", json={"prompt": prompt, "model_state": "original"}).json()
                res_curr = requests.post("http://127.0.0.1:8000/inference", json={"prompt": prompt, "model_state": current_state}).json()
                
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown("**Orijinal Model Yanıtı:**")
                    st.info(res_orig.get("output", "Yanıt alınamadı."))
                    st.caption(f"⚡ Hız: {res_orig.get('speed', 0)} token/sn")
                    
                with tc2:
                    st.markdown(f"**{current_state.capitalize()} Model Yanıtı:**")
                    if current_state == "compressed":
                        st.error(res_curr.get("output", "Yanıt alınamadı."))
                    else:
                        st.success(res_curr.get("output", "Yanıt alınamadı."))
                    st.caption(f"⚡ Hız: {res_curr.get('speed', 0)} token/sn")