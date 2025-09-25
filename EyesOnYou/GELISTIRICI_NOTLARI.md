## YOLOv8 Modelini Geliştirme Rotası

### Fine-tuning Nedir ve Neden Önemlidir?

**Fine-tuning**, geniş veri kümeleri üzerinde önceden eğitilmiş YOLOv8s modelini alıp, daha dar ve özel bir domaine (ör. kendi okulunuzdaki öğrenciler) uyarlama sürecidir.  
Bu sayede model:  
- Ortama özgü ışık koşullarını,  
- Öğrencilerin üniforma / kıyafet çeşitliliğini,  
- Sınıfın yerleşim düzenini  
öğrenir ve optimize eder.  

### Veri Toplama ve Etiketleme Stratejisi

1. **Veri Toplama**  
   - Farklı sınıflar, ışık koşulları ve oturma düzenlerini kapsayan 10–15 dakikalık kısa videolar çekin.  

2. **Kare Örnekleme**  
   - Videolardan her 10–15 karede bir görüntü almak için `ffmpeg` veya `supervision` araçlarını kullanın.

3. **Etiketleme Araçları**  
   - **Roboflow Annotate:** Ekip çalışması, veri versiyonlama ve otomatik augmentasyon için uygun.  
   - **LabelImg / CVAT:** Açık kaynak ve çevrimdışı etiketleme ihtiyacı olan ekipler için güçlü alternatifler.  

4. **Sınıf Şeması**  
   - İlk aşamada yalnızca **person** sınıfını kullanmak yeterlidir.  
   - İlerleyen aşamalarda “öğrenci”, “öğretmen” gibi alt sınıflar eklenebilir. .

### Ultralytics ile Eğitim Yapmayı Öğrenin - Kaynaklar ve Derinleşme İçin Linkler

- [Ultralytics YOLOv8 Dokümantasyonu](https://docs.ultralytics.com/)
- [Roboflow Blog: Object Detection için Veri Toplama](https://blog.roboflow.com/data-collection-guide/)
- [Albumentations augmentasyon rehberi](https://albumentations.ai/docs/)

## StrongSort Takip Algoritmasını Güçlendirme

### Re-Identification (ReID) ve Appearance Embedding Nedir?

- **ReID:** Bir kişinin görüntüsünden kimlik temsilini çıkarıp, farklı karelerde aynı kişiyi yeniden tanıma problemidir.
- **Appearance Embedding:** ReID ağının ürettiği 256-512 boyutlu vektörlerdir. Bu vektörler StrongSort içinde kullanılan mesafe metriğinde aynı kişiye ait kutuların eşleştirilmesini sağlar. Hareket modeli (Kalman filtresi) kısa süreli kayıpları telafi ederken, görünüm vektörü kimliklerin karışmasını önler.

### Kendi ReID Modelini Eğitme ve Entegre Etme - Derinleşmek İçin Kaynaklar

1. **Veri seti oluştur:** Öğrencilerin farklı açılardan, farklı ışık koşullarında ve farklı aksesuarlarla (çanta, şapka) görüntülerini topla. Her öğrenci için en az 10-15 görüntü hedefle.
2. **Veriyi etiketle:** ReID için klasör isimleri genellikle kişi ID'si ile eşleşir (dataset/person_001/img1.jpg).
3. **OSNet tabanlı eğitim:**
   - [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) kütüphanesini kullanarak osnet_x0_25 mimarisini kendi verinle eğitebilirsin.
   - Eğitimden sonra elde ettiğin .pth ağı, BoxMOT/StrongSort konfigürasyonundaki 
eid_model_path yoluna yerleştir.
4. **StrongSort ayarlarını güncelle:** src/config.py içindeki TrackerConfig.reid_model_path, max_distance, max_iou_distance, mc_lambda gibi alanları veri setine göre yeniden ayarla.
5. **Saha testi yap:** Yeni ReID ağıyla kalabalık sahnelerde kaç kimlik kırılması yaşandığını ölç, önceki sürümle karşılaştır.

- [StrongSORT Resmi Deposu](https://github.com/dyhBUPT/StrongSORT)
- [BoxMOT Dokümantasyonu](https://github.com/mikel-brostrom/boxmot)
- [OSNet Makalesi: Omni-Scale Feature Learning for Person Re-ID](https://arxiv.org/abs/1905.00953)
- [BoT-SORT ve ByteTrack karşılaştırması](https://arxiv.org/abs/2303.11977)
- [Torchreid Belgeleri](https://kaiyangzhou.github.io/deep-person-reid/)

