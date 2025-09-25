## Proje Özeti

EyesOnYou, sınıf ortamındaki öğrencileri gerçek zamanlı tespit eden, her birine kalıcı dijital kimlik atayan ve sahnedeki benzersiz öğrenci sayısını sürekli güncelleyen bir bilgisayarla görü prototipidir. Proje, kamera akışındaki insanları yakalayıp tanımlayarak, geçici kaybolmalarda bile kimlik sürekliliğini korur ve anlık öğrenci varlığını güvenilir biçimde raporlar.

## Teknoloji Yığını ve Mimari

### Tespit - Yolov8s

- **Neden bu seçim?** YOLOv8s modeli tek GPU üzerinde yüksek FPS sağlayarak hızlı ve verimli çalışır. Eğitim, dışa aktarma (ONNX, TensorRT) ve entegrasyon adımlarında güçlü topluluk desteği vardır. Diğer modeller daha yüksek doğruluk sağlasa da, YOLOv8s hız ve doğruluk arasında dengeli bir çözümdür.

### Takip - StrongSort + OSNet_x0_25
- **Neden bu seçim?** StrongSort, SORT/DeepSORT mirasını geliştirerek hem hareket hem görünüm bilgisi kullanır. OSNet tabanlı hafif ReID ağı sayesinde kısmi örtüşmelerde kimlikleri korumada güçlüdür. BoxMOT kütüphanesi StrongSort'u OSNet_x0_25 ağıyla birlikte getiriyor; ağırlık dosyaları ihtiyaç halinde otomatik indiriliyor.
- **Alternatifler:** ByteTrack daha hızlıdır fakat görünüm bilgisi olmadığı için kimlik kaybı yaşatabilir. BoT-SORT güçlüdür ancak fazladan kalibrasyon ister. StrongSort, dengeli varsayılan ayarlarla üretime hızlı geçiş sağlar.

### Yardımcı Kütüphaneler - Supervision ve OpenCV
- **supervision:** YOLO çıktılarının StrongSort girdisine dönüştürülmesini hazır fonksiyonlarla yapar, her kareyi yeniden düzenlemek için ek kod yazma ihtiyacını ortadan kaldırır.
- **OpenCV:** RTSP, USB kamera veya video dosyasından kareleri çeker, temel görüntü işleme fonksiyonlarını sağlar. 


## Dosya Yapısı

- **src/:** Uygulamanın ana kaynak kodları 

  - **config.py:** Model yolları, eşik değerleri ve giriş/çıkış ayarlarını tutan yapılandırma dosyası  
  - **main.py:** YOLOv8s ile tespit, StrongSORT ile takip ve sayım akışını yöneten ana uygulama dosyası

- **weights/** → StrongSORT için gerekli re-identification ağırlıklarının saklandığı klasör (ilk çalıştırmada otomatik indirilir). 

- **.gitignore** → Geçici veya büyük dosyaların sürüm kontrolüne eklenmesini engeller.  

- **requirements.txt** → Projenin çalışması için gerekli Python paketlerinin listesi.  

- **README.md** → Proje özeti, mimari açıklamalar ve çalışma talimatlarını içerir.  

- **GELISTIRICI_NOTLARI.md** → Geliştiriciler için ileri seviye eğitim, ince ayar ve ek notları barındıran rehber.  

## Kurulum ve Çalıştırma Adımları

1. **Python ortamını hazırlayın:** Python 3.10+ sürümü yüklü olmalıdır. İzole bir sanal ortam oluşturmak için python -m venv .venv komutunu kullanabilirsiniz.
2. **Sanal ortamı etkinleştirin:** Windows PowerShell için ./.venv/Scripts/Activate.ps1, Unix tabanlı sistemler için source .venv/bin/activate.
3. **Bağımlılıkları yükleyin:** pip install --upgrade pip ardından pip install -r requirements.txt.
4. **Test videosunu yerleştirin:** data/videos/ klasörüne sinif_videosu.mp4 adında bir örnek video kopyalayın.
5. **Uygulamayı çalıştırın:** python src/main.py --kaynak data/videos/sinif_videosu.mp4 komutu, betiğin sağladığı kaynak argümanını kullanarak prototipi başlatır. Kamera veya RTSP akışı kullanmak için aynı argümana ilgili URI verilebilir.
6. **Sonuçları izleyin:** Betik, gerçek zamanlı kareleri ve kimlik etiketlerini pencere üzerinde gösterir; terminalde ise aktif kimlik sayısı raporlanır.

## Gelecek Geliştirmeler (Yol Haritası)
- **Yüz tanıma entegrasyonu:** Öğrenci kimliklerini yüz tabanlı doğrulama ile eşleştirerek yoklama otomasyonuna zemin hazırlamak.
- **Web arayüzü:** FastAPI + WebSocket üzerine kurulacak bir pano ile canlı sayım, geçmiş raporlar ve uyarı yönetimi.
- **Çoklu kamera senkronizasyonu:** Birden fazla sınıf veya açıdan gelen akışların tek kimlik deposunda birleştirilmesi.
- **Model optimize etme:** TensorRT ile hızlandırılmış sürümler, INT8 quantization ve edge cihaz dağıtımları.
- **Gelişmiş analitik:** Giriş-çıkış kapıları, ısı haritaları, dikkat bölgeleri gibi ikinci seviye davranış analizleri.
