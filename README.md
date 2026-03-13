# Hipertansif Retinopati (HR) Tespit Uygulaması

Bu proje, fundus görüntülerini kullanarak **Hipertansif Retinopati** (Hypertensive Retinopathy) hastalığını tespit eden AI destekli bir masaüstü uygulamasıdır. Kullanıcı dostu, modern bir arayüz ile tıbbi görüntülerin analiz edilmesini ve sağlık profesyonellerine karar destek mekanizması sunulmasını amaçlamaktadır.

## Özellikler

- **AI Destekli Sınıflandırma**: İkili (Binary) sınıflandırma yaparak görüntünün Hipertansif Retinopati bulguları taşıyıp taşımadığını tespit eder.
- **Damar Segmentasyonu (Maskeleme)**: UNet++ mimarisi kullanarak fundus görüntülerindeki damar yapısını çıkarır ve kullanıcıya sunar.
- **Kapsamlı Model Mimarisi**: Sınıflandırma işlemleri için çift omurgalı (DenseNet121 ve EfficientNetB0) bir model kullanarak yüksek doğruluk oranlarını hedefler.
- **Modern UI**: PyQt5 ile geliştirilmiş, karanlık tema (dark-theme) ve glassmorphism efektleri barındıran kullanıcı dostu masaüstü arayüzü.

## Teknolojiler / Araçlar

- **Programlama Dili**: Python 3.x
- **Derin Öğrenme Frameworkleri**: PyTorch
- **Modeller**: DenseNet121, EfficientNetB0, UNet++
- **Arayüz (GUI)**: PyQt5
- **Görüntü İşleme**: OpenCV, PIL (Pillow)
- **Paketleme**: PyInstaller

## Kurulum ve Çalıştırma

### Bağımlılıkların Yüklenmesi
Projeyi klonladıktan sonra, gerekli Python kütüphanelerini yüklemek için terminali açıp aşağıdaki komutu çalıştırın:
```bash
pip install -r requirements.txt
```
*(Not: Eğer `requirements.txt` henüz yoksa genel olarak `torch`, `torchvision`, `PyQt5`, `opencv-python`, `Pillow`, `numpy`, `segmentation_models_pytorch` gibi paketler gereklidir.)*

### Model Ağırlıkları (.pth / .pt)
Modelin çalışabilmesi için eğitilmiş ağırlık dosyalarının (`.pth`) ilgili klasörlere (`densenet121/`, `efficientnet_b0/`, `unetpp/`) yerleştirilmiş olması gerekmektedir. (GitHub'ın 100 MB limiti nedeniyle `.pth` dosyaları bu repoda barındırılmamaktadır/dışlanmıştır.)

### Uygulamanın Başlatılması
Gerekli paketler ve model dosyaları hazırlandıktan sonra uygulamayı başlatmak için:
```bash
cd PyQt_App
python main.py
```
komutunu kullanabilirsiniz.

## Kullanım

1. **Görüntü Yükleme**: Uygulama açıldığında "Görüntü Seç" butonuna tıklayarak bilgisayarınızdaki bir fundus (retina) görüntüsünü (örn. `.jpg` veya `.png`) yükleyin.
2. **Analiz Etme**: Yüklenen görüntü üzerinde AI modelini çalıştırmak için "Analiz Et" veya ilgili butona basın.
3. **Sonuçları Görüntüleme**: Analiz işlemi tamamlandıktan sonra ekranda şunları göreceksiniz:
   - Orijinal görüntü.
   - Çıkarılan damar maskesi (UNet++ sonuçları).
   - "Hasta" veya "Sağlıklı" teşhisi ve modelin güvenilirlik (olasılık) yüzdesi.

## İletişim & Katkı
Bu proje tez çalışması kapsamında geliştirilmiştir. Hata bildirimleri veya önerileriniz için issues kısmından bildirim bırakabilirsiniz.