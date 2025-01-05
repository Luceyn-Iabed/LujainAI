 # LujainAI: Göğüs Röntgeni Görüntü Analizi Uygulaması

LujainAI, Flask tabanlı bir web uygulamasıdır. Bu uygulama, kullanıcıların göğüs röntgeni görüntülerini yükleyerek yapay zeka modeli ile gölgeleri veya kitleleri tespit etmelerini sağlar. Deep Learning teknikleri kullanarak, özellikle Convolutional Neural Networks (CNNs), röntgen görüntülerindeki anormallikleri analiz eder. Kullanıcı dostu arayüzü ve hızlı sonuç görüntüleme özelliği ile, uygulama sağlık sektöründe önemli bir destek aracı olabilir.

## Özellikler:
- Web arayüzü üzerinden röntgen görüntüsü yükleme.
- Yapay zeka ile gölgeleri ve kitleleri tespit etme.
- Sonuçların hızlı ve dinamik görüntülenmesi.

## Gereksinimler

Uygulamayı çalıştırmadan önce sisteminizde şunların yüklü olduğundan emin olun:

- Python 3.8+
- pip (Python paket yöneticisi)

## Kurulum

1. Depoyu klonlayın:

```bash
git clone https://github.com/Luceyn-Iabed/LujainAI.git
cd LujainAI
```
2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

Uygulamayı çalıştırmak için aşağıdaki komutu kullanın:

```bash
 python app.py
```
Web tarayıcınızda http://127.0.0.1:5000  adresine giderek uygulama ile etkileşime geçebilirsiniz.


## Dizin Yapısı

```bash
 LujainAI/
|
|-- TRAINED_MODEL_FOLDER/    # Önceden eğitilmiş yapay zeka modelleri
|-- static/                  # Statik dosyalar (CSS, JavaScript, görüntüler)
|-- templates/               # Flask için HTML şablonları
|-- örnek_Lujain_images/     # Test için örnek görüntüler
|-- app.py                   # Ana Flask uygulaması
|-- df_fin_preds.csv         # Örnek tahmin sonuçları
|-- requirements.txt         # Proje bağımlılıkları
|-- utils.py                 # Yardımcı fonksiyonlar

```
## Bağımlılıklar

Bu projede kullanılan kütüphaneler:

- Flask: Uygulama için web framework'ü.
- PyTorch: Yapay zeka modelleri için derin öğrenme framework'ü.
- OpenCV: Görüntü işleme.
- Pandas: Veri manipülasyonu ve analizi.
- Matplotlib & Seaborn: Veri görselleştirme.
- 
Tam bağımlılık listesi için requirements.txt dosyasına bakabilirsiniz.

## Örnek Akış

1. Web arayüzü üzerinden bir görüntü veya veri seti yükleyin.
2. Sistem, OpenCV ve diğer araçları kullanarak girdiyi önişler.
3. Yapay zeka modeli tahmin yapmak için çalışır.
4. Sonuçlar, görsel olarak web arayüzünde görüntülenir.


                 ## not
                           -  share/man/man1
                                  - YANLIŞIKLA EKLADIM
