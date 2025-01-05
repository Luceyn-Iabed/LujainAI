# LujainAI

LujainAI

Hakkında

LujainAI, Flask, PyTorch, OpenCV ve Pandas gibi popüler kütüphane ve framework'leri kullanan Python tabanlı bir uygulamadır. Uygulama, veri işleme, model çıkarımı ve sonuç görüntüleme gibi özellikler sunar.

Özellikler

Web Arayüzü: Model ile etkileşim için Flask tabanlı bir ön yüz.

Görüntü İşleme: OpenCV ve Pillow kullanarak ileri düzey görüntü işleme.

Veri Analizi: Yapılandırılmış veri işleme için Pandas ve Numpy kullanımı.

Model Çıkarımı: Tahmin için PyTorch tabanlı yapay zeka modelleri.

Görsel Sunum: Matplotlib ve Seaborn ile zengin görsel çıktılar oluşturma.

Gereksinimler

Uygulamayı çalıştırmadan önce sisteminizde şunların yüklü olduğundan emin olun:

Python 3.8+

pip (Python paket yöneticisi)

Kurulum

Depoyu klonlayın:

git clone https://github.com/Luceyn-Iabed/LujainAI.git
cd LujainAI

Bağımlılıkları yükleyin:

pip install -r requirements.txt

Kullanım

Uygulamayı çalıştırın:

python app.py

Web tarayıcınızda http://127.0.0.1:5000 adresine giderek uygulama ile etkileşime geçebilirsiniz.

Dizin Yapısı

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

Bağımlılıklar

Bu projede kullanılan kütüphaneler:

Flask: Uygulama için web framework.

PyTorch: Yapay zeka modelleri için derin öğrenme framework'ü.

OpenCV: Görüntü işleme.

Pandas: Veri manipülasyonu ve analizi.

Matplotlib & Seaborn: Veri görüntüleme.

Diğerleri: Tam liste için requirements.txt dosyasına bakın.

Örnek Akış

Web arayüzü üzerinden bir görüntü veya veri seti yükleyin.

Sistem, OpenCV ve diğer araçları kullanarak girdiyi önişler.

Yapay zeka modeli tahmin yapmak için çalışır.

Sonuçlar, görsel olarak web arayüzünde görüntülenir.
