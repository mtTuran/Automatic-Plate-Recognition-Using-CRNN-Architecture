! dizinleri gerektiği şekilde değiştirmeyi unutmayın !

Plaka Tespit Modeli  ->	\plaka_tespit_modeli\weights\best.pt
----------------------
plate.py -> YOLO modelini eğitme kodu. Uygun formatta klosörlenmiş resimlere ve bir .yaml uzantılı config dosyasına ihtiyaç duyuyor. data.yaml buna bir örnek.

crop.py -> YOLO modeli ile verilen dizindeki dosyaları kırpıp, verilen json dosyasındaki plakaları ile eşleştirerek bir csv dosyasıyla birlikte kayıt ediyor. okuma modelini eğitmek için gerekli veriyi elde ederken kullandık. dizinleri değiştirmeyi unutmayın.

------------------------------------------------

Plaka Okuma Modeli   -> \plaka_okuma_modeli\license_model_6.pth"
----------------------
PlatesDataset.py -> okuma modelinin eğitimi için gerekli resimlerin ve etiketlerinin uygun şekilde tutulması için kullanılan sınıfı ve modelin eşsiz çıktılarını tanımlıyor.

arch.py -> model mimarisini, eğitim ve test fonksiyonlarını tanımlıyor.

clstm_ctc.py -> modelin eğitimini yapan fonksiyonları çağıran main dosyası
-------------------------------------------------

Diğer
-----------------------
calculate_acc.py -> okuma modelinin zaten kesilmiş plakalar üzerindeki performansını ölçmek için oluşturuldu. Ancak içindeki modelin kullanıldığı ve ölçüm yapıldığı diğer kodlarda da kullanılıyor.

inter.py -> plaka tespit ve okuma modellerini birlikte kullanan ve test eden arayüz uygulaması

m_api.py -> plaka tespit ve okuma modelleri ile aldığı multipart form şeklinde aldığı resim requestleri için json dosyası şeklinde plakayı döndüren api

api_test.py -> api test etmek için kullanılıyor. dizinleri değiştirmeyi unutmayın

index.html -> api test etmek için chatgpt tarafından hazırlanan basit arayüz