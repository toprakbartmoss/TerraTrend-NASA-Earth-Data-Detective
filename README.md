# NASA Earth Data Detective (Space Apps: "Be An Earth System Trend Detective")

Türkçe arayüzlü bir Streamlit uygulaması: haritadan seçilen nokta(lar) için aylık Dünya sistemi serilerini
çeker, çizer, aylık klimatolojiyi çıkararak mevsimsellikten arındırır ve trendi test eder (Mann-Kendall + Theil-Sen;
Var(S), AR(1), uzun hafıza (yerel Whittle) ve EWC düşük frekans tahminlerinin en büyüğüyle büyütülür, p ve %95 GA
aynı varyansla t dağılımından; "anlamlı" yalnızca p < 0.05 ve GA sıfırı dışlıyorsa; ayrıntı statistics_engine.py
başlığında, yanlış-pozitif oranları `scripts/mc_validate_stats.py` ile ölçülür). Çiftli/çoklu modda bölgeler arası ilişkiyi
(arındırılmış ve trendi çıkarılmış serilerde Spearman ρ, eğim farkı) gösterir ve 60 aylık tek değişkenli istatistiksel
bir uzatma çizer (iklim projeksiyonu değildir).

## Gerçek veri kaynakları ve sınırları

| Değişken | Kaynak (tokensız) | Başlangıç | Not |
|---|---|---|---|
| T2M, TQV, GWETROOT, EVLAND, FRSEAICE | NASA POWER aylık nokta API (MERRA-2 türevli; POWER koleksiyon adı vermez) | 1981-01 | GWETROOT/EVLAND yalnızca kara, FRSEAICE yalnızca okyanus hücresi |
| PRECTOTCORR | NASA POWER `PRECTOTCORR_SUM` (aylık toplam, mm/ay) | 1981-01 | POWER meta verisi "mm/day" dese de değer aylık toplamdır; gün sayısıyla çarpılmaz |
| TSKINWTR | NASA POWER `TS` (yüzey cilt sıcaklığı) | 1981-01 | Yalnızca okyanus hücresi, FRSEAICE ≤ 0.15 ve TS ≥ −2 °C olan aylar; MERRA-2'nin ayrı TSKINWTR değişkeni değildir |
| SWGDN | NASA POWER `ALLSKY_SFC_SW_DWN` (CERES SYN1deg Ed. 4.2) | 2001-01 | POWER 1984–2000'i GEWEX SRB'den verir; 2001'deki kaynak değişimi yapay bir sıçrama (ör. Sahra'da pre-2001 eğimi ~+1.8 W/m²/yıl) ve sahte "anlamlı" trendler ürettiği için bu yıllar kullanılmaz |
| ALBEDO | NASA POWER `ALLSKY_SRF_ALB` (CERES SYN1deg Ed. 4.2) | 2001-01 | Kutup gecesi ayları boştur |
| RUNOFF | POWER'da yok | — | Yalnızca Earthdata token'ı ile Giovanni üzerinden (dal test edilmedi) |

- API: `https://power.larc.nasa.gov/api/temporal/monthly/point` (community=SB). Değerler en yakın kaynak hücresinin
  değerleridir (MERRA-2 0.5°×0.625°, CERES 1°; POWER Docs FAQ). Uygulama her nokta için hücre merkezini gösterir; aynı
  hücreye düşen iki bölge özdeş seri verir ve aralarında ilişki kararı verilmez.
- En son ay(lar) POWER'ın düşük gecikmeli akışlarından gelir: meteorolojide GEOS-IT (MERRA-2'nin bittiği aydan sonrası),
  ışınımda CERES FLASHFlux 4A (son ~3–4 ay); FLASHFlux sonradan yeniden işlenmez (POWER Docs, Data Sources).
- Hiçbir eksik ay doldurulmaz veya türetilmez. Doluluk oranı yalnızca gerçek aylarla, değişkenin var olabildiği
  pencerede (max(başlangıç, değişken başlangıcı) … son yayımlanmış ay) hesaplanır; %70 altı geçersiz sayılır.
- Yüzey sınıfı veriden gelir (`data_loader.surface_from_bundle`): kara-yalnız parametrelerden (GWETROOT, EVLAND) biri
  gerçek değer veriyorsa **kara**; ikisi de hep -999 ise POWER hücre yüksekliği < 40 m ve (herhangi bir ayda deniz buzu
  > 0.15 veya "deniz buzu ≤ 0.15 ama TS < −2 °C" ay oranı ≤ %5) ise **okyanus** (açık su −2 °C'den soğuk olamaz; buz
  sahanlıkları hiç deniz buzu göstermez), değilse ortalama TS < 0 °C ise **kara buzu** (Antarktika,
  Grönland iç kesimi, buz sahanlıkları), değilse **iç su** (yüksek rakımlı büyük göl, ör. Victoria). |enlem| ≥ 66.5° ise
  kara/okyanus "kutup" olarak etiketlenir. 40 m eşiği kanıta göre seçildi: kara olmayan hücrelerde deniz buzu gösterenler
  0–25 m ve 53–158 m'de, 25–53 m arasında hiç hücre yok (eski 10 m eşiği 9.98 m'lik bir hücrede sınıfı oynatıyordu);
  eşiğe yakın hücreler ızgara köken kaydında listelenir. **Karışık kıyı/göl hücreleri:** kısmen kara içeren kıyı/iç
  deniz/göl hücreleri (ör. Baltık 58°K 20°D, Ege, Marmara, Superior, Baykal) POWER kara değişkenlerine değer verdiği için
  kara sayılır. Bu durumda POWER hücresi kara içerir ve kara değişkenlerinin (GWETROOT, EVLAND) değerleri hücrenin kara
  kısmını tanımlar, su yüzeyini değil; uygulama bunu (adres bulunamayan deniz noktası veya hücrede deniz buzu görülmesi
  durumunda) açıkça yazar. Hazar okyanus/deniz sınıfındadır.
- Servis hataları "bu konumda veri yok" olarak değil, hata olarak gösterilir ve önbelleğe alınmaz.
- Yer adları OpenStreetMap Nominatim'den gelir (© OpenStreetMap contributors). Kullanım politikası gereği en fazla
  1 istek/sn yapılır ve yalnızca başarılı yanıtlar 24 saat önbelleğe alınır. İletişim bilgisini User-Agent'a eklemek
  için `EDD_CONTACT` ortam değişkenini (e-posta veya proje URL'si) ayarlayın.
- Earthdata token (isteğe bağlı, kenar çubuğu veya `NASA_EARTHDATA_TOKEN`): her değişken önce Giovanni'den istenir;
  Giovanni seriyi tamamen sağlarsa o, aksi halde POWER kullanılır (aylar karıştırılmaz). **Bu dal token olmadan
  geliştirildi ve test edilmedi.**
- Önem puanları, önerilen kombinasyonlar ve bütünleşik indeksin işaret kuralı ekibin kendi seçimleridir. Bütünleşik
  indeks yalnızca seçili değişkenlerin hepsinin gözlendiği ortak aylar, en az birinin gözlendiği ayların en az %70'i ise
  kurulur; aksi halde (ör. yalnızca buzsuz aylarda var olan TSKINWTR ile deniz buzu) gerekçesiyle gösterilmez ve
  değişken bazlı sonuçlar yine gösterilir. Kutup/Enerji vakası bu yüzden T2M + FRSEAICE + ALBEDO + SWGDN ile 2001'de başlar.
- 60 aylık uzatma ve %95 bandı değişkenin fiziksel sınırlarında kırpılır (oranlar ve albedo 0–1, yağış ve ışınım ≥ 0,
  açık su sıcaklığı ≥ −2 °C; buharlaşma negatif olabilir); kırpıldığında not düşülür.

## Dünya haritaları

`data/global_grid_power_5deg.csv` NASA POWER aylık nokta API'sinin 5° düzenli ızgaranın 2592 hücre merkezinde
alınmış **nokta örnekleridir** (alan ortalaması değildir). Her değişken için iki sabit dönem saklanır: ilk 5 yıl
(SWGDN ve ALBEDO 2001–2005, diğerleri 1981–1985) ve son 5 tam yıl. Bir hücre yalnızca dönemin 60 ayının
tamamı geçerliyse doludur. Köken kaydı (kullanılan dönemler, yüzey kuralı eşiğine yakın hücreler):
`data/global_grid_power_5deg.json`. Bütünleşik indeks ve RUNOFF için harita yoktur. Değişim panelinin renk ölçeği
|değişim|'in 98. yüzdeliğinde kesilir ve renk çubuğu gerçek min/max'ı yazar. Tek bir 5° hücrenin değişimi (özellikle
yağış) yerel olarak gürültülüdür: tek hücreleri değil geniş örüntüleri okuyun.

## Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dil / Language

Arayüzün varsayılan dili İngilizcedir; Türkçe, üst menünün sağındaki 🌐 seçiciden seçilir. Varsayılan dili ortam değişkeniyle değiştirmek için: `EDD_LANG=tr streamlit run app.py`. Metinler `i18n.py` içindeki `L("English", "Türkçe")` çağrılarıyla tutulur.

The UI defaults to English; Turkish can be selected from the 🌐 selector at the right of the top menu (or set `EDD_LANG=tr`). Tests run in Turkish mode (`tests/conftest.py`); `tests/test_i18n.py` covers English.

## Izgara dosyasını yeniden üretme

```bash
python scripts/build_global_grid.py   # ~10–15 dk, 5 eşzamanlı istek; yarıda kalırsa aynı yapılandırmayla kaldığı yerden
                                      # devam eder, başarıyla bitince ara dosyayı siler
```

## Testler

```bash
pip install -r requirements-dev.txt   # pytest, pymannkendall (yalnızca testler)
python -m pytest -q tests/            # ağ erişimi gerekir (POWER, Nominatim)
python scripts/mc_validate_stats.py 1000 --real   # Monte Carlo yanlış-pozitif/güç tablosu (--real: gerçek seri vekilleri)
```
