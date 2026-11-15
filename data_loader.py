"""
NASA Earth Data Detective (EDD) - Veri Modülü (data_loader.py)

Gerçek veri kaynağı (tokensız): NASA POWER aylık nokta API'si
(https://power.larc.nasa.gov/api/temporal/monthly/point, community=SB). POWER verisi 1981-01'de başlar.
- Meteorolojik parametreler (T2M, PRECTOTCORR, TS, GWETROOT, EVLAND, TQV, FRSEAICE): POWER'ın NASA GMAO MERRA-2
  yeniden analizinden türettiği değerler; en son ay(lar) POWER'ın düşük gecikmeli GEOS-IT akışından gelir
  (POWER Docs, Data Sources: Meteorological). POWER hangi MERRA-2 koleksiyonunu kullandığını parametre meta
  verisinde belirtmez; bu yüzden koleksiyon adı verilmez.
- Işınım parametreleri (ALLSKY_SFC_SW_DWN -> SWGDN, ALLSKY_SRF_ALB -> ALBEDO): POWER'da 1984-2000 GEWEX SRB 4.0-IP,
  2001'den ~3-4 ay öncesine kadar CERES SYN1deg Ed. 4.2, en son aylar CERES FLASHFlux 4A (POWER Docs, Data Sources:
  Energy Fluxes). Uygulama ikisini de yalnızca 2001-01'den itibaren kullanır: 2001'deki SRB -> CERES kaynak
  değişimi seride yapay bir kırılma oluşturduğundan SWGDN'nin 1984-2000 değerleri atılır.
- RUNOFF POWER'da yoktur: yalnızca Earthdata token'ı ile Giovanni üzerinden (test edilmemiş dal) alınabilir.
Hiçbir eksik ay doldurulmaz/türetilmez; servis hataları "veri yok" olarak değil, hata olarak bildirilir.
"""

import datetime as _dt
import os
import threading
import time
import urllib.parse
from typing import Any, Dict, Mapping, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st

from i18n import L, LocalizedMapping, get_lang, pick

# Uç noktalar
POWER_MONTHLY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
GIOVANNI_CLOUD_API_URL = "https://api.giovanni.earthdata.nasa.gov/timeseries"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

POWER_FIRST_YEAR = 1981  # POWER aylık API: start=1980 -> HTTP 422 "data starts at 1981/01/01"
SEA_ICE_MASK_THRESHOLD = 0.15  # TS bu deniz buzu oranının üstündeki aylarda buz yüzeyi sıcaklığıdır -> NaN
OPEN_WATER_MIN_TS = -2.0  # Deniz suyu ~-1.8 °C'nin altına inemez: daha soğuk TS açık su sıcaklığı değildir
# POWER hücre yüksekliği (geometry z) eşiği. Kanıt (5° ızgaranın kara olmayan hücreleri, 2026-11): deniz buzu gösteren
# hücreler 0–25 m ve 53–158 m'de, deniz buzu göstermeyen buz sahanlığı/örtüsü hücreleri ise "buzsuz ama TS < -2 °C"
# oranı 1.0 ile; 25.0–53.1 m arasında hiç hücre yok. Eşik bu boşluğun ortasında (eski 10 m eşiği 9.98 m'lik bir
# hücrede yeniden üretimde sınıf değiştiriyordu). Eşiğe ±10 m yakın hücreler ızgara köken kaydında listelenir.
OCEAN_MAX_ELEVATION_M = 40.0
OCEAN_MAX_COLD_OPEN_FRACTION = 0.05  # "buzsuz ama TS < -2 °C" ay oranı bunu aşarsa hücre su değil (buz sahanlığı)
POLAR_LAT = 66.5  # Kutup daireleri (kutup sınıfı için tek enlem kuralı)
# POWER nokta API'si en yakın kaynak hücresinin değerini verir (POWER Docs, FAQ Solar): MERRA-2 0.5° x 0.625°
# (merkezler enlemde 0.5°, boylamda -180 + 0.625° katları), CERES/SRB ışınımı 1° x 1° (merkezler x.5°).
MERRA2_DLAT, MERRA2_DLON = 0.5, 0.625
RADIATION_VARS = ("ALBEDO", "SWGDN")
OSM_ATTRIBUTION = "© OpenStreetMap contributors"
_POWER_PARAMETERS = "T2M,PRECTOTCORR_SUM,TS,GWETROOT,EVLAND,TQV,FRSEAICE,ALLSKY_SRF_ALB,ALLSKY_SFC_SW_DWN"
_LAND_ONLY_POWER_PARAMS = ("GWETROOT", "EVLAND")  # Okyanus ve buz örtüsü hücrelerinde POWER -999 döndürür
MERRA2_COLLECTION_TR = "MERRA-2 (koleksiyon POWER tarafından belirtilmiyor)"
MERRA2_COLLECTION_EN = "MERRA-2 (collection not specified by POWER)"
MERRA2_COLLECTION = {"en": MERRA2_COLLECTION_EN, "tr": MERRA2_COLLECTION_TR}
SURFACE_NAMES_TR = LocalizedMapping({"ocean": {"en": "ocean/sea", "tr": "okyanus/deniz"}, "land": {"en": "land", "tr": "kara"}, "land_ice": {"en": "land ice (ice sheet / ice shelf)", "tr": "kara buzu (buz örtüsü / buz sahanlığı)"}, "inland_water": {"en": "inland water (lake)", "tr": "iç su (göl)"}})


def latest_year() -> int:
    """İçinde bulunulan UTC yılı (her çağrıda yeniden hesaplanır; uzun süre açık sunucuda yılbaşında eskimez)."""
    return _dt.datetime.now(_dt.timezone.utc).year


class DataServiceError(RuntimeError):
    """Uzak servise ulaşılamadı / HTTP hatası / beklenmeyen yanıt ("bu konumda veri yok" ile karıştırılmaz)."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# 10 DEĞİŞKEN (9'u tokensız NASA POWER'dan; RUNOFF yalnızca token + Giovanni ile)
# Önem puanları (/10) ve "polarity" işaretleri ekibin kendi seçimidir; harici bir kaynağa dayanmaz.
# polarity: Bütünleşik indeksin sabit yön kuralı -> indeks yükselir = daha sıcak, daha fazla güneş ışınımı,
# daha az su (yağış, toprak nemi, buharlaşma, su buharı, akış), daha az deniz buzu ve daha düşük albedo.
# surface: "any" | "land" | "ocean" -> NASA POWER'ın hücre maskesine göre (bkz. power_surface_class).
# ---------------------------------------------------------------------------
OFFICIAL_VARIABLES: Mapping[str, Dict[str, Any]] = LocalizedMapping({
    "T2M": {
        "rank": 1,
        "importance_score_10": 10.0,
        "base_weight": 1.00,  # 10.0 / 10
        "short_name": "T2M",
        "collection": MERRA2_COLLECTION,
        "satellite_dataset": "M2TMNXSLV_5_12_4_T2M",
        "satellite_start_year": 1981,
        "label": {"en": "1. 🌡️ 2 m Air Temperature (T2M — NASA POWER / MERRA-2)", "tr": "1. 🌡️ 2 m Hava Sıcaklığı (T2M — NASA POWER / MERRA-2)"},
        "short_label": {"en": "2 m Air Temperature (T2M)", "tr": "2 m Hava Sıcaklığı (T2M)"},
        "unit": "°C",
        "agg_method": "mean",
        "category": {"en": "Atmosphere & Surface Thermodynamics", "tr": "Atmosfer & Yüzey Termodinamiği"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — MERRA-2 derived", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — MERRA-2 türevli"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA GMAO MERRA-2 reanalysis (via NASA POWER)", "tr": "NASA GMAO MERRA-2 yeniden analizi (NASA POWER üzerinden)"},
        "surface": "any",
        "power_parameter": "T2M",
        "collection_url": None,
        "bounds": None,  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": 1.0,
        "student_explanation": {"en": (
            "The monthly mean air temperature in the part of the atmosphere closest to the surface (2 metres above "
            "the ground). It is a basic climate indicator because it is directly related to the environment people "
            "live in and is well suited for tracking changes such as heat waves, frost events and seasonal warming. "
            "It helps compare regional and global warming/cooling trends. A single monthly value represents weather, "
            "while a long-term series represents climate behaviour; calculating a slope without removing seasonality "
            "can be misleading."
        ), "tr": (
            "Atmosferin yüzeye yakın bölümündeki (yerden 2 metre) hava sıcaklığının aylık ortalamasıdır. "
            "İnsanların yaşadığı ortamla doğrudan ilişkili olduğu ve ısı dalgaları, don olayları, mevsimsel ısınma "
            "gibi değişimleri izlemeye elverişli olduğu için temel iklim göstergesidir. Bölgesel ve küresel "
            "ısınma/soğuma eğilimlerini karşılaştırmaya yardım eder. Tek bir aylık değer hava durumunu, uzun "
            "dönemli seri ise iklim davranışını temsil eder; mevsimsellik giderilmeden eğim hesaplamak yanıltıcı olabilir."
        )},
        "correlated_with": ["PRECTOTCORR (-)", "GWETROOT (-)", "EVLAND (+)", "SWGDN (+)"],
    },
    "PRECTOTCORR": {
        "rank": 2,
        "importance_score_10": 9.8,
        "base_weight": 0.98,  # 9.8 / 10
        "short_name": "PRECTOTCORR",
        "collection": MERRA2_COLLECTION,
        "satellite_dataset": "M2TMNXFLX_5_12_4_PRECTOTCORR",
        "satellite_start_year": 1981,
        "label": {"en": "2. 🌧️ Precipitation (PRECTOTCORR — NASA POWER / MERRA-2)", "tr": "2. 🌧️ Yağış (PRECTOTCORR — NASA POWER / MERRA-2)"},
        "short_label": {"en": "Precipitation (PRECTOTCORR)", "tr": "Yağış (PRECTOTCORR)"},
        "unit": {"en": "mm/month", "tr": "mm/ay"},
        "agg_method": "sum",
        "category": {"en": "Water Cycle & Hydrological Flux", "tr": "Su Döngüsü & Hidrolojik Akı"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — MERRA-2 derived", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — MERRA-2 türevli"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA GMAO MERRA-2 bias-corrected precipitation (NASA POWER PRECTOTCORR_SUM, monthly total)", "tr": "NASA GMAO MERRA-2 düzeltilmiş yağış (NASA POWER PRECTOTCORR_SUM, aylık toplam)"},
        "surface": "any",
        "power_parameter": "PRECTOTCORR_SUM",
        "collection_url": None,
        "bounds": (0.0, None),  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": -1.0,
        "student_explanation": {"en": (
            "Precipitation is the total amount of water that falls on a given area over a given period. NASA POWER "
            "provides this variable directly as a monthly total (mm/month) under the name PRECTOTCORR_SUM; the app "
            "uses this value as is. It is a central variable for studying drought, extreme precipitation, the "
            "replenishment of water resources and hydrological change. The regional distribution of precipitation is "
            "highly variable; a global average can hide local trends. This product is not a direct rain-gauge "
            "observation but a reanalysis estimate constrained by observations."
        ), "tr": (
            "Yağış, belirli bir alana belirli sürede düşen toplam su miktarını ifade eder. NASA POWER bu değişkeni "
            "PRECTOTCORR_SUM olarak doğrudan aylık toplam (mm/ay) verir; uygulama bu değeri olduğu gibi kullanır. Kuraklık, "
            "aşırı yağış, su kaynaklarının yenilenmesi ve hidrolojik değişimlerin incelenmesinde merkezi bir değişkendir. "
            "Yağışın bölgesel dağılımı çok değişkendir; küresel ortalama yerel eğilimleri gizleyebilir. Bu ürün doğrudan "
            "yağmur ölçer gözlemi değil, gözlemlerle kısıtlanmış yeniden analiz tahminidir."
        )},
        "correlated_with": ["GWETROOT (+)", "RUNOFF (+)", "TQV (+)", "T2M (-)"],
    },
    "TSKINWTR": {
        "rank": 3,
        "importance_score_10": 9.6,
        "base_weight": 0.96,  # 9.6 / 10
        "short_name": "TSKINWTR",
        "collection": {"en": MERRA2_COLLECTION_EN + " — POWER TS", "tr": MERRA2_COLLECTION_TR + " — POWER TS"},
        "satellite_dataset": "M2TMNXOCN_5_12_4_TSKINWTR",
        "satellite_start_year": 1981,
        "label": {"en": "3. 🌊 Ocean Surface Skin Temperature (POWER TS, ice-free ocean cells only)", "tr": "3. 🌊 Okyanus Yüzey Cilt Sıcaklığı (POWER TS, yalnızca buzsuz okyanus hücresi)"},
        "short_label": {"en": "Ocean Surface Skin Temperature (TSKINWTR = POWER TS)", "tr": "Okyanus Yüzey Cilt Sıcaklığı (TSKINWTR = POWER TS)"},
        "unit": "°C",
        "agg_method": "mean",
        "category": {"en": "Ocean & Marine Energy Exchange", "tr": "Okyanus & Denizel Enerji Alışverişi"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — MERRA-2 derived (TS)", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — MERRA-2 türevli (TS)"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA GMAO MERRA-2 surface skin temperature TS (via NASA POWER)", "tr": "NASA GMAO MERRA-2 yüzey cilt sıcaklığı TS (NASA POWER üzerinden)"},
        "surface": "ocean",
        "power_parameter": "TS",
        "collection_url": None,
        "bounds": (OPEN_WATER_MIN_TS, None),  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": 1.0,
        "student_explanation": {"en": (
            "NASA POWER's TS parameter (MERRA-2-derived surface skin temperature) is the temperature of whatever "
            "surface is in the cell. For this reason the app shows it only in ocean cells (where POWER returns -999 "
            "for land variables) and in months when the sea-ice fraction is below 15%; under these conditions it "
            "corresponds to the ocean surface skin temperature. It is not MERRA-2's separate TSKINWTR (open water) "
            "variable, and it does not represent ocean temperature at all depths or an independent satellite SST "
            "product. It is not shown in land cells."
        ), "tr": (
            "NASA POWER'ın TS parametresi (MERRA-2 türevli yüzey cilt sıcaklığı) hücrede hangi yüzey varsa onun "
            "sıcaklığıdır. Bu nedenle uygulama onu yalnızca okyanus hücrelerinde (POWER kara değişkenlerine -999 "
            "döndürüyorsa) ve deniz buzu oranı %15'in altındaki aylarda gösterir; bu koşullarda okyanus yüzey cilt "
            "sıcaklığına karşılık gelir. MERRA-2'nin ayrı TSKINWTR (açık su) değişkeni değildir, tüm derinliklerdeki "
            "okyanus sıcaklığını veya bağımsız bir uydu SST ürününü temsil etmez. Kara hücrelerinde gösterilmez."
        )},
        "correlated_with": ["T2M (+)", "FRSEAICE (-)", "ALBEDO (-)", "TQV (+)"],
    },
    "GWETROOT": {
        "rank": 4,
        "importance_score_10": 9.5,
        "base_weight": 0.95,  # 9.5 / 10
        "short_name": "GWETROOT",
        "collection": MERRA2_COLLECTION,
        "satellite_dataset": "M2TMNXLND_5_12_4_GWETROOT",
        "satellite_start_year": 1981,
        "label": {"en": "4. 🌱 Root Zone Soil Moisture (GWETROOT — NASA POWER / MERRA-2)", "tr": "4. 🌱 Kök Bölgesi Toprak Nemi (GWETROOT — NASA POWER / MERRA-2)"},
        "short_label": {"en": "Root Zone Soil Moisture (GWETROOT)", "tr": "Kök Bölgesi Toprak Nemi (GWETROOT)"},
        "unit": {"en": "Wetness Fraction (0–1)", "tr": "Islaklık Oranı (0–1)"},
        "agg_method": "mean",
        "category": {"en": "Land Hydrology & Soil Water Balance", "tr": "Kara Hidrolojisi & Toprak Su Dengesi"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — MERRA-2 derived", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — MERRA-2 türevli"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA GMAO MERRA-2 Land (via NASA POWER)", "tr": "NASA GMAO MERRA-2 Land (NASA POWER üzerinden)"},
        "surface": "land",
        "power_parameter": "GWETROOT",
        "collection_url": None,
        "bounds": (0.0, 1.0),  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": -1.0,
        "student_explanation": {"en": (
            "Expresses the water status of the soil layers that plant roots can reach, as the wetness fraction (0–1) "
            "that the model represents over the root zone. It helps track how drought develops not only as a lack of "
            "precipitation but as a decrease in usable water in the soil; studied together with precipitation, "
            "evaporation and plant water stress, it shows how these processes are connected. This dimensionless "
            "wetness indicator is not a direct volumetric moisture percentage and is not defined in ocean cells."
        ), "tr": (
            "Bitki köklerinin erişebildiği toprak katmanlarındaki su durumunu, modelin kök bölgesi boyunca temsil ettiği "
            "ıslaklık oranı (0–1) olarak ifade eder. Kuraklığın yalnızca yağış eksikliği değil, toprakta kullanılabilir "
            "su azalması biçiminde nasıl geliştiğini izlemeye yarar; yağış, buharlaşma ve bitki su stresiyle birlikte "
            "incelendiğinde süreçlerin bağlantısını gösterir. Bu boyutsuz ıslaklık göstergesi doğrudan hacimsel nem "
            "yüzdesi değildir ve okyanus hücrelerinde tanımlı değildir."
        )},
        "correlated_with": ["PRECTOTCORR (+)", "EVLAND (+)", "RUNOFF (+)", "T2M (-)"],
    },
    "EVLAND": {
        "rank": 5,
        "importance_score_10": 9.3,
        "base_weight": 0.93,  # 9.3 / 10
        "short_name": "EVLAND",
        "collection": MERRA2_COLLECTION,
        "satellite_dataset": "M2TMNXLND_5_12_4_EVLAND",
        "satellite_start_year": 1981,
        "label": {"en": "5. 💨 Land Surface Evaporation (EVLAND — NASA POWER / MERRA-2)", "tr": "5. 💨 Kara Yüzeyi Buharlaşması (EVLAND — NASA POWER / MERRA-2)"},
        "short_label": {"en": "Land Surface Evaporation (EVLAND)", "tr": "Kara Yüzeyi Buharlaşması (EVLAND)"},
        "unit": {"en": "mm/month", "tr": "mm/ay"},
        "agg_method": "sum",
        "category": {"en": "Land Hydrology & Evapotranspiration", "tr": "Kara Hidrolojisi & Evapotranspirasyon"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — MERRA-2 derived", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — MERRA-2 türevli"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA GMAO MERRA-2 Land (via NASA POWER)", "tr": "NASA GMAO MERRA-2 Land (NASA POWER üzerinden)"},
        "surface": "land",
        "power_parameter": "EVLAND",
        "collection_url": None,
        "bounds": None,  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": -1.0,
        "student_explanation": {"en": (
            "Represents the flux of water vapour from the land surface to the atmosphere; it reflects the combined "
            "effect of evaporation from soil and vegetation and plant transpiration. It is one of the important links"
            " between the atmosphere and the land in the water cycle. It provides information on how much of the "
            "precipitation stays at the surface, how much returns to the atmosphere, and how drought/heat conditions "
            "affect water loss. It is a flux variable; the monthly total (mm/month) should not be confused with the "
            "instantaneous rate."
        ), "tr": (
            "Kara yüzeyinden atmosfere geçen su buharı akısını temsil eder; toprak ve bitki örtüsünden buharlaşma ile "
            "bitki terlemesinin toplam etkisini yansıtır. Su döngüsünün atmosfer ve kara arasındaki önemli bağlantılarındandır. "
            "Yağışın ne kadarının yüzeyde kaldığı, atmosfere döndüğü ve kuraklık/ısı koşullarının su kaybını nasıl "
            "etkilediği hakkında bilgi sağlar. Bir akı değişkenidir; aylık toplam (mm/ay) ile anlık hız birbirine "
            "karıştırılmamalıdır."
        )},
        "correlated_with": ["GWETROOT (+)", "PRECTOTCORR (+)", "T2M (-)", "SWGDN (+)"],
    },
    "RUNOFF": {
        "rank": 6,
        "importance_score_10": 9.1,
        "base_weight": 0.91,  # 9.1 / 10
        "short_name": "RUNOFF",
        "collection": {"en": "M2TMNXLND v5.12.4 (Giovanni variable ID; not verified)", "tr": "M2TMNXLND v5.12.4 (Giovanni değişken kimliği; doğrulanmadı)"},
        "satellite_dataset": "M2TMNXLND_5_12_4_RUNOFF",
        "satellite_start_year": 1981,
        "label": {"en": "6. 🏞️ Surface Runoff (RUNOFF — Earthdata token + Giovanni only)", "tr": "6. 🏞️ Yüzey Akışı (RUNOFF — yalnızca Earthdata token + Giovanni)"},
        "short_label": {"en": "Surface Runoff (RUNOFF)", "tr": "Yüzey Akışı / Runoff (RUNOFF)"},
        "unit": {"en": "mm/month", "tr": "mm/ay"},
        "agg_method": "sum",
        "category": {"en": "Land Hydrology & Flood/Drought", "tr": "Kara Hidrolojisi & Taşkın/Kuraklık"},
        "platform": {"en": "NASA Giovanni (Earthdata token required; not available in NASA POWER)", "tr": "NASA Giovanni (Earthdata token gerekli; NASA POWER'da yok)"},
        "platform_url": "https://giovanni.gsfc.nasa.gov/giovanni/",
        "sensors": {"en": "NASA GMAO MERRA-2 Land RUNOFF (Giovanni only; not tested)", "tr": "NASA GMAO MERRA-2 Land RUNOFF (yalnızca Giovanni; test edilmedi)"},
        "surface": "land",
        "power_parameter": None,
        "collection_url": "https://disc.gsfc.nasa.gov/datasets/M2TMNXLND_5.12.4/summary",
        "bounds": (0.0, None),  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": -1.0,
        "student_explanation": {"en": (
            "Not available in NASA POWER; only when an Earthdata token is entered does the app try to retrieve it "
            "through Giovanni as MERRA-2 M2TMNXLND RUNOFF (this branch has not been tested without a token). Without "
            "a token, no data is shown. It is the total water flux of the surface and subsurface runoff components "
            "produced by the model over the land surface. It helps assess the potential for water to drain away from "
            "the land after precipitation and snowmelt, as well as the water balance and flood/drought processes. "
            "This variable is not the discharge measured at river gauging stations: MERRA-2 does not give the volume "
            "passing a particular river mouth through basin routing. In regional analysis it should be considered "
            "together with precipitation and soil moisture, and flux units should be converted to monthly totals "
            "(mm/month)."
        ), "tr": (
            "NASA POWER'da bulunmaz; yalnızca Earthdata token'ı girildiğinde Giovanni üzerinden MERRA-2 M2TMNXLND "
            "RUNOFF olarak alınmaya çalışılır (bu dal token olmadan test edilmemiştir). Token yoksa veri gösterilmez. "
            "Modelin kara yüzeyinde ürettiği yüzey ve yeraltı kaynaklı akış bileşenlerinin toplam su akısıdır. "
            "Yağış ve kar erimesinden sonra suyun araziden uzaklaşma potansiyelini, su dengesi ve taşkın/kuraklık "
            "süreçlerini değerlendirmeye yardımcı olur. Bu değişken nehir istasyonlarında ölçülen debi değildir: "
            "MERRA-2'de havza yönlendirmesiyle belirli bir nehir ağzından geçen hacmi vermez. Bölgesel analizde "
            "yağış ve toprak nemiyle birlikte ele alınmalı, akı birimleri aylık toplam (mm/ay) olarak dönüştürülmelidir."
        )},
        "correlated_with": ["PRECTOTCORR (+)", "GWETROOT (+)", "TQV (+)"],
    },
    "TQV": {
        "rank": 7,
        "importance_score_10": 8.9,
        "base_weight": 0.89,  # 8.9 / 10
        "short_name": "TQV",
        "collection": MERRA2_COLLECTION,
        "satellite_dataset": "M2TMNXSLV_5_12_4_TQV",
        "satellite_start_year": 1981,
        "label": {"en": "7. ☁️ Total Precipitable Water Vapour (TQV — NASA POWER / MERRA-2)", "tr": "7. ☁️ Toplam Çökelebilir Su Buharı (TQV — NASA POWER / MERRA-2)"},
        "short_label": {"en": "Precipitable Water Vapour (TQV)", "tr": "Çökelebilir Su Buharı (TQV)"},
        "unit": "kg/m² (mm)",
        "agg_method": "mean",
        "category": {"en": "Atmospheric Moisture & Transport", "tr": "Atmosferik Nem & Taşınım"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — MERRA-2 derived", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — MERRA-2 türevli"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA GMAO MERRA-2 (via NASA POWER)", "tr": "NASA GMAO MERRA-2 (NASA POWER üzerinden)"},
        "surface": "any",
        "power_parameter": "TQV",
        "collection_url": None,
        "bounds": (0.0, None),  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": -1.0,
        "student_explanation": {"en": (
            "The total amount of water vapour in the entire atmospheric column, expressed as an equivalent depth of "
            "liquid water at the surface (kg/m²; numerically equal to mm of water). It helps track the moisture "
            "content of the atmosphere and changes related to moisture transport and heavy-precipitation potential. "
            "This is not the relative humidity at the surface or the humidity at 2 m only; it represents the total "
            "vapour in the vertical atmospheric column. Although it is related to precipitation, it is not the same "
            "variable as the amount of precipitation."
        ), "tr": (
            "Atmosfer sütununun tamamı boyunca bulunan su buharı miktarının, yüzeyde sıvı su derinliği cinsinden "
            "ifade edilen toplamıdır (kg/m²; sayısal olarak mm suya karşılık gelir). Atmosferdeki nem içeriğini, "
            "nem taşınımı ve yoğun yağış potansiyeliyle ilişkili değişimleri izlemeye yarar. Bu, yüzeydeki bağıl nem "
            "veya yalnızca 2 m'deki nem değildir; dikey atmosfer sütunundaki toplam buharı temsil eder. Yağışla "
            "ilişkili olsa da yağış miktarıyla aynı değişken değildir."
        )},
        "correlated_with": ["PRECTOTCORR (+)", "T2M (+)", "TSKINWTR (+)", "RUNOFF (+)"],
    },
    "FRSEAICE": {
        "rank": 8,
        "importance_score_10": 8.7,
        "base_weight": 0.87,  # 8.7 / 10
        "short_name": "FRSEAICE",
        "collection": MERRA2_COLLECTION,
        "satellite_dataset": "M2TMNXOCN_5_12_4_FRSEAICE",
        "satellite_start_year": 1981,
        "label": {"en": "8. 🧊 Sea-Ice Covered Area Fraction (FRSEAICE — NASA POWER / MERRA-2)", "tr": "8. 🧊 Deniz Buzu Kaplı Alan Oranı (FRSEAICE — NASA POWER / MERRA-2)"},
        "short_label": {"en": "Sea-Ice Fraction (FRSEAICE)", "tr": "Deniz Buzu Oranı (FRSEAICE)"},
        "unit": {"en": "Fraction (0–1)", "tr": "Oran (0–1)"},
        "agg_method": "mean",
        "category": {"en": "Cryosphere & Polar Sea Ice", "tr": "Kriyosfer & Kutup Deniz Buzu"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — MERRA-2 derived", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — MERRA-2 türevli"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA GMAO MERRA-2 (via NASA POWER)", "tr": "NASA GMAO MERRA-2 (NASA POWER üzerinden)"},
        "surface": "ocean",
        "power_parameter": "FRSEAICE",
        "collection_url": None,
        "bounds": (0.0, 1.0),  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": -1.0,
        "student_explanation": {"en": (
            "Represents the fraction (0–1) of an ocean grid cell covered by sea ice. In polar regions, the seasonal "
            "cycle and long-term change in sea-ice area are among the important indicators of the climate system; sea"
            " ice also affects albedo and ocean–atmosphere heat exchange. Interpretation should be made only in "
            "suitable polar sea areas. This is a MERRA-2 reanalysis field; it is not the same data as an independent "
            "satellite sea-ice concentration product, and an observational cross-check is recommended."
        ), "tr": (
            "Bir okyanus ızgara hücresinde deniz buzunun kapladığı alan oranını (0–1) temsil eder. Kutup bölgelerinde "
            "deniz buzunun mevsimsel döngüsü ve uzun dönemli alan değişimi, iklim sisteminin önemli göstergelerindendir; "
            "albedo ve okyanus-atmosfer ısı alışverişini de etkiler. Yorum yalnızca uygun kutup deniz alanlarında "
            "yapılmalıdır. Bu MERRA-2 yeniden analiz alanıdır; bağımsız uydu deniz buzu konsantrasyon ürünüyle aynı "
            "veri değildir ve gözlemsel çapraz kontrol önerilir."
        )},
        "correlated_with": ["ALBEDO (+)", "TSKINWTR (-)", "T2M (-)", "SWGDN (-)"],
    },
    "ALBEDO": {
        "rank": 9,
        "importance_score_10": 8.5,
        "base_weight": 0.85,  # 8.5 / 10
        "short_name": "ALBEDO",
        "collection": {"en": "CERES SYN1deg Ed. 4.2; most recent ~3–4 months CERES FLASHFlux 4A (NASA POWER ALLSKY_SRF_ALB)", "tr": "CERES SYN1deg Ed. 4.2; en son ~3–4 ay CERES FLASHFlux 4A (NASA POWER ALLSKY_SRF_ALB)"},
        "satellite_dataset": "M2TMNXRAD_5_12_4_ALBEDO",
        "satellite_start_year": 2001,
        "label": {"en": "9. ☀️ All-Sky Surface Albedo (ALBEDO — NASA POWER / CERES, 2001+)", "tr": "9. ☀️ Tüm Gökyüzü Yüzey Albedosu (ALBEDO — NASA POWER / CERES, 2001+)"},
        "short_label": {"en": "Surface Albedo (ALBEDO)", "tr": "Yüzey Albedosu (ALBEDO)"},
        "unit": {"en": "Fraction (0–1)", "tr": "Oran (0–1)"},
        "agg_method": "mean",
        "category": {"en": "Surface Energy Budget & Radiation", "tr": "Yüzey Enerji Bütçesi & Radyasyon"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — CERES SYN1deg / FLASHFlux (2001+)", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — CERES SYN1deg / FLASHFlux (2001+)"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA CERES SYN1deg satellite-derived product (via NASA POWER)", "tr": "NASA CERES SYN1deg uydu türevli ürün (NASA POWER üzerinden)"},
        "surface": "any",
        "power_parameter": "ALLSKY_SRF_ALB",
        "collection_url": None,
        "bounds": (0.0, 1.0),  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": -1.0,
        "student_explanation": {"en": (
            "A ratio-type quantity (0–1) showing how much of the incoming shortwave solar radiation the Earth's "
            "surface reflects back. It varies with changes in snow/ice cover, vegetation, soil and surface moisture; "
            "it helps us understand the surface's energy gain and climate feedbacks. When albedo rises, the surface "
            "generally reflects a larger share of solar energy; however, the effects of cloudiness, sun angle and "
            "surface type must be kept in mind. Source: NASA POWER ALLSKY_SRF_ALB (CERES SYN1deg satellite-derived, "
            "from 2001 onward; the most recent few months come from the low-latency CERES FLASHFlux stream and are "
            "not reprocessed later); there are no data for earlier years and polar-night months, and these are left "
            "blank."
        ), "tr": (
            "Yeryüzünün gelen kısa dalga güneş ışınımının ne kadarını geri yansıttığını gösteren oran (0–1) türünde "
            "bir büyüklüktür. Kar/buz örtüsü, bitki örtüsü, toprak ve yüzey nemindeki değişimlerle farklılaşır; "
            "yüzeyin enerji kazanımı ve iklim geri beslemelerini anlamaya yardımcı olur. Albedo yükseldiğinde yüzey "
            "genellikle güneş enerjisinin daha büyük payını yansıtır; ancak bulutluluk, güneş açısı ve yüzey türü "
            "etkileri göz önünde tutulmalıdır. Kaynak: NASA POWER ALLSKY_SRF_ALB (CERES SYN1deg uydu türevli, "
            "2001'den itibaren; en son birkaç ay düşük gecikmeli CERES FLASHFlux akışından gelir ve sonradan "
            "yeniden işlenmez); daha önceki yıllar ve kutup gecesi ayları için veri yoktur ve boş bırakılır."
        )},
        "correlated_with": ["FRSEAICE (+)", "SWGDN (-)", "T2M (-)", "TSKINWTR (-)"],
    },
    "SWGDN": {
        "rank": 10,
        "importance_score_10": 8.3,
        "base_weight": 0.83,  # 8.3 / 10
        "short_name": "SWGDN",
        "collection": {"en": (
            "CERES SYN1deg Ed. 4.2; most recent ~3–4 months CERES FLASHFlux 4A (NASA POWER ALLSKY_SFC_SW_DWN; POWER's"
            " 1984–2000 GEWEX SRB values are not used)"
        ), "tr": "CERES SYN1deg Ed. 4.2; en son ~3–4 ay CERES FLASHFlux 4A (NASA POWER ALLSKY_SFC_SW_DWN; POWER'ın 1984–2000 GEWEX SRB değerleri kullanılmaz)"},
        "satellite_dataset": "M2TMNXRAD_5_12_4_SWGDN",
        "satellite_start_year": 2001,  # POWER 1984-2000'i GEWEX SRB'den verir; kaynak kırılması nedeniyle kullanılmaz
        "label": {"en": "10. 🔆 All-Sky Surface Downward Shortwave Radiation (SWGDN — NASA POWER / CERES, 2001+)", "tr": "10. 🔆 Tüm Gökyüzü Yüzeye İnen Kısa Dalga Işınım (SWGDN — NASA POWER / CERES, 2001+)"},
        "short_label": {"en": "Shortwave Solar Radiation (SWGDN)", "tr": "Kısa Dalga Güneş Işınımı (SWGDN)"},
        "unit": "W/m²",
        "agg_method": "mean",
        "category": {"en": "Surface Energy Budget & Radiation", "tr": "Yüzey Enerji Bütçesi & Radyasyon"},
        "platform": {"en": "NASA POWER monthly point API (power.larc.nasa.gov) — CERES SYN1deg / FLASHFlux (2001+)", "tr": "NASA POWER aylık nokta API (power.larc.nasa.gov) — CERES SYN1deg / FLASHFlux (2001+)"},
        "platform_url": "https://power.larc.nasa.gov/docs/",
        "sensors": {"en": "NASA CERES SYN1deg and FLASHFlux satellite-derived products (via NASA POWER)", "tr": "NASA CERES SYN1deg ve FLASHFlux uydu türevli ürünler (NASA POWER üzerinden)"},
        "surface": "any",
        "power_parameter": "ALLSKY_SFC_SW_DWN",
        "collection_url": None,
        "bounds": (0.0, None),  # fiziksel sınır (alt, üst); tahmin bandı burada kırpılır
        "polarity": 1.0,
        "student_explanation": {"en": (
            "The downward shortwave solar energy flux that passes through the atmosphere and reaches the surface "
            "(W/m²). It is a basic component of the surface energy budget; it can vary in time and by region due to "
            "cloudiness, atmospheric composition and surface conditions. It is suitable for studying trends linked to"
            " plant growth conditions, surface warming and climate feedbacks. This variable does not directly measure"
            " cloud amount; to separate the cause of a radiation trend it should be interpreted together with other "
            "data. Source: NASA POWER ALLSKY_SFC_SW_DWN, only from 2001-01 onward (CERES SYN1deg; the most recent few"
            " months from low-latency CERES FLASHFlux). POWER also provides GEWEX SRB values for 1984–2000, but the "
            "app does not use them because the 2001 source change creates an artificial jump in the series and "
            "produces spurious 'significant' trends."
        ), "tr": (
            "Atmosferden geçerek yüzeye ulaşan aşağı yönlü güneş kısa dalga enerji akısını ifade eder (W/m²). "
            "Yüzey enerji bütçesinin temel bileşenidir; bulutluluk, atmosferik bileşim ve yüzey koşullarının etkisiyle "
            "zaman ve bölgeye göre değişebilir. Bitki büyüme koşulları, yüzey ısınması ve iklim geri beslemeleriyle "
            "bağlantılı trendleri incelemeye uygundur. Bu değişken doğrudan bulut miktarını ölçmez; ışınım trendinin "
            "nedenini ayırmak için başka verilerle birlikte yorumlanmalıdır. Kaynak: NASA POWER ALLSKY_SFC_SW_DWN, "
            "yalnızca 2001-01'den itibaren (CERES SYN1deg; en son birkaç ay düşük gecikmeli CERES FLASHFlux). POWER "
            "1984–2000 için GEWEX SRB değerleri de verir, ancak 2001'deki kaynak değişimi seride yapay bir sıçrama "
            "oluşturup sahte 'anlamlı' trendler ürettiği için uygulama bu değerleri kullanmaz."
        )},
        "correlated_with": ["T2M (+)", "EVLAND (+)", "ALBEDO (-)"],
    },
})

# ---------------------------------------------------------------------------
# 3 HAZIR VAKA ÇALIŞMASI (kombinasyonlar ekibin kendi seçimidir)
# 1. Kuraklık: T2M + PRECTOTCORR + GWETROOT + EVLAND
# 2. Su Döngüsü (İç Anadolu vs Atina/Ege): PRECTOTCORR + GWETROOT + EVLAND + TQV
#    (RUNOFF POWER'da olmadığından yerine gerçek verisi olan EVLAND kullanılır)
# 3. Kutup / Enerji (Barents Denizi okyanus noktası, 2001+): T2M + FRSEAICE + ALBEDO + SWGDN
#    (TSKINWTR yalnızca buzsuz aylarda var; bileşik indeks onunla deniz buzunu sansürlü alt kümede görürdü.
#    ALBEDO ve SWGDN 2001'de başladığından vaka 2001'de başlar; ortak ay oranı %70 kuralını geçer.)
# end_year=None -> en son yıl (latest_year())
# ---------------------------------------------------------------------------
DEMO_SCENARIOS: Mapping[str, Dict[str, Any]] = LocalizedMapping({
    "aral_tuz": {
        "id": "aral_tuz",
        "title": {"en": "1. 🏜️ Drought Combination (T2M + PRECTOTCORR + GWETROOT + EVLAND)", "tr": "1. 🏜️ Kuraklık Kombinasyonu (T2M + PRECTOTCORR + GWETROOT + EVLAND)"},
        "route": "/analyze/single",
        "mode": "single",
        "summary": {"en": (
            "The team's Drought combination: studies 2 m air temperature (T2M), precipitation (PRECTOTCORR), root "
            "zone soil moisture (GWETROOT) and land surface evaporation (EVLAND) together from 1981 to today."
        ), "tr": (
            "Ekibin Kuraklık kombinasyonu: 2 m hava sıcaklığı (T2M), yağış (PRECTOTCORR), "
            "kök bölgesi toprak nemi (GWETROOT) ve kara yüzeyi buharlaşmasını (EVLAND) 1981'den bugüne birlikte inceler."
        )},
        "markers": [
            {"lat": 45.00, "lon": 59.60, "name": {"en": "Aral Sea Basin (dried-up lake bed, Karakalpakstan / Uzbekistan)", "tr": "Aral Gölü Havzası (kurumuş göl tabanı, Karakalpakistan / Özbekistan)"}},
        ],
        "alt_marker_tuz": {"lat": 38.83, "lon": 33.33, "name": {"en": "Lake Tuz (Tuz Gölü) Closed Basin (Türkiye)", "tr": "Tuz Gölü Kapalı Havzası (Türkiye)"}},
        "variables": ["T2M", "PRECTOTCORR", "GWETROOT", "EVLAND"],
        "start_year": POWER_FIRST_YEAR,
        "end_year": None,
        "hide_seasonality": True,
    },
    "water_cycle": {
        "id": "water_cycle",
        "title": {"en": "2. 💧 Water Cycle Combination (PRECTOTCORR + GWETROOT + EVLAND + TQV)", "tr": "2. 💧 Su Döngüsü Kombinasyonu (PRECTOTCORR + GWETROOT + EVLAND + TQV)"},
        "route": "/analyze/dual",
        "mode": "dual",
        "summary": {"en": (
            "The team's Water Cycle combination: compares the water-budget variables of two regions using "
            "precipitation (PRECTOTCORR), root zone soil moisture (GWETROOT), land surface evaporation (EVLAND) and "
            "total precipitable water vapour (TQV) (RUNOFF is not used because it is not available in NASA POWER)."
        ), "tr": (
            "Ekibin Su Döngüsü kombinasyonu: Yağış (PRECTOTCORR), kök bölgesi toprak nemi (GWETROOT), "
            "kara yüzeyi buharlaşması (EVLAND) ve toplam çökelebilir su buharı (TQV) ile iki bölgenin su bütçesi "
            "değişkenlerini kıyaslar (RUNOFF NASA POWER'da bulunmadığı için kullanılmaz)."
        )},
        "markers": [
            {"lat": 39.93, "lon": 32.85, "name": {"en": "Ankara / Central Anatolia (Türkiye)", "tr": "Ankara / İç Anadolu (Türkiye)"}},
            {"lat": 37.98, "lon": 23.72, "name": {"en": "Athens / Aegean Coast (Greece)", "tr": "Atina / Ege Kıyısı (Yunanistan)"}},
        ],
        "variables": ["PRECTOTCORR", "GWETROOT", "EVLAND", "TQV"],
        "start_year": POWER_FIRST_YEAR,
        "end_year": None,
        "hide_seasonality": True,
    },
    "arctic_albedo": {
        "id": "arctic_albedo",
        "title": {"en": "3. ❄️ Polar / Energy Combination (T2M + FRSEAICE + ALBEDO + SWGDN, 2001+)", "tr": "3. ❄️ Kutup / Enerji Kombinasyonu (T2M + FRSEAICE + ALBEDO + SWGDN, 2001+)"},
        "route": "/analyze/single",
        "mode": "single",
        "summary": {"en": (
            "The team's Polar/Energy combination: studies 2 m air temperature (T2M), sea-ice covered area fraction "
            "(FRSEAICE), all-sky surface albedo (ALBEDO) and surface downward shortwave radiation (SWGDN) at an ocean"
            " point in the Barents Sea from 2001 to today (ALBEDO and SWGDN start in 2001 with CERES data). Albedo is"
            " blank in polar-night months; ocean surface temperature (TSKINWTR), which is defined only in ice-free "
            "months, is not included in the composite index, because it would show sea ice only in ice-free months."
        ), "tr": (
            "Ekibin Kutup/Enerji kombinasyonu: Barents Denizi'nde bir okyanus noktasında 2 m hava sıcaklığı (T2M), "
            "deniz buzu kaplı alan oranı (FRSEAICE), tüm gökyüzü yüzey albedosu (ALBEDO) ve yüzeye inen kısa dalga "
            "ışınımını (SWGDN) 2001'den bugüne inceler (ALBEDO ve SWGDN CERES verisiyle 2001'de başlar). Albedonun "
            "kutup gecesi ayları boş kalır; yalnızca buzsuz aylarda tanımlı okyanus yüzey sıcaklığı (TSKINWTR) bileşik "
            "indekse alınmaz, çünkü deniz buzunu yalnızca buzsuz aylarda gösterirdi."
        )},
        "markers": [
            {"lat": 76.00, "lon": 30.00, "name": {"en": "Barents Sea (southeast of Svalbard)", "tr": "Barents Denizi (Svalbard'ın güneydoğusu)"}},
        ],
        "variables": ["T2M", "FRSEAICE", "ALBEDO", "SWGDN"],
        "start_year": 2001,
        "end_year": None,
        "hide_seasonality": True,
    },
})

# ---------------------------------------------------------------------------
# DİNAMİK TERS COĞRAFİ KODLAMA (REVERSE GEOCODING: ÜLKE / ŞEHİR / BİYOM)
# ---------------------------------------------------------------------------
def _iso2_to_flag(iso2: str) -> str:
    """2 harfli ülke kodunu (örn. 'tr', 'az', 'kz') bayrak emojisine çevirir."""
    if not iso2 or len(iso2) != 2 or not iso2.isalpha():
        return "🌍"
    return "".join(chr(0x1F1E6 + ord(c.upper()) - ord("A")) for c in iso2)


# Nominatim kullanım politikası (https://operations.osmfoundation.org/policies/nominatim/):
# tanımlayıcı User-Agent (iletişim bilgisi EDD_CONTACT ortam değişkeninden), en fazla 1 istek/sn,
# yalnızca başarılı yanıtlar önbelleğe alınır, adların yanında "© OpenStreetMap contributors" gösterilir.
_NOMINATIM_LOCK = threading.Lock()
_NOMINATIM_LAST_CALL = [0.0]


def _app_user_agent() -> str:
    contact = os.environ.get("EDD_CONTACT", "").strip()
    base = "NASA-SpaceApps-EarthDataDetective/2.1 (NASA Space Apps Challenge student project"
    return f"{base}; contact: {contact})" if contact else f"{base})"


@st.cache_data(ttl=86400, show_spinner=False)
def _nominatim_reverse_lookup(lat_r: float, lon_r: float, lang: str = "tr") -> Dict[str, str]:
    """
    OpenStreetMap Nominatim'den (accept-language=tr,en) koordinatın ülke ve bölge adını çeker.
    Hata durumunda DataServiceError fırlatır (st.cache_data istisnaları önbelleğe almaz).
    """
    with _NOMINATIM_LOCK:
        wait = 1.0 - (time.monotonic() - _NOMINATIM_LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(
                NOMINATIM_REVERSE_URL,
                params={
                    "lat": round(float(lat_r), 3),
                    "lon": round(float(lon_r), 3),
                    "format": "jsonv2",
                    "accept-language": "tr,en" if lang == "tr" else "en",
                    "zoom": 8,
                },
                headers={"User-Agent": _app_user_agent()},
                timeout=5.0,
            )
        except requests.RequestException as exc:
            raise DataServiceError(L(f"Could not reach Nominatim ({type(exc).__name__})", f"Nominatim'e ulaşılamadı ({type(exc).__name__})")) from exc
        finally:
            _NOMINATIM_LAST_CALL[0] = time.monotonic()
    if resp.status_code != 200:
        raise DataServiceError(f"Nominatim HTTP {resp.status_code}", resp.status_code)
    try:
        data = resp.json()
    except ValueError as exc:
        raise DataServiceError(L("Could not read the Nominatim response", "Nominatim yanıtı okunamadı")) from exc
    if "error" in data:
        # Nominatim adres bulamadı: genellikle açık deniz/okyanus
        return {"is_water": "1"}
    addr = data.get("address", {})
    country = addr.get("country", "").strip()
    cc = addr.get("country_code", "").strip()
    region = (
        addr.get("state")
        or addr.get("province")
        or addr.get("city")
        or addr.get("region")
        or addr.get("county")
        or addr.get("town")
        or data.get("name")
        or ""
    ).strip()
    if not country:
        return {"is_water": "1"}
    return {
        "is_water": "0",
        "country": f"{_iso2_to_flag(cc)} {country}",
        "city_region": region or "—",
        "country_code": cc.lower(),
    }


def _classify_water_basin(lat: float, lon: float) -> Tuple[str, str]:
    """Nominatim adres bulamadığında (su) kaba deniz/okyanus adı döndürür (dikdörtgen kutulara dayalı yaklaşık ad)."""
    if 45.0 <= lat <= 47.3 and 34.8 <= lon <= 39.3:
        return L("🌊 Sea of Azov", "🌊 Azak Denizi"), L("Semi-enclosed Inland Sea", "Yarı Kapalı İç Deniz")
    if 40.9 <= lat <= 46.7 and 27.4 <= lon <= 41.8:
        return L("🌊 Black Sea", "🌊 Karadeniz"), L("Semi-enclosed Inland Sea Basin", "Yarı Kapalı İç Deniz Havzası")
    if 30.0 <= lat <= 46.0 and -5.5 <= lon <= 36.5:
        if 35.0 <= lat <= 41.0 and 22.5 <= lon <= 28.5:
            return L("🌊 Aegean Sea", "🌊 Ege Denizi"), L("Eastern Mediterranean Belt", "Doğu Akdeniz Kuşağı")
        return L("🌊 Mediterranean Sea", "🌊 Akdeniz"), L("Mediterranean Marine Basin", "Akdeniz Denizel Havzası")
    if 36.5 <= lat <= 47.1 and 46.5 <= lon <= 54.5:
        return L("🌊 Caspian Sea", "🌊 Hazar Denizi"), L("Enclosed Inland Sea Basin", "Kapalı İç Deniz Havzası")
    if 12.0 <= lat <= 30.0 and 32.0 <= lon <= 44.0:
        return L("🌊 Red Sea", "🌊 Kızıldeniz"), L("Subtropical Sea Basin", "Subtropikal Deniz Havzası")
    if 23.5 <= lat <= 30.5 and 47.5 <= lon <= 56.5:
        return L("🌊 Persian Gulf", "🌊 Basra Körfezi"), L("Shallow Marine Basin", "Sığ Denizel Havza")
    if 51.0 <= lat <= 62.0 and -4.0 <= lon <= 9.0:
        return L("🌊 North Sea", "🌊 Kuzey Denizi"), L("Northeast Atlantic Coast", "Kuzeydoğu Atlantik Kıyısı")
    if 53.5 <= lat <= 66.0 and 9.5 <= lon <= 30.0:
        return L("🌊 Baltic Sea", "🌊 Baltık Denizi"), L("Northern European Inland Sea", "Kuzey Avrupa İç Denizi")
    if -3.1 <= lat <= 0.6 and 31.5 <= lon <= 34.9:
        return L("🌊 Lake Victoria", "🌊 Victoria Gölü"), L("East African Great Lake", "Doğu Afrika Büyük Gölü")
    if 18.0 <= lat <= 31.0 and -98.0 <= lon <= -80.5:
        return L("🌊 Gulf of Mexico", "🌊 Meksika Körfezi"), L("Western Atlantic Marginal Sea", "Batı Atlantik Kenar Denizi")
    if 9.0 <= lat < 22.0 and -88.0 <= lon <= -60.0:
        return L("🌊 Caribbean Sea", "🌊 Karayip Denizi"), L("Western Atlantic Marginal Sea", "Batı Atlantik Kenar Denizi")
    if 51.0 <= lat <= 64.0 and -95.0 <= lon <= -77.0:
        return L("🌊 Hudson Bay", "🌊 Hudson Körfezi"), L("North American Inland Sea", "Kuzey Amerika İç Denizi")
    if 0.0 <= lat <= 25.0 and 99.0 <= lon <= 121.0:
        return L("🌊 South China Sea", "🌊 Güney Çin Denizi"), L("Western Pacific Marginal Sea", "Batı Pasifik Kenar Denizi")
    if 33.0 <= lat <= 52.0 and 127.0 <= lon <= 142.0:
        return L("🌊 Sea of Japan", "🌊 Japon Denizi"), L("Western Pacific Marginal Sea", "Batı Pasifik Kenar Denizi")
    if 5.0 <= lat <= 23.0 and 80.0 <= lon <= 95.0:
        return L("🌊 Bay of Bengal", "🌊 Bengal Körfezi"), L("Indian Ocean", "Hint Okyanusu")
    if 5.0 <= lat <= 25.0 and 51.0 <= lon <= 77.0:
        return L("🌊 Arabian Sea", "🌊 Arap Denizi"), L("Indian Ocean", "Hint Okyanusu")
    if 68.0 <= lat <= 81.0 and 15.0 <= lon <= 60.0:
        return L("❄️ Barents Sea", "❄️ Barents Denizi"), L("Arctic Ocean Marginal Sea", "Arktik Okyanusu Kenar Denizi")
    if lat >= 66.0:
        return L("❄️ Arctic Ocean", "❄️ Arktik Okyanusu"), L("North Polar Sea Basin", "Kuzey Kutup Deniz Havzası")
    if lat <= -60.0:
        return L("❄️ Southern Ocean", "❄️ Güney Okyanusu"), L("Antarctic Sea Basin", "Antarktik Deniz Havzası")
    hemi = "Kuzey " if lat > 0 else ("Güney " if lat < 0 else "Ekvatoral ")
    hemi_en = "North " if lat > 0 else ("South " if lat < 0 else "Equatorial ")
    if (lat >= 0 and -80.0 <= lon <= 20.0) or (lat < 0 and -68.0 <= lon <= 20.0):
        return L(f"🌊 {hemi_en}Atlantic Ocean", f"🌊 {hemi}Atlantik Okyanusu"), L("Open Ocean Basin", "Açık Okyanus Havzası")
    if -60.0 <= lat <= 30.0 and 20.0 <= lon <= 120.0:
        return L("🌊 Indian Ocean", "🌊 Hint Okyanusu"), L("Open Ocean Basin", "Açık Okyanus Havzası")
    return L(f"🌊 {hemi_en}Pacific Ocean", f"🌊 {hemi}Pasifik Okyanusu"), L("Open Ocean Basin", "Açık Okyanus Havzası")


# Yüzey sınıfı yalnızca verinin söyleyebildiği kadardır: NASA POWER hücre maskesi (okyanus / kara / kara buzu / iç su) +
# kutup daireleri enlemi (|enlem| >= 66.5°). Daha ince bir biyom (çöl, orman, şehir) için savunulabilir
# çevrimdışı bir kural olmadığından eski enlem-bandı "Çöl" kuralı kaldırıldı.
BIOME_DEFINITIONS: Mapping[str, Dict[str, Any]] = LocalizedMapping({
    "Ocean": {
        "name": {"en": "🌊 Ocean / Sea Cell (NASA POWER mask)", "tr": "🌊 Okyanus / Deniz Hücresi (NASA POWER maskesi)"},
        "recommended": ["T2M", "PRECTOTCORR", "TSKINWTR", "TQV"],
        "reason": {"en": (
            "Team recommendation: land variables (GWETROOT, EVLAND, RUNOFF) are not defined in an ocean cell; "
            "atmosphere and ocean-surface variables can be used."
        ), "tr": "Ekip önerisi: okyanus hücresinde kara değişkenleri (GWETROOT, EVLAND, RUNOFF) tanımlı değildir; atmosfer ve okyanus yüzeyi değişkenleri kullanılabilir."},
    },
    "Land": {
        "name": {"en": "🏞️ Land Cell (NASA POWER mask)", "tr": "🏞️ Kara Hücresi (NASA POWER maskesi)"},
        "recommended": ["T2M", "PRECTOTCORR", "GWETROOT", "EVLAND"],
        "reason": {"en": (
            "Team recommendation (Drought combination): in a land cell, temperature, precipitation, soil moisture and"
            " evaporation together show the water balance. Ocean variables (TSKINWTR, FRSEAICE) are not shown in a "
            "land cell."
        ), "tr": "Ekip önerisi (Kuraklık kombinasyonu): kara hücresinde sıcaklık, yağış, toprak nemi ve buharlaşma birlikte su dengesini gösterir. Okyanus değişkenleri (TSKINWTR, FRSEAICE) kara hücresinde gösterilmez."},
    },
    "Polar_Ocean": {
        "name": {"en": "❄️ Polar Ocean Cell (|lat| ≥ 66.5°, NASA POWER mask)", "tr": "❄️ Kutup Okyanus Hücresi (|enlem| ≥ 66.5°, NASA POWER maskesi)"},
        "recommended": ["T2M", "FRSEAICE", "ALBEDO", "SWGDN"],
        "reason": {"en": (
            "Team recommendation (Polar/Energy combination): air temperature, sea ice, albedo and solar radiation. "
            "ALBEDO and SWGDN start in 2001: choose a range after 2001 for the composite index. Albedo is blank in "
            "polar-night months."
        ), "tr": "Ekip önerisi (Kutup/Enerji kombinasyonu): hava sıcaklığı, deniz buzu, albedo ve güneş ışınımı. ALBEDO ve SWGDN 2001'de başlar: bileşik indeks için 2001 sonrası bir aralık seçin. Albedonun kutup gecesi ayları boş kalır."},
    },
    "Polar_Land": {
        "name": {"en": "🏔️ Polar Land Cell (|lat| ≥ 66.5°, NASA POWER mask)", "tr": "🏔️ Kutup Kara Hücresi (|enlem| ≥ 66.5°, NASA POWER maskesi)"},
        "recommended": ["T2M", "PRECTOTCORR", "ALBEDO", "SWGDN"],
        "reason": {"en": (
            "Team recommendation: in a polar land cell, temperature, precipitation, albedo and solar radiation. Ocean"
            " variables (TSKINWTR, FRSEAICE) are not shown in a land cell."
        ), "tr": "Ekip önerisi: kutup kara hücresinde sıcaklık, yağış, albedo ve güneş ışınımı. Okyanus değişkenleri (TSKINWTR, FRSEAICE) kara hücresinde gösterilmez."},
    },
    "Land_Ice": {
        "name": {"en": "🧊 Land Ice Cell (ice sheet / ice shelf, NASA POWER mask)", "tr": "🧊 Kara Buzu Hücresi (buz örtüsü / buz sahanlığı, NASA POWER maskesi)"},
        "recommended": ["T2M", "PRECTOTCORR", "ALBEDO", "SWGDN"],
        "reason": {"en": (
            "Team recommendation: on an ice sheet/ice shelf, temperature, precipitation (snow), albedo and solar "
            "radiation. Soil variables (GWETROOT, EVLAND) are undefined here in POWER; since an ice surface is not "
            "ocean, TSKINWTR and FRSEAICE are not shown either."
        ), "tr": "Ekip önerisi: buz örtüsü/sahanlığında sıcaklık, yağış (kar), albedo ve güneş ışınımı. Toprak değişkenleri (GWETROOT, EVLAND) burada POWER'da tanımsızdır; buz yüzeyi okyanus olmadığı için TSKINWTR ve FRSEAICE de gösterilmez."},
    },
    "Inland_Water": {
        "name": {"en": "🏞️ Inland Water (Large Lake) Cell (NASA POWER mask)", "tr": "🏞️ İç Su (Büyük Göl) Hücresi (NASA POWER maskesi)"},
        "recommended": ["T2M", "PRECTOTCORR", "TQV", "SWGDN"],
        "reason": {"en": (
            "Team recommendation: atmospheric variables in a lake cell. Soil variables are undefined here in POWER; "
            "since a lake surface is not ocean, TSKINWTR and FRSEAICE are not shown."
        ), "tr": "Ekip önerisi: göl hücresinde atmosfer değişkenleri. Toprak değişkenleri burada POWER'da tanımsızdır; göl yüzeyi okyanus olmadığı için TSKINWTR ve FRSEAICE gösterilmez."},
    },
    "Unknown": {
        "name": {"en": "❔ Surface class could not be determined", "tr": "❔ Yüzey sınıfı belirlenemedi"},
        "recommended": [],
        "reason": {"en": (
            "The land/ocean status of the cell could not be determined because no valid response was received from "
            "NASA POWER (details in parentheses)."
        ), "tr": "NASA POWER'dan geçerli yanıt alınamadığı için hücrenin kara/okyanus durumu belirlenemedi (ayrıntı parantez içinde)."},
    },
})


def _demo_point_name(lat: float, lon: float) -> Optional[str]:
    """Yalnızca hazır vakaların kendi noktaları için (tam eşleşme) sabit ad döndürür."""
    for sc in DEMO_SCENARIOS.values():
        pts = list(sc["markers"]) + ([sc["alt_marker_tuz"]] if "alt_marker_tuz" in sc else [])
        for m in pts:
            if round(float(m["lat"]), 2) == round(float(lat), 2) and round(float(m["lon"]), 2) == round(float(lon), 2):
                return m["name"]
    return None


def power_cell(lat: float, lon: float) -> Dict[str, Tuple[float, float]]:
    """
    Noktanın düştüğü NASA POWER kaynak hücrelerinin merkezleri (POWER en yakın hücreyi verir; sınırlar
    API'ye karşı denendi: MERRA-2 enlem sınırı 39.75/40.25, boylam 32.8125; CERES 1° sınırları tam sayılar).
    """
    la_m = float(np.clip(round(float(lat) / MERRA2_DLAT) * MERRA2_DLAT, -90.0, 90.0))
    lo_m = -180.0 + round((float(lon) + 180.0) / MERRA2_DLON) * MERRA2_DLON
    lo_m = lo_m - 360.0 if lo_m >= 180.0 else lo_m
    la_c = float(min(np.floor(float(lat)), 89.0)) + 0.5
    lo_c = float(min(np.floor(float(lon)), 179.0)) + 0.5
    return {"merra2": (round(la_m, 3), round(lo_m, 4)), "radiation": (la_c, lo_c)}


def power_cell_note(lat: float, lon: float, elevation_m: Optional[float] = None) -> str:
    c = power_cell(lat, lon)
    elev = L(f"; mean cell elevation {elevation_m:.0f} m", f"; hücre ortalama yüksekliği {elevation_m:.0f} m") if elevation_m is not None else ""
    return L(
        f"NASA POWER cell: MERRA-2 centre {c['merra2'][0]:.2f}°, {c['merra2'][1]:.3f}° (0.5°×0.625°), radiation "
        f"(ALBEDO, SWGDN) CERES 1° centre {c['radiation'][0]:.1f}°, {c['radiation'][1]:.1f}°{elev}. The values are "
        f"those of these cells, not of the exact point; two points in the same cell give the same series.",
        f"NASA POWER hücresi: MERRA-2 merkezi {c['merra2'][0]:.2f}°, {c['merra2'][1]:.3f}° (0.5°×0.625°), ışınım "
        f"(ALBEDO, SWGDN) CERES 1° merkezi {c['radiation'][0]:.1f}°, {c['radiation'][1]:.1f}°{elev}. Değerler bu "
        f"hücrelerin değeridir, tam noktanın değil; aynı hücredeki iki nokta aynı seriyi verir.",
    )


MIXED_CELL_NOTE = {"en": (
    "⚠️ This NASA POWER cell contains sea/lake as well as land (coastal or lake cell). Because POWER gives values for "
    "land variables (GWETROOT, EVLAND) in this cell, the app treats the cell as 'land': these values describe the land "
    "part of the cell, not the water surface. Ocean variables (TSKINWTR, FRSEAICE) are not shown here."
), "tr": (
    "⚠️ Bu NASA POWER hücresi kara ile birlikte deniz/göl de içeriyor (kıyı veya göl hücresi). POWER bu hücrede kara "
    "değişkenlerine (GWETROOT, EVLAND) değer verdiği için uygulama hücreyi 'kara' sayar: bu değerler hücrenin kara "
    "kısmını tanımlar, su yüzeyini değil. Okyanus değişkenleri (TSKINWTR, FRSEAICE) burada gösterilmez."
)}


def reverse_geocode_and_detect_biome(lat: float, lon: float) -> Dict[str, Any]:
    """
    Ad: önce OpenStreetMap Nominatim; Nominatim erişilemezse yalnızca hazır vaka noktaları için sabit ad,
    aksi halde "ad alınamadı". Yüzey sınıfı: NASA POWER hücre maskesi + kutup enlemi (power_surface_class).
    """
    name_note = OSM_ATTRIBUTION
    try:
        nom_res = _nominatim_reverse_lookup(round(float(lat), 2), round(float(lon), 2), get_lang())
    except DataServiceError as exc:
        nom_res = None
        name_note = L(f"Name service error: {exc}", f"Ad servisi hatası: {exc}")
    if nom_res is None:
        demo_name = _demo_point_name(lat, lon)
        country, city_region = ("📍 " + demo_name, L("demo case point", "hazır vaka noktası")) if demo_name else (L("📍 Name unavailable", "📍 Ad alınamadı"), f"{lat:.2f}°, {lon:.2f}°")
        if demo_name:
            name_note += L(" — showing the demo case name", " — hazır vaka adı gösteriliyor")
    elif nom_res.get("is_water") == "1":
        country, city_region = _classify_water_basin(lat, lon)
        name_note = L(
            f"{OSM_ATTRIBUTION} (no address found → approximate sea name, from the app's own boxes)",
            f"{OSM_ATTRIBUTION} (adres bulunamadı → yaklaşık deniz adı, uygulamanın kendi kutularından)",
        )
    else:
        country, city_region = nom_res["country"], nom_res["city_region"]

    surface_note = ""
    mixed_note = ""
    elevation = None
    try:
        bundle = get_power_point_bundle(lat, lon)
        surface = surface_from_bundle(bundle)
        elevation = bundle["_elevation_m"]
        polar = abs(float(lat)) >= POLAR_LAT
        if surface in ("ocean", "land"):
            biome_code = ("Polar_" if polar else "") + ("Ocean" if surface == "ocean" else "Land")
        else:
            biome_code = {"land_ice": "Land_Ice", "inland_water": "Inland_Water"}[surface]
        # Karışık hücre: kara sayılan hücrede deniz buzu (yalnızca suda olur) veya adres bulunamayan (deniz) nokta.
        water_hint = (nom_res or {}).get("is_water") == "1"
        ice_hint = any(v is not None and float(v) > 0 for k, v in bundle["FRSEAICE"].items() if not k.endswith("13"))
        if surface == "land" and (water_hint or ice_hint):
            mixed_note = pick(MIXED_CELL_NOTE)
    except DataServiceError as exc:
        biome_code = "Unknown"
        surface_note = f" ({exc})"

    biome_meta = BIOME_DEFINITIONS[biome_code]
    return {
        "country": country,
        "city_region": city_region,
        "name_source": name_note,
        "biome_code": biome_code,
        "biome_name": biome_meta["name"] + surface_note,
        "recommended_vars": biome_meta["recommended"],
        "biome_reason": biome_meta["reason"],
        "mixed_cell_note": mixed_note,
        "power_cell_note": power_cell_note(lat, lon, elevation),
        "display_label": f"{country} — {city_region} ({lat:.2f}°, {lon:.2f}°)",
        "tooltip_html": (
            f"<b>{L('Country / Belt', 'Ülke / Kuşak')}:</b> {country}<br>"
            f"<b>{L('Region / Basin', 'Bölge / Havza')}:</b> {city_region}<br>"
            f"<b>{L('Surface class', 'Yüzey sınıfı')}:</b> {biome_meta['name']}{surface_note}<br>"
            f"<b>{L('Latitude / Longitude', 'Enlem / Boylam')}:</b> {lat:.2f}°, {lon:.2f}°<br>"
            f"<small>{name_note}</small>"
        ),
    }


# ---------------------------------------------------------------------------
# NASA POWER AYLIK NOKTA VERİSİ
# ---------------------------------------------------------------------------
def _power_request(lat: float, lon: float, end_year: int, parameters: str = _POWER_PARAMETERS) -> Dict[str, Dict[str, float]]:
    """Tek bir POWER isteği; başarısızlıkta DataServiceError (durum kodu ve POWER mesajıyla)."""
    try:
        resp = requests.get(
            POWER_MONTHLY_POINT_URL,
            params={
                "parameters": parameters,
                "community": "SB",
                "longitude": round(float(lon), 3),
                "latitude": round(float(lat), 3),
                "start": POWER_FIRST_YEAR,
                "end": int(end_year),
                "format": "JSON",
            },
            headers={"User-Agent": _app_user_agent()},
            timeout=30.0,
        )
    except requests.RequestException as exc:
        raise DataServiceError(L(f"Could not reach the NASA POWER service ({type(exc).__name__})", f"NASA POWER servisine ulaşılamadı ({type(exc).__name__})")) from exc
    if resp.status_code != 200:
        try:
            msg = "; ".join(resp.json().get("messages", []))[:300]
        except ValueError:
            msg = resp.text[:200]
        raise DataServiceError(f"NASA POWER HTTP {resp.status_code}: {msg}", resp.status_code)
    try:
        js = resp.json()
        bundle = dict(js["properties"]["parameter"])
        # Hücre yüzey yüksekliği (m), yüzey maskesi için; parametre olmayan ayrılmış anahtar
        bundle["_elevation_m"] = float(js["geometry"]["coordinates"][2])
        # Eksik parametre veya sayısal olmayan değer içeren 200 yanıtı da hatadır (önbelleğe girmesin).
        for p_name in parameters.split(","):
            if not bundle.get(p_name):
                raise KeyError(p_name)
            for v in bundle[p_name].values():
                if v is not None and not isinstance(v, (int, float)):
                    raise TypeError(p_name)
        return bundle
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise DataServiceError(L(f"NASA POWER response is not in the expected format ({type(exc).__name__}: {exc})", f"NASA POWER yanıtı beklenen biçimde değil ({type(exc).__name__}: {exc})")) from exc


@st.cache_data(ttl=7200, show_spinner=False)
def _fetch_power_point_bundle(lat: float, lon: float) -> Dict[str, Dict[str, float]]:
    """
    Koordinat için tüm POWER parametrelerini 1981'den içinde bulunulan yıla kadar tek istekle çeker.
    Yalnızca başarılı yanıt önbelleğe alınır: hata DataServiceError olarak fırlatılır (st.cache_data
    istisnaları saklamaz). Yıl başında POWER cari yılı henüz sunmuyorsa (HTTP 422 'end date') bir önceki yıl istenir.
    """
    end_year = _dt.datetime.now(_dt.timezone.utc).year
    try:
        return _power_request(lat, lon, end_year)
    except DataServiceError as exc:
        if exc.status == 422 and "end date" in str(exc).lower():
            return _power_request(lat, lon, end_year - 1)
        raise


_OUTAGE_KEY = "_power_outage_this_run"


def reset_service_failure() -> None:
    """Uygulama her çalıştırmanın başında çağırır: önceki çalıştırmadaki servis kesintisi kaydını siler."""
    st.session_state[_OUTAGE_KEY] = None


def get_power_point_bundle(lat: float, lon: float) -> Dict[str, Dict[str, float]]:
    """
    POWER noktası (0.01° yuvarlamalı önbellek anahtarı). Hata: DataServiceError.
    Bağlantı/zaman aşımı/5xx/429 kesintisinden sonra aynı çalıştırmada yeniden denenmez (her çağrı 30 sn
    beklemesin); kayıt reset_service_failure() ile her çalıştırmanın başında silinir.
    """
    prior = st.session_state.get(_OUTAGE_KEY)
    if prior:
        raise DataServiceError(L(f"{prior} (not retried in this page run)", f"{prior} (bu sayfa çalıştırmasında yeniden denenmedi)"))
    try:
        return _fetch_power_point_bundle(round(float(lat), 2), round(float(lon), 2))
    except DataServiceError as exc:
        if exc.status is None or exc.status >= 500 or exc.status in (408, 429):
            st.session_state[_OUTAGE_KEY] = str(exc)
        raise


def surface_from_bundle(bundle: Dict[str, Any]) -> str:
    """
    NASA POWER verisinden hücre yüzey sınıfı: 'land' | 'ocean' | 'land_ice' | 'inland_water'.
    1. Kara-yalnız parametrelerden (GWETROOT, EVLAND) herhangi biri gerçek değer veriyorsa -> 'land'.
    2. İkisi de hep -999 ise (okyanus, buz örtüsü, buz sahanlığı veya büyük göl):
       - hücre yüksekliği < 40 m VE (herhangi bir ayda FRSEAICE > 0.15 [deniz buzu yalnızca denizde olur]
         VEYA "FRSEAICE <= 0.15 ama TS < -2 °C" olan ay oranı <= %5) -> 'ocean'
         (açık su -2 °C'den soğuk olamaz; buz sahanlıkları 0 m'dedir, hiç deniz buzu göstermez ve her ay bu koşulu sağlar);
       - değilse ortalama TS < 0 °C -> 'land_ice' (Antarktika, Grönland iç kesimi, buz sahanlıkları);
       - değilse -> 'inland_water' (yüksekteki büyük göller, ör. Victoria Gölü 1134 m).
    Kıyı/göl içeren karışık hücreler, POWER kara-yalnız parametrelere değer verdiği için 'land' sayılır
    (bkz. mixed_cell_note). Sayısal olmayan değerler DataServiceError verir.
    """
    try:
        return _surface_from_bundle(bundle)
    except (TypeError, ValueError, AttributeError) as exc:
        raise DataServiceError(L(f"Non-numeric value in NASA POWER response; surface mask could not be built ({exc})", f"NASA POWER yanıtında sayısal olmayan değer; yüzey maskesi kurulamadı ({exc})")) from exc


def surface_rule_inputs(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Kara olmayan hücre için yüzey kuralının girdileri: yükseklik, deniz buzu var mı, 'buzsuz ama soğuk' oranı, ort. TS."""
    if "TS" not in bundle or "FRSEAICE" not in bundle or "_elevation_m" not in bundle:
        raise DataServiceError(L("NASA POWER response has no TS/FRSEAICE/elevation; surface mask could not be built", "NASA POWER yanıtında TS/FRSEAICE/yükseklik yok; yüzey maskesi kurulamadı"))
    ts = {k: float(v) for k, v in bundle["TS"].items() if not k.endswith("13") and v is not None and float(v) > -900}
    ice = bundle["FRSEAICE"]
    if not ts:
        raise DataServiceError(L("NASA POWER returned no TS for this cell; surface mask could not be built", "NASA POWER bu hücre için TS döndürmedi; yüzey maskesi kurulamadı"))
    cold_open = sum(1 for k, t in ts.items() if -900 < float(ice.get(k, -999)) <= SEA_ICE_MASK_THRESHOLD and t < OPEN_WATER_MIN_TS)
    return {
        "elevation_m": float(bundle["_elevation_m"]),
        "has_sea_ice": any(v is not None and float(v) > SEA_ICE_MASK_THRESHOLD for k, v in ice.items() if not k.endswith("13")),
        "cold_open_fraction": cold_open / len(ts),
        "mean_ts": float(np.mean(list(ts.values()))),
    }


def surface_from_rule_inputs(r: Dict[str, Any]) -> str:
    if r["elevation_m"] < OCEAN_MAX_ELEVATION_M and (r["has_sea_ice"] or r["cold_open_fraction"] <= OCEAN_MAX_COLD_OPEN_FRACTION):
        return "ocean"
    return "land_ice" if r["mean_ts"] < 0.0 else "inland_water"


def _surface_from_bundle(bundle: Dict[str, Any]) -> str:
    for p in _LAND_ONLY_POWER_PARAMS:
        if p not in bundle:
            raise DataServiceError(L(f"NASA POWER response has no {p}; surface mask could not be built", f"NASA POWER yanıtında {p} yok; yüzey maskesi kurulamadı"))
        if any(v is not None and float(v) > -900 for k, v in bundle[p].items() if not k.endswith("13")):
            return "land"
    return surface_from_rule_inputs(surface_rule_inputs(bundle))


def power_surface_class(lat: float, lon: float) -> str:
    """'land' | 'ocean' | 'land_ice' | 'inland_water' (NASA POWER veri maskesi). Hata: DataServiceError."""
    return surface_from_bundle(get_power_point_bundle(lat, lon))


def power_bundle_to_monthly(bundle: Dict[str, Dict[str, float]], variable: str) -> Dict[Tuple[int, int], float]:
    """
    POWER yanıtını değişkenin uygulama birimine çevirir; yalnızca gerçek (dolgu olmayan) aylar döner.
    - PRECTOTCORR: PRECTOTCORR_SUM zaten aylık toplamdır (POWER meta verisi 'mm/day' dese de) -> olduğu gibi.
    - EVLAND: mm/gün oranı x ayın gün sayısı -> mm/ay (negatif değerler gerçek yoğuşma/kırağıdır, kırpılmaz).
    - TSKINWTR: POWER TS; FRSEAICE > SEA_ICE_MASK_THRESHOLD olan (veya buz bilgisi eksik) aylar ve
      TS < OPEN_WATER_MIN_TS (-2 °C; deniz suyu daha soğuk olamaz) olan aylar atılır.
    - Değişkenin satellite_start_year'ından önceki aylar atılır (SWGDN: yalnızca 2001+ CERES dönemi).
    Okyanus/kara uygunluğu bu fonksiyonda değil, çağıranda (surface_from_bundle) denetlenir.
    """
    meta = OFFICIAL_VARIABLES[variable]
    p_name = meta["power_parameter"]
    if p_name is None:
        raise KeyError(L(f"{variable} is not available in NASA POWER", f"{variable} NASA POWER'da yok"))
    if p_name not in bundle:
        raise DataServiceError(L(f"NASA POWER response has no {p_name}", f"NASA POWER yanıtında {p_name} yok"))
    ice = bundle.get("FRSEAICE", {}) if variable == "TSKINWTR" else None
    out: Dict[Tuple[int, int], float] = {}
    for ym, raw_v in bundle[p_name].items():
        if len(ym) != 6 or not ym.isdigit() or ym.endswith("13"):
            continue
        if raw_v is None or float(raw_v) <= -900:
            continue
        yr, mo = int(ym[:4]), int(ym[4:6])
        if yr < int(meta["satellite_start_year"]):
            continue  # SWGDN: POWER'ın 1984-2000 GEWEX SRB değerleri kullanılmaz (2001 kaynak kırılması)
        val = float(raw_v)
        if variable == "EVLAND":
            val *= pd.Period(f"{yr:04d}-{mo:02d}", freq="M").days_in_month
        if ice is not None:
            ice_v = ice.get(ym)
            if ice_v is None or float(ice_v) <= -900 or float(ice_v) > SEA_ICE_MASK_THRESHOLD or val < OPEN_WATER_MIN_TS:
                continue
        out[(yr, mo)] = val
    return out


# MERRA-2 doğal birimleri (Giovanni dalı için açık dönüşümler)
_MERRA2_NATIVE_UNITS = {
    "T2M": "K", "TSKINWTR": "K",
    "PRECTOTCORR": "kg m-2 s-1", "EVLAND": "kg m-2 s-1", "RUNOFF": "kg m-2 s-1",
    "TQV": "kg m-2", "FRSEAICE": "1", "ALBEDO": "1", "SWGDN": "W m-2",
}


def _convert_merra2_unit(variable: str, raw_val: float, year: int, month: int) -> float:
    """MERRA-2 doğal biriminden uygulama birimine: K -> °C; kg m-2 s-1 -> mm/ay (x 86400 x gün); diğerleri aynen."""
    unit = _MERRA2_NATIVE_UNITS[variable]
    if unit == "K":
        return raw_val - 273.15
    if unit == "kg m-2 s-1":
        return raw_val * 86400.0 * pd.Period(f"{year:04d}-{month:02d}", freq="M").days_in_month
    return raw_val


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_giovanni_series(lat: float, lon: float, variable: str, start_year: int, end_year: int, token: str) -> Dict[Tuple[int, int], float]:
    """
    NASA Giovanni Cloud Time Series API (Earthdata Bearer token). BU DAL TOKEN OLMADAN TEST EDİLMEMİŞTİR:
    yanıt biçimi (CSV düzeni, dolgu değerleri) ve değişken kimlikleri doğrulanamadı. Başarısızlıkta veya
    hiç satır ayrıştırılamazsa DataServiceError fırlatır (önbelleğe alınmaz).
    """
    meta = OFFICIAL_VARIABLES[variable]
    try:
        resp = requests.get(
            GIOVANNI_CLOUD_API_URL,
            params={
                "data": meta["satellite_dataset"],
                "location": f"[{lat:.3f},{lon:.3f}]",
                "time": f"{start_year}-01-01T00:00:00/{end_year}-12-31T23:59:59",
            },
            headers={"Authorization": f"Bearer {token}", "User-Agent": _app_user_agent()},
            timeout=30.0,
        )
    except requests.RequestException as exc:
        raise DataServiceError(L(f"Could not reach Giovanni ({type(exc).__name__})", f"Giovanni'ye ulaşılamadı ({type(exc).__name__})")) from exc
    if resp.status_code != 200:
        raise DataServiceError(f"Giovanni HTTP {resp.status_code}: {resp.text[:200]}", resp.status_code)
    out: Dict[Tuple[int, int], float] = {}
    try:
        for line in resp.text.splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2 and len(parts[0]) >= 7 and parts[0][:4].isdigit() and parts[0][5:7].isdigit():
                yr, mo = int(parts[0][:4]), int(parts[0][5:7])
                raw_val = float(parts[1])
                if not np.isnan(raw_val) and raw_val > -900 and abs(raw_val) < 1e10:
                    out[(yr, mo)] = _convert_merra2_unit(variable, raw_val, yr, mo)
    except ValueError as exc:
        raise DataServiceError(L(f"Could not parse the Giovanni response: {exc}", f"Giovanni yanıtı ayrıştırılamadı: {exc}")) from exc
    if not out:
        raise DataServiceError(L("No parsable monthly values in the Giovanni response", "Giovanni yanıtında ayrıştırılabilir aylık değer yok"))
    return out


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "year", "month", "value", "source_tag"])


def _invalid(status: str, reason: str, df: Optional[pd.DataFrame] = None, coverage: float = 0.0, window: str = "") -> Dict[str, Any]:
    return {
        "valid": False,
        "status": status,
        "coverage_ratio": round(coverage, 3),
        "reason": reason,
        "window": window,
        "df": df if df is not None else _empty_df(),
    }


def fetch_hybrid_climate_series(
    lat: float,
    lon: float,
    variable: str,
    start_year: int = POWER_FIRST_YEAR,
    end_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Değişkenin gerçek aylık serisini döndürür. Tokensız kaynak: NASA POWER. Earthdata token'ı varsa önce
    Giovanni denenir; Giovanni ya değişkenin tüm serisini sağlar ya da hiç kullanılmaz (aylar karıştırılmaz).
    Dönen sözlük: valid, status ('ok' | 'service_error' | 'invalid_request' | 'not_applicable' | 'no_data' | 'low_coverage'),
    coverage_ratio (yalnızca gerçek aylar / pencere ayları), window (değişkenin var olabildiği pencere), reason, df.
    Pencere: max(başlangıç yılı, değişkenin POWER başlangıç yılı)-01 ... min(bitiş yılı-12, son gerçek veri ayı).
    """
    if variable not in OFFICIAL_VARIABLES:
        raise KeyError(L(f"Unknown variable code: {variable!r}", f"Bilinmeyen değişken kodu: {variable!r}"))
    meta = OFFICIAL_VARIABLES[variable]
    end_year = latest_year() if end_year is None else end_year

    try:
        surface = power_surface_class(lat, lon)
    except DataServiceError as exc:
        if exc.status == 422:
            return _invalid("invalid_request", L(
                f"NASA POWER did not accept this coordinate request ({exc}). This is not a service outage; check the coordinate.",
                f"NASA POWER bu koordinat isteğini kabul etmedi ({exc}). Bu bir servis kesintisi değildir; koordinatı kontrol edin.",
            ))
        return _invalid("service_error", L(
            f"Data service error (a service problem, not the location): {exc}. Please try again later.",
            f"Veri servisi hatası (konum değil, servis sorunu): {exc}. Daha sonra tekrar deneyin.",
        ))
    if meta["surface"] != "any" and meta["surface"] != surface:
        need = "yalnızca okyanus/deniz hücrelerinde" if meta["surface"] == "ocean" else "yalnızca (buzsuz) kara hücrelerinde"
        need_en = "only in ocean/sea cells" if meta["surface"] == "ocean" else "only in (ice-free) land cells"
        return _invalid(
            "not_applicable",
            L(
                f"'{meta['short_label']}' is defined {need_en}; according to NASA POWER data this point is in a "
                f"'{SURFACE_NAMES_TR[surface]}' cell (surface mask: land-only parameters, cell elevation and TS/sea-ice "
                f"consistency; see README).",
                f"'{meta['short_label']}' {need} tanımlıdır; bu nokta NASA POWER verisine göre bir "
                f"{SURFACE_NAMES_TR[surface]} hücresi (yüzey maskesi: kara-yalnız parametreler, hücre yüksekliği ve TS/deniz buzu "
                f"tutarlılığı; bkz. README).",
            ),
        )

    token_val = str(st.session_state.get("nasa_earthdata_token", "")).strip()
    token_val = token_val or os.environ.get("NASA_EARTHDATA_TOKEN", "").strip()

    monthly: Optional[Dict[Tuple[int, int], float]] = None
    giovanni_note = ""
    if token_val:
        try:
            monthly = _fetch_giovanni_series(round(float(lat), 3), round(float(lon), 3), variable, int(start_year), int(end_year), token_val)
            source_tag = L(
                f"NASA Giovanni (MERRA-2 {meta['collection']} : {variable}) — untested branch",
                f"NASA Giovanni (MERRA-2 {meta['collection']} : {variable}) — test edilmemiş dal",
            )
            var_start = int(start_year)
        except DataServiceError as exc:
            giovanni_note = L(f" Giovanni attempt failed: {exc}.", f" Giovanni denemesi başarısız: {exc}.")

    if monthly is None:
        if meta["power_parameter"] is None:
            if token_val:
                reason = L(
                    f"'{meta['short_label']}' is not available in NASA POWER and could not be retrieved from Giovanni.{giovanni_note}",
                    f"'{meta['short_label']}' NASA POWER'da yok ve Giovanni'den alınamadı.{giovanni_note}",
                )
                return _invalid("service_error", reason)
            return _invalid(
                "not_applicable",
                L(
                    f"'{meta['short_label']}' is not available in NASA POWER; real MERRA-2 {variable} data requires a "
                    f"NASA Earthdata token to be entered in the left menu (via Giovanni; this branch has not been tested "
                    f"without a token). Without a token, no data is shown for this variable.",
                    f"'{meta['short_label']}' NASA POWER'da bulunmuyor; gerçek MERRA-2 {variable} verisi için sol menüye "
                    f"NASA Earthdata token'ı girilmesi gerekir (Giovanni üzerinden; bu dal token olmadan test edilmemiştir). "
                    f"Token olmadan bu değişken için veri gösterilmez.",
                ),
            )
        try:
            monthly = power_bundle_to_monthly(get_power_point_bundle(lat, lon), variable)
        except DataServiceError as exc:
            return _invalid("service_error", L(
                f"Data service error (a service problem, not the location): {exc}.",
                f"Veri servisi hatası (konum değil, servis sorunu): {exc}.",
            ))
        source_tag = f"NASA POWER {meta['power_parameter']} ({meta['sensors']})"
        var_start = int(meta["satellite_start_year"])

    if not monthly:
        return _invalid("no_data", L(
            f"At this location the source returned no valid monthly values for '{meta['short_label']}'.{giovanni_note}",
            f"Bu konumda kaynak '{meta['short_label']}' için hiç geçerli aylık değer döndürmedi.{giovanni_note}",
        ))

    w_start = (max(int(start_year), var_start), 1)
    w_end = min((int(end_year), 12), max(monthly.keys()))
    if w_end < w_start:
        return _invalid(
            "no_data",
            L(
                f"Data for '{meta['short_label']}' start in {var_start} and the last real month is "
                f"{max(monthly.keys())[0]}-{max(monthly.keys())[1]:02d}; there are no data in the selected {start_year}–{end_year} range.",
                f"'{meta['short_label']}' için veri {var_start} yılında başlar ve son gerçek ay "
                f"{max(monthly.keys())[0]}-{max(monthly.keys())[1]:02d}; seçilen {start_year}–{end_year} aralığında veri yok.",
            ),
        )
    window = f"{w_start[0]}-{w_start[1]:02d} – {w_end[0]}-{w_end[1]:02d}"

    records = []
    no_data_tag = L("No data (fill value at source / masked)", "Veri yok (kaynakta dolgu değeri / maskelendi)")
    n_total = n_valid = 0
    y, m = w_start
    while (y, m) <= w_end:
        n_total += 1
        val = monthly.get((y, m))
        if val is not None:
            n_valid += 1
        records.append(
            {
                "date": pd.Timestamp(year=y, month=m, day=1),
                "year": y,
                "month": m,
                "value": round(float(val), 4) if val is not None else np.nan,
                "source_tag": source_tag if val is not None else no_data_tag,
            }
        )
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    df = pd.DataFrame(records)
    coverage = n_valid / max(1, n_total)
    mask_note = L(" TS is masked in months when the sea-ice fraction exceeds 15%.", " TS, deniz buzu oranı %15'i aşan aylarda maskelenir.") if variable == "TSKINWTR" else ""
    nrt_note = ""
    if meta["power_parameter"] is not None and not giovanni_note and source_tag.startswith("NASA POWER"):
        nrt_note = L(
            " POWER's most recent month(s) come from a low-latency stream (CERES FLASHFlux for radiation, GEOS-IT for "
            "meteorology) and may differ slightly from the climate-quality main source.",
            " POWER'ın en son ay(lar)ı düşük gecikmeli akıştan gelir (ışınımda CERES FLASHFlux, meteorolojide GEOS-IT) "
            "ve iklim kalitesindeki ana kaynaktan biraz farklı olabilir.",
        )
    detail = L(
        f"Source: {source_tag}. Window {window}: {n_valid}/{n_total} months of real observations "
        f"(coverage {coverage * 100:.1f}%).{mask_note}{nrt_note}{giovanni_note}",
        f"Kaynak: {source_tag}. Pencere {window}: {n_valid}/{n_total} ay gerçek gözlem "
        f"(doluluk %{coverage * 100:.1f}).{mask_note}{nrt_note}{giovanni_note}",
    )
    if coverage < 0.70:
        return _invalid("low_coverage", L(
            f"Insufficient data: coverage {coverage * 100:.1f}% < 70% threshold. {detail}",
            f"Yetersiz veri: doluluk %{coverage * 100:.1f} < %70 eşiği. {detail}",
        ), df, coverage, window)
    return {"valid": True, "status": "ok", "coverage_ratio": round(coverage, 3), "reason": detail, "window": window, "df": df}


def build_giovanni_direct_url(
    lat: float,
    lon: float,
    variable: str,
    start_year: int = POWER_FIRST_YEAR,
    end_year: Optional[int] = None,
) -> str:
    """
    Giovanni'de MERRA-2 değişkeni için 1°x1° alan ortalaması zaman serisi bağlantısı (doğrulanmamış).
    Uygulamadaki NASA POWER nokta değerleriyle birebir aynı sonucu vermez.
    """
    var_meta = OFFICIAL_VARIABLES[variable]
    end_year = latest_year() if end_year is None else end_year
    bbox = f"{lon - 0.5:.2f},{lat - 0.5:.2f},{lon + 0.5:.2f},{lat + 0.5:.2f}"
    query = urllib.parse.urlencode(
        {
            "service": "ArAvTs",
            "starttime": f"{start_year}-01-01T00:00:00Z",
            "endtime": f"{end_year}-12-31T23:59:59Z",
            "bbox": bbox,
            "data": var_meta["satellite_dataset"],
        }
    )
    return f"https://giovanni.gsfc.nasa.gov/giovanni/#{query}"
