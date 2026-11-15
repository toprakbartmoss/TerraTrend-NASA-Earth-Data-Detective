"""
NASA SPACE APPS CHALLENGE: "EARTH DATA DETECTIVE"
Çok Sayfalı (Multi-Route) İklim Trendi Analizi, Bölgeler Arası İlişki ve İstatistiksel Uzatma Platformu (app.py)
Veri: NASA POWER aylık nokta API'si (1981+); ayrıntılar data_loader.py ve README.md.

Rotalar (Routes):
- /                : Tanıtım Ekranı (Landing Page), Canlı Harita Önizlemesi ve Bilimsel Metodoloji Özeti
- /explore         : Ana Menü (Tekli / Çiftli / Çoklu Bölge Modları), İnteraktif Vaka Haritası ve 3 Hazır Vaka Merkezi
- /analyze/single  : Tekli Bölge Analizi (Haritada Tek İkon, mevsimsellikten arındırma, Mann-Kendall + Sen, istatistiksel uzatma)
- /analyze/dual    : Çiftli Bölge Karşılaştırması (2 Çizgili Bütünleşik Grafik + Tekil Grafikler + Teleconnection)
- /analyze/multi   : Çoklu Bölge Karşılaştırması (Maksimum 10 Bölge, Korelasyon Isı Haritası, Projeksiyon)
- /parameters      : 10 Parametrelik Sistem Havuzu, Ekip Önem Puanları ve Öğrenci Rehberi
"""

from typing import Any, Dict, List, Optional, Tuple
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from data_loader import (
    DEMO_SCENARIOS,
    OFFICIAL_VARIABLES,
    OSM_ATTRIBUTION,
    POWER_FIRST_YEAR,
    build_giovanni_direct_url,
    fetch_hybrid_climate_series,
    latest_year,
    power_cell,
    reset_service_failure,
    reverse_geocode_and_detect_biome,
)
from statistics_engine import (
    apply_weights_to_composite,
    compare_parameter_across_regions,
    compare_regions_teleconnection,
    compute_objective_weights_and_composite,
    compute_shared_weights,
    decimal_years,
    generate_5year_sarima_forecast,
    hamed_rao_mann_kendall_test,
    stl_decompose_series,
)
from i18n import L, LANG_LABELS, LANGS, default_lang, pick
from visualizer import (
    PARAM_PALETTE,
    plot_dual_region_integrated_composite,
    plot_multi_region_comparison,
    plot_parameter_regions_comparison,
    plot_single_parameter_chart,
    plot_single_region_composite_and_forecast,
    guard_forecast,
    plot_world_period_change_map,
    world_map_caption,
)

LATEST_YEAR = latest_year()  # her çalıştırmada yeniden hesaplanır (yılbaşında eskimez)
reset_service_failure()  # bir önceki çalıştırmadaki servis kesintisi kaydı bu çalıştırmayı etkilemez
MIN_YEAR_SPAN = 5  # İzin verilen en kısa aralık (60 ay); istatistik motoru 10 yıldan (120 ay) kısa serilerde trend kararını 'güvenilir değil' olarak işaretler

# ---------------------------------------------------------------------------
# SAYFA YAPILANDIRMASI VE HARİTA IFRAME GÖRÜNÜRLÜK STİLİ
# ---------------------------------------------------------------------------
if st.session_state.get("lang") not in LANGS:
    st.session_state["lang"] = default_lang()  # English unless EDD_LANG says otherwise

st.set_page_config(
    page_title=L("NASA Earth Data Detective | Climate Trend & Projection Platform", "NASA Earth Data Detective | İklim Trendi & Projeksiyon Platformu"),
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0E1117; }
    iframe[title="streamlit_folium.st_folium"] {
        width: 100% !important;
        min-height: 450px !important;
        height: 450px !important;
        display: block !important;
        border-radius: 10px;
        border: 1px solid #30363D;
    }
    .route-card {
        background: linear-gradient(145deg, #161B22 0%, #0D1117 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .case-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-left: 5px solid #38BDF8;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .sig-badge-ok {
        background-color: rgba(16, 185, 129, 0.16);
        border: 1px solid #10B981;
        color: #A7F3D0;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    .sig-badge-noise {
        background-color: rgba(245, 158, 11, 0.16);
        border: 1px solid #F59E0B;
        color: #FDE68A;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    .tele-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
        border-left: 5px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# OTURUM DURUMU (SESSION STATE) VE ROTA YÖNETİMİ
# ---------------------------------------------------------------------------
VALID_ROUTES = ["/", "/explore", "/analyze/single", "/analyze/dual", "/analyze/multi", "/parameters"]

if "route" not in st.session_state:
    qp_route = st.query_params.get("route", "/analyze/dual")
    st.session_state.route = qp_route if qp_route in VALID_ROUTES else "/analyze/dual"

if "selected_vars" not in st.session_state:
    st.session_state.selected_vars = [
        "T2M",
        "PRECTOTCORR",
        "GWETROOT",
        "EVLAND",
    ]
else:
    valid_existing = [v for v in st.session_state.selected_vars if v in OFFICIAL_VARIABLES]
    st.session_state.selected_vars = valid_existing[:4] if valid_existing else [
        "T2M",
        "PRECTOTCORR",
        "GWETROOT",
        "EVLAND",
    ]

if "chk_version" not in st.session_state:
    st.session_state.chk_version = 0

# Yan panel kontrolleri (yıl aralığı / hibrit / mevsimsellik) ve koordinat kutuları için sürüm sayaçları.
# Değerler programatik olarak değiştiğinde (vaka yükleme, haritaya tıklama) widget anahtarı yenilenir;
# böylece eski widget durumu yeni değeri ezmez (tarih aralığının "geri sıçrama" hatası).
if "ctrl_version" not in st.session_state:
    st.session_state.ctrl_version = 0
if "coord_version" not in st.session_state:
    st.session_state.coord_version = 0

if "year_range" not in st.session_state:
    st.session_state.year_range = (POWER_FIRST_YEAR, LATEST_YEAR)

if "hide_seasonality" not in st.session_state:
    st.session_state.hide_seasonality = True

# 1. Tekli Bölge: Otomatik seçim yok, kullanıcı haritadan veya koordinatla seçer
if "marker_single" not in st.session_state:
    st.session_state.marker_single = None

# 2. Çiftli Bölge: Otomatik seçim yok ([Bölge A, Bölge B]), kullanıcı kendisi seçer
if "markers_dual" not in st.session_state:
    st.session_state.markers_dual = [None, None]

# 3. Çoklu Bölge: Otomatik seçim yok (0..10 Bölge), kullanıcı kendisi ekler ve dilediğini silebilir
if "markers_multi" not in st.session_state:
    st.session_state.markers_multi = []

if "active_case_title" not in st.session_state:
    st.session_state.active_case_title = None


COORD_LAT_LIMIT = 85.0  # koordinat kutularının sınırı (Leaflet/Web Mercator ±85.05)


def norm_coord(lat: float, lon: float) -> Tuple[float, float]:
    """Haritadan/oturumdan gelen koordinatı widget sınırlarına getirir: boylam [-180, 180), enlem ±85 (2 ondalık)."""
    lon_n = round(((float(lon) + 180.0) % 360.0) - 180.0, 2)
    lon_n = -180.0 if lon_n >= 180.0 else lon_n
    lat_n = round(min(max(float(lat), -COORD_LAT_LIMIT), COORD_LAT_LIMIT), 2)
    return lat_n, lon_n


def norm_marker(mk: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if mk is None:
        return None
    lat_n, lon_n = norm_coord(mk["lat"], mk["lon"])
    return {"lat": lat_n, "lon": lon_n}


def _clicked(map_out: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """st_folium tıklaması (dünya kopyasında boylam ±180 dışına taşabilir) -> normalize (enlem, boylam)."""
    if map_out and map_out.get("last_clicked"):
        return norm_coord(map_out["last_clicked"]["lat"], map_out["last_clicked"]["lng"])
    return None


# Oturumdaki işaretçiler her çalıştırmada normalize edilir (koordinat kutuları sınır dışı değerle çökmesin).
st.session_state.marker_single = norm_marker(st.session_state.marker_single)
st.session_state.markers_dual = [norm_marker(m) for m in st.session_state.markers_dual]
st.session_state.markers_multi = [norm_marker(m) for m in st.session_state.markers_multi]


def _create_base_folium_map(center_lat: float, center_lon: float, zoom: int = 3) -> folium.Map:
    """
    Her ağda ve tarayıcıda kesintisiz açılan OpenStreetMap ana katmanlı ve
    Esri World Imagery (Uydu) yedek katmanlı Folium haritası oluşturur.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name=L("Satellite View (Esri)", "Uydu Görünümü (Esri)"),
        show=False,
    ).add_to(m)
    folium.LatLngPopup().add_to(m)
    return m


def navigate_to(target_route: str) -> None:
    """Uygulama içi alt rotayı günceller ve URL query parametresiyle senkronize eder."""
    if target_route in VALID_ROUTES:
        st.session_state.route = target_route
        st.query_params["route"] = target_route


def load_case_study(case_id: str, use_tuz_golu: bool = False) -> None:
    """3 Hazır Vaka Çalışmasından birini ilgili rotaya ve parametrelere yükler."""
    sc = DEMO_SCENARIOS[case_id]
    st.session_state.selected_vars = list(sc["variables"][:4])
    st.session_state.chk_version += 1
    st.session_state.ctrl_version += 1
    st.session_state.coord_version += 1
    st.session_state.year_range = (int(sc["start_year"]), int(sc["end_year"] or LATEST_YEAR))
    st.session_state.hide_seasonality = bool(sc["hide_seasonality"])
    st.session_state.active_case_title = DEMO_SCENARIOS.raw[case_id]["title"]  # {"en","tr"} pair; resolved when shown

    if sc["mode"] == "single":
        if case_id == "aral_tuz" and use_tuz_golu:
            m = sc["alt_marker_tuz"]
        else:
            m = sc["markers"][0]
        st.session_state.marker_single = {"lat": float(m["lat"]), "lon": float(m["lon"])}
        navigate_to("/analyze/single")
    elif sc["mode"] == "dual":
        m0, m1 = sc["markers"][0], sc["markers"][1]
        st.session_state.markers_dual = [
            {"lat": float(m0["lat"]), "lon": float(m0["lon"])},
            {"lat": float(m1["lat"]), "lon": float(m1["lon"])},
        ]
        navigate_to("/analyze/dual")


def _on_param_checkbox_change(var_code: str, key_name: str) -> None:
    """
    10 parametrelik havuzda maksimum 4 seçim kısıtını yönetir.
    Kullanıcı en az 1, en fazla 4 parametre seçebilir.
    """
    checked = bool(st.session_state.get(key_name, False))
    current = list(st.session_state.selected_vars)
    if checked and var_code not in current:
        if len(current) < 4:
            current.append(var_code)
    elif not checked and var_code in current:
        if len(current) > 1:
            current.remove(var_code)
        else:
            st.session_state[key_name] = True
            return
    current.sort(key=lambda x: OFFICIAL_VARIABLES[x]["rank"])
    st.session_state.selected_vars = current


def apply_recommended_biome_vars(rec_vars: List[str]) -> None:
    """Seçilen koordinatın biyomuna uygun 4 parametreyi otomatik seçer."""
    valid_recs = [v for v in rec_vars if v in OFFICIAL_VARIABLES][:4]
    if valid_recs:
        valid_recs.sort(key=lambda x: OFFICIAL_VARIABLES[x]["rank"])
        st.session_state.selected_vars = valid_recs
        st.session_state.chk_version += 1


# ---------------------------------------------------------------------------
# ÜST GEZİNME ÇUBUĞU (MULTI-ROUTE NAVIGATION BAR)
# ---------------------------------------------------------------------------
def render_top_navbar() -> None:
    st.markdown(
        "### 🛰️ NASA Space Apps Challenge — **Earth Data Detective** "
        + L(
            "<span style='font-size:14px;color:#38BDF8;font-weight:normal;'>| A Platform for Climate Trends, Inter-Regional Relationships and Statistical Extension with NASA POWER Data</span>",
            "<span style='font-size:14px;color:#38BDF8;font-weight:normal;'>| NASA POWER Verisiyle İklim Trendi, Bölgeler Arası İlişki ve İstatistiksel Uzatma Platformu</span>",
        ),
        unsafe_allow_html=True,
    )
    nav_cols = st.columns([1, 1, 1, 1, 1, 1, 0.75])
    nav_cols[6].selectbox(
        "Language / Dil",
        options=list(LANGS),
        format_func=lambda c: f"🌐 {LANG_LABELS[c]}",
        key="lang",
        label_visibility="collapsed",
    )
    routes_info = [
        ("/", L("🏠 Home (/)", "🏠 Tanıtım (/)")),
        ("/explore", L("🧭 Explore & Cases (/explore)", "🧭 Keşif & Vakalar (/explore)")),
        ("/analyze/single", L("📍 Single Region (/single)", "📍 Tekli Bölge (/single)")),
        ("/analyze/dual", L("⚡ Two Regions (/dual)", "⚡ Çiftli Bölge (/dual)")),
        ("/analyze/multi", L("🌐 Multiple Regions (/multi)", "🌐 Çoklu Bölge (/multi)")),
        ("/parameters", L("📚 10 Parameters (/parameters)", "📚 10 Parametre (/parameters)")),
    ]
    for col, (r_path, r_label) in zip(nav_cols, routes_info):
        is_active = st.session_state.route == r_path
        if col.button(
            r_label,
            key=f"nav_btn_{r_path}",
            type="primary" if is_active else "secondary",
            width="stretch",
        ):
            navigate_to(r_path)
            st.rerun()
    st.markdown("---")


# ---------------------------------------------------------------------------
# ROTA 1: `/` — TANITIM EKRANI (LANDING PAGE + CANLI KÜRESEL HARİTA)
# ---------------------------------------------------------------------------
def render_landing_route() -> None:
    st.markdown(
        L(
        f"""
        <div class="route-card">
            <h2 style="margin-top:0;color:#38BDF8;">🌍 NASA Earth Data Detective: Separate Global Climate Signals from Noise</h2>
            <p style="font-size:16px;line-height:1.6;color:#E2E8F0;">
                As part of the <b>NASA Space Apps Challenge</b>, this platform pulls monthly Earth system data from the
                <b>NASA POWER</b> monthly point API (meteorology derived from the MERRA-2 reanalysis; CERES-derived
                radiation), deseasonalizes the changes <b>from {POWER_FIRST_YEAR} to today</b>,
                tests the statistical significance of the trend, examines relationships between regions (correlation and slope difference)
                and draws a 60-month <b>statistical extension</b> (not a climate projection) from the end of the selected period.
            </p>
        </div>
        """,
        f"""
        <div class="route-card">
            <h2 style="margin-top:0;color:#38BDF8;">🌍 NASA Earth Data Detective: Küresel İklim Sinyallerini Gürültüden Ayırın</h2>
            <p style="font-size:16px;line-height:1.6;color:#E2E8F0;">
                Bu platform, <b>NASA Space Apps Challenge</b> kapsamında aylık Dünya sistemi verilerini
                <b>NASA POWER</b> aylık nokta API'sinden (MERRA-2 yeniden analizi türevli meteoroloji; CERES türevli
                ışınım) çekerek <b>{POWER_FIRST_YEAR}'den bugüne</b> kadarki değişimleri mevsimsellikten arındırır,
                trendin istatistiksel anlamlılığını test eder, bölgeler arasındaki ilişkileri (korelasyon ve eğim farkı)
                inceler ve seçilen dönemin sonundan itibaren 60 aylık <b>istatistiksel bir uzatma</b> (iklim projeksiyonu değil) çizer.
            </p>
        </div>
        """,
        ),
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button(L("🚀 Start Exploring (/explore)", "🚀 Keşfetmeye Başla (/explore)"), key="land_btn_explore", type="primary", width="stretch"):
            navigate_to("/explore")
            st.rerun()
    with btn_col2:
        if st.button(L("📍 Single Region Map (/analyze/single)", "📍 Tekli Bölge Haritası (/analyze/single)"), key="land_btn_single", width="stretch"):
            navigate_to("/analyze/single")
            st.rerun()
    with btn_col3:
        if st.button(L("⚡ Two-Region Map (/analyze/dual)", "⚡ Çiftli Bölge Haritası (/analyze/dual)"), key="land_btn_dual", width="stretch"):
            navigate_to("/analyze/dual")
            st.rerun()
    with btn_col4:
        if st.button(L("📚 10-Parameter Guide (/parameters)", "📚 10 Parametre Rehberi (/parameters)"), key="land_btn_params", width="stretch"):
            navigate_to("/parameters")
            st.rerun()

    st.markdown(L("#### 🗺️ Location Selection Map (Click a Point on the Map to Go Straight to Single-Region Analysis)", "#### 🗺️ Konum Seçim Haritası (Haritadan Bir Noktaya Tıklayarak Doğrudan Tekli Analize Geçebilirsiniz)"))
    m_land = _create_base_folium_map(25.0, 20.0, zoom=2)
    # 3 Hazır Vaka noktasını önizleme olarak göster (sabit vaka adları; sayfa açılışında ağ çağrısı yapılmaz)
    preview_pts = [
        (DEMO_SCENARIOS["aral_tuz"]["markers"][0], "red", L("Combination 1 (Drought)", "Kombinasyon 1 (Kuraklık)")),
        (DEMO_SCENARIOS["aral_tuz"]["alt_marker_tuz"], "orange", L("Combination 1b (Drought)", "Kombinasyon 1b (Kuraklık)")),
        (DEMO_SCENARIOS["water_cycle"]["markers"][0], "green", L("Combination 2 (Water Cycle)", "Kombinasyon 2 (Su Döngüsü)")),
        (DEMO_SCENARIOS["water_cycle"]["markers"][1], "blue", L("Combination 2 (Water Cycle)", "Kombinasyon 2 (Su Döngüsü)")),
        (DEMO_SCENARIOS["arctic_albedo"]["markers"][0], "purple", L("Combination 3 (Polar/Energy)", "Kombinasyon 3 (Kutup/Enerji)")),
    ]
    for pt, pcol, ptitle in preview_pts:
        plat, plon = float(pt["lat"]), float(pt["lon"])
        folium.Marker(
            location=[plat, plon],
            tooltip=folium.Tooltip(f"<b>{ptitle}</b><br>{pick(pt['name'])}<br>{plat:.2f}°, {plon:.2f}°", sticky=True),
            icon=folium.Icon(color=pcol, icon="info-sign"),
        ).add_to(m_land)
    folium.LayerControl().add_to(m_land)

    land_click = st_folium(
        m_land,
        use_container_width=True,
        height=430,
        key="nasa_map_landing",
        returned_objects=["last_clicked"],
    )
    if _clicked(land_click):
        c_lat, c_lon = _clicked(land_click)
        st.session_state.marker_single = {"lat": c_lat, "lon": c_lon}
        st.session_state.coord_version += 1
        navigate_to("/analyze/single")
        st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            L(
                """
            <div class="route-card">
                <h4>📡 1. Data Source: NASA POWER Monthly Point API (1981+)</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>MERRA-2 derived (POWER):</b> <code>T2M</code>, <code>TQV</code>, <code>PRECTOTCORR</code> (monthly total), <code>GWETROOT</code>, <code>EVLAND</code>, <code>FRSEAICE</code>, <code>TS</code> (only in ice-free ocean cells, as <code>TSKINWTR</code>)</li>
                    <li><b>CERES SYN1deg (2001+):</b> <code>SWGDN</code>, <code>ALBEDO</code> — only from 2001 onward (POWER's 1984–2000 SRB radiation is not used because of a break between sources)</li>
                    <li>POWER's most recent months come from low-latency streams (GEOS-IT for meteorology, CERES FLASHFlux for radiation).</li>
                    <li><b>RUNOFF:</b> not in POWER; only via Giovanni with an Earthdata token (untested)</li>
                    <li>Missing months are not filled in; service errors are reported openly.</li>
                </ul>
            </div>
            """,
                """
            <div class="route-card">
                <h4>📡 1. Veri Kaynağı: NASA POWER Aylık Nokta API'si (1981+)</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>MERRA-2 türevli (POWER):</b> <code>T2M</code>, <code>TQV</code>, <code>PRECTOTCORR</code> (aylık toplam), <code>GWETROOT</code>, <code>EVLAND</code>, <code>FRSEAICE</code>, <code>TS</code> (yalnızca buzsuz okyanus hücresinde <code>TSKINWTR</code> olarak)</li>
                    <li><b>CERES SYN1deg (2001+):</b> <code>SWGDN</code>, <code>ALBEDO</code> — yalnızca 2001'den itibaren (POWER'ın 1984–2000 SRB ışınımı, kaynak kırılması nedeniyle kullanılmaz)</li>
                    <li>POWER'ın en son ayları düşük gecikmeli akışlardan gelir (meteorolojide GEOS-IT, ışınımda CERES FLASHFlux).</li>
                    <li><b>RUNOFF:</b> POWER'da yok; yalnızca Earthdata token'ı ile Giovanni üzerinden (test edilmemiş)</li>
                    <li>Eksik aylar doldurulmaz; servis hataları açıkça bildirilir.</li>
                </ul>
            </div>
            """,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            L(
                """
            <div class="route-card">
                <h4>📐 2. Weighting (CRITIC) and Limits of Use</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>10 Variables (team score 10.0 → 8.3):</b> The scores are the team's own choice; they are not based on any external source.</li>
                    <li><b>Maximum of 4 Selections:</b> Once 4 parameters are selected, the remaining 6 are locked automatically.</li>
                    <li><b>Weight:</b> Computed from the data itself with the CRITIC method (σ × Σ(1 − |r|), normalized to sum to 1); the team score does not enter the weight.</li>
                    <li><b>Limits of Use & VIF > 5:</b> NASA POWER land/ocean mask; VIF only gives a diagnostic warning.</li>
                </ul>
            </div>
            """,
                """
            <div class="route-card">
                <h4>📐 2. Ağırlıklandırma (CRITIC) ve Kullanım Sınırları</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>10 Değişken (ekip puanı 10.0 → 8.3):</b> Puanlar ekibin kendi seçimidir; harici bir kaynağa dayanmaz.</li>
                    <li><b>Maksimum 4 Seçim Kısıtı:</b> 4 parametre seçildiğinde kalan 6 parametre otomatik kilitlenir.</li>
                    <li><b>Ağırlık:</b> CRITIC yöntemiyle verinin kendisinden (σ × Σ(1 − |r|), toplam 1'e normalize); ekip puanı ağırlığa katılmaz.</li>
                    <li><b>Kullanım Sınırları & VIF > 5:</b> NASA POWER kara/okyanus maskesi; VIF yalnızca tanı amaçlı uyarı verir.</li>
                </ul>
            </div>
            """,
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            L(
                """
            <div class="route-card">
                <h4>📈 3. Deseasonalizing, Mann-Kendall + Sen & Statistical Extension</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>Deseasonalizing:</b> The long-term mean of each calendar month (monthly climatology) is subtracted; missing months are not filled in.</li>
                    <li><b>Mann-Kendall + Theil-Sen:</b> $p$-value corrected for autocorrelation and long memory (the largest of the AR(1), local Whittle and EWC estimates), Kendall $\\tau_b$ and a 95% CI for the Sen slope using the same variance; for series shorter than 10 years the result is marked "not reliable".</li>
                    <li><b>Inter-Regional Relationship:</b> Spearman ρ (p corrected for effective sample size) and $\\Delta\\text{Slope}$ on deseasonalized and detrended series; this is a measure of association, not proof of causation.</li>
                    <li><b>60-Month Statistical Extension:</b> Linear trend + AR(1) (fallback: mean of the last 5 years); its skill is reported through backtesting; it is not a climate projection.</li>
                </ul>
            </div>
            """,
                """
            <div class="route-card">
                <h4>📈 3. Arındırma, Mann-Kendall + Sen & İstatistiksel Uzatma</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>Mevsimsellikten Arındırma:</b> Her takvim ayının uzun dönem ortalaması (aylık klimatoloji) çıkarılır; eksik aylar doldurulmaz.</li>
                    <li><b>Mann-Kendall + Theil-Sen:</b> otokorelasyon ve uzun hafıza düzeltmeli $p$-değeri (AR(1), yerel Whittle, EWC tahminlerinin en büyüğü), Kendall $\\tau_b$ ve aynı varyansla Sen eğimi %95 GA; 10 yıldan kısa serilerde karar "güvenilir değil" işaretlenir.</li>
                    <li><b>Bölgeler Arası İlişki:</b> Arındırılmış ve trendi çıkarılmış serilerde Spearman ρ (etkin örneklem düzeltmeli p) ve $\\Delta\\text{Slope}$; ilişki ölçüsüdür, nedensellik kanıtı değildir.</li>
                    <li><b>60 Aylık İstatistiksel Uzatma:</b> Doğrusal trend + AR(1) (yedek: son 5 yıl ortalaması); becerisi geriye dönük sınamayla raporlanır; iklim projeksiyonu değildir.</li>
                </ul>
            </div>
            """,
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# ROTA 2: `/explore` — ANA MENÜ VE 3 HAZIR VAKA ÇALIŞMASI MERKEZİ
# ---------------------------------------------------------------------------
def render_explore_route() -> None:
    st.subheader(L("🧭 Analysis Mode Selection and Case Center (/explore)", "🧭 Analiz Modu Seçimi ve Vaka Merkezi (/explore)"))
    st.caption(
        L(
            "Choose one of the 3 analysis modes below depending on the geographic scope you want to study, "
            "start one of the 3 Ready-Made Case Studies, or pick a point directly on the map.",
            "İncelemek istediğiniz coğrafi kapsama göre aşağıdaki 3 analiz modundan birini seçin, "
            "3 Hazır Vaka Çalışmasından birini başlatın veya haritadan doğrudan nokta seçin.",
        )
    )

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            L(
                """
            <div class="route-card">
                <h4>📍 1. Single Region Selection</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Select <b>a single point/icon</b> on the map to examine the deseasonalized series from 1981 to today,
                    the Mann-Kendall + Sen trend test and the 60-month statistical extension.
                </p>
            </div>
            """,
                """
            <div class="route-card">
                <h4>📍 1. Tekli Bölge Seçimi</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Harita üzerinde <b>tek bir nokta/ikon</b> seçerek 1981'den bugüne mevsimsellikten arındırılmış seriyi,
                    Mann-Kendall + Sen trend testini ve 60 aylık istatistiksel uzatmayı inceleyin.
                </p>
            </div>
            """,
            ),
            unsafe_allow_html=True,
        )
        if st.button(L("📍 Go to Single-Region Analysis (/analyze/single)", "📍 Tekli Bölge Analizine Git (/analyze/single)"), key="explore_btn_single", type="primary", width="stretch"):
            navigate_to("/analyze/single")
            st.rerun()

    with m_col2:
        st.markdown(
            L(
                """
            <div class="route-card">
                <h4>⚡ 2. Two-Region Selection (Relationship / Contrast)</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Select <b>Region A and Region B</b> on the map to examine the integrated two-line comparison chart,
                    the cross-correlation and the $\\Delta\\text{Slope}$ slope difference (measures of association; not proof of causation).
                </p>
            </div>
            """,
                """
            <div class="route-card">
                <h4>⚡ 2. Çiftli Bölge Seçimi (İlişki / Zıtlık)</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Haritada <b>Bölge A ve Bölge B</b>'yi seçerek iki çizgili bütünleşik karşılaştırma grafiğini,
                    çapraz korelasyonu ve $\\Delta\\text{Slope}$ eğim farkını inceleyin (ilişki ölçüleri; nedensellik kanıtı değildir).
                </p>
            </div>
            """,
            ),
            unsafe_allow_html=True,
        )
        if st.button(L("⚡ Go to Two-Region Analysis (/analyze/dual)", "⚡ Çiftli Bölge Analizine Git (/analyze/dual)"), key="explore_btn_dual", type="primary", width="stretch"):
            navigate_to("/analyze/dual")
            st.rerun()

    with m_col3:
        st.markdown(
            L(
                """
            <div class="route-card">
                <h4>🌐 3. Multi-Region Selection (Max. 10 Regions)</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Mark points in <b>up to 10 regions</b> on the world map to produce a multi-region trend comparison across continents,
                    an $N \\times N$ correlation heat map and a 60-month statistical extension.
                </p>
            </div>
            """,
                """
            <div class="route-card">
                <h4>🌐 3. Çoklu Bölge Seçimi (Maks. 10 Bölge)</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Dünya haritası üzerinde <b>10 bölgeye kadar</b> nokta işaretleyerek kıtalar arası çoklu trend
                    karşılaştırması, $N \\times N$ korelasyon ısı haritası ve 60 aylık istatistiksel uzatma üretin.
                </p>
            </div>
            """,
            ),
            unsafe_allow_html=True,
        )
        if st.button(L("🌐 Go to Multi-Region Analysis (/analyze/multi)", "🌐 Çoklu Bölge Analizine Git (/analyze/multi)"), key="explore_btn_multi", type="primary", width="stretch"):
            navigate_to("/analyze/multi")
            st.rerun()

    st.markdown("---")
    st.subheader(L(f"🧪 3 Ready-Made Case Studies (Demonstration Scenarios — {POWER_FIRST_YEAR}–{LATEST_YEAR})", f"🧪 3 Hazır Vaka Çalışması (Demonstration Scenarios — {POWER_FIRST_YEAR}–{LATEST_YEAR})"))

    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        sc1 = DEMO_SCENARIOS["aral_tuz"]
        st.markdown(
            f"""
            <div class="case-card">
                <h5>{pick(sc1['title'])}</h5>
                <p style="font-size:13.5px;color:#CBD5E1;">{pick(sc1['summary'])}</p>
                <p style="font-size:12px;color:#38BDF8;"><b>{L("Parameters (Drought Combination):", "Parametreler (Kuraklık Kombinasyonu):")}</b> <code>T2M</code> + <code>PRECTOTCORR</code> + <code>GWETROOT</code> + <code>EVLAND</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1a, b1b = st.columns(2)
        if b1a.button(L("🏜️ Load the Aral Sea Case", "🏜️ Aral Gölü Vakasını Yükle"), key="case_aral", width="stretch"):
            load_case_study("aral_tuz", use_tuz_golu=False)
            st.rerun()
        if b1b.button(L("🇹🇷 Load the Lake Tuz (Tuz Gölü) Case", "🇹🇷 Tuz Gölü Vakasını Yükle"), key="case_tuz", width="stretch"):
            load_case_study("aral_tuz", use_tuz_golu=True)
            st.rerun()

    with sc_col2:
        sc2 = DEMO_SCENARIOS["water_cycle"]
        st.markdown(
            f"""
            <div class="case-card">
                <h5>{pick(sc2['title'])}</h5>
                <p style="font-size:13.5px;color:#CBD5E1;">{pick(sc2['summary'])}</p>
                <p style="font-size:12px;color:#38BDF8;"><b>{L("Parameters (Water Cycle Combination):", "Parametreler (Su Döngüsü Kombinasyonu):")}</b> <code>PRECTOTCORR</code> + <code>GWETROOT</code> + <code>EVLAND</code> + <code>TQV</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(L("💧 Load the Water Cycle Combination (/dual)", "💧 Su Döngüsü Kombinasyonunu Yükle (/dual)"), key="case_water_cycle", width="stretch"):
            load_case_study("water_cycle")
            st.rerun()

    with sc_col3:
        sc3 = DEMO_SCENARIOS["arctic_albedo"]
        st.markdown(
            f"""
            <div class="case-card">
                <h5>{pick(sc3['title'])}</h5>
                <p style="font-size:13.5px;color:#CBD5E1;">{pick(sc3['summary'])}</p>
                <p style="font-size:12px;color:#38BDF8;"><b>{L("Parameters (Polar / Energy Combination, 2001+):", "Parametreler (Kutup / Enerji Kombinasyonu, 2001+):")}</b> <code>T2M</code> + <code>FRSEAICE</code> + <code>ALBEDO</code> + <code>SWGDN</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(L("❄️ Load the Arctic Albedo Case (/single)", "❄️ Arktik Albedo Vakasını Yükle (/single)"), key="case_arctic", width="stretch"):
            load_case_study("arctic_albedo")
            st.rerun()

    st.markdown("---")
    p_col1, p_col2 = st.columns([2.5, 1.5])
    with p_col1:
        st.markdown(L("#### 🔬 10 Variables, Data Sources and Weighting Methodology", "#### 🔬 10 Değişken, Veri Kaynakları ve Ağırlıklandırma Metodolojisi"))
        st.caption(
            L(
                "Go to the guide page to explore the 10 variables (`T2M`, `PRECTOTCORR`, `TSKINWTR`, `GWETROOT`, `EVLAND`, `RUNOFF`, `TQV`, `FRSEAICE`, "
                "`ALBEDO`, `SWGDN`), their actual sources and start years, the team's importance scores (10.0 → 8.3) "
                "and their limits of use.",
                "10 değişkeni (`T2M`, `PRECTOTCORR`, `TSKINWTR`, `GWETROOT`, `EVLAND`, `RUNOFF`, `TQV`, `FRSEAICE`, "
                "`ALBEDO`, `SWGDN`), gerçek kaynaklarını ve başlangıç yıllarını, ekibin önem puanlarını (10.0 → 8.3) "
                "ve kullanım sınırlarını incelemek için rehber sayfasına geçebilirsiniz.",
            )
        )
    with p_col2:
        if st.button(L("📚 Open the Parameter and Importance Ranking Guide (/parameters)", "📚 Parametre ve Önem Sırası Rehberini Aç (/parameters)"), key="explore_btn_params", width="stretch"):
            navigate_to("/parameters")
            st.rerun()


# ---------------------------------------------------------------------------
# ROTA 6: `/parameters` — 10 PARAMETRELİK SİSTEM HAVUZU VE ÖĞRENCİ REHBERİ
# ---------------------------------------------------------------------------
def render_parameters_route() -> None:
    st.subheader(L("📚 10 Variables, Their Real Sources and Guide (/parameters)", "📚 10 Değişken, Gerçek Kaynakları ve Rehber (/parameters)"))
    st.markdown(L(
        f"""
        > **Data source:** Without a token, all values come from the **NASA POWER monthly point API** (data starts in {POWER_FIRST_YEAR}-01).
        > Meteorological parameters are values POWER derives from the **NASA GMAO MERRA-2** reanalysis; POWER does not state which MERRA-2
        > collection it uses, so no collection name is given. `SWGDN` and `ALBEDO` are derived from CERES SYN1deg and
        > are used only from 2001-01 onward (POWER's 1984–2000 SWGDN values come from GEWEX SRB; the 2001 source
        > change creates an artificial jump, so they are discarded). POWER's most recent month(s) come from low-latency streams
        > (GEOS-IT for meteorology, CERES FLASHFlux for radiation). `RUNOFF` is not available in POWER. Missing months are not filled. Values
        > are those of the POWER grid cell the point falls in (MERRA-2 0.5°×0.625°, CERES 1°), not of the point itself.
        >
        > **Weighting:** The weights of the 1–4 selected parameters are computed from the data itself with the CRITIC method
        > (C = σ × Σ(1 − |r|), on deseasonalized series; total `1.00`). VIF > 5 is a diagnostic
        > warning only. Importance scores (`10.0` → `8.3`) are **the team's own choice**, not taken from an external source, and are not
        > used in the weights (ranking/information only). The exact method is shown in the weights table on the analysis page.
        """,
        f"""
        > **Veri kaynağı:** Tokensız tüm değerler **NASA POWER aylık nokta API'sinden** gelir (veri {POWER_FIRST_YEAR}-01'de başlar).
        > Meteorolojik parametreler POWER'ın **NASA GMAO MERRA-2** yeniden analizinden türettiği değerlerdir; POWER hangi MERRA-2
        > koleksiyonunu kullandığını belirtmediği için koleksiyon adı verilmez. `SWGDN` ve `ALBEDO` CERES SYN1deg türevlidir ve
        > yalnızca 2001-01'den itibaren kullanılır (POWER'ın 1984–2000 SWGDN değerleri GEWEX SRB'den gelir; 2001'deki kaynak
        > değişimi yapay bir sıçrama yarattığı için atılır). POWER'ın en son ay(lar)ı düşük gecikmeli akışlardan gelir
        > (meteorolojide GEOS-IT, ışınımda CERES FLASHFlux). `RUNOFF` POWER'da yoktur. Eksik aylar doldurulmaz. Değerler
        > noktanın değil, düştüğü POWER hücresinin (MERRA-2 0.5°×0.625°, CERES 1°) değeridir.
        >
        > **Ağırlıklandırma:** Seçilen 1–4 parametrenin ağırlıkları CRITIC yöntemiyle verinin kendisinden hesaplanır
        > (C = σ × Σ(1 − |r|), mevsimsellikten arındırılmış seriler üzerinde; toplam `1.00`). VIF > 5 yalnızca tanı amaçlı
        > uyarıdır. Önem puanları (`10.0` → `8.3`) **ekibin kendi seçimidir**, harici bir kaynaktan alınmamıştır ve ağırlığa
        > katılmaz (yalnızca sıralama/bilgi). Kesin yöntem analiz sayfasındaki ağırlık tablosunda gösterilir.
        """
    ))

    table_rows = []
    for code, meta in sorted(OFFICIAL_VARIABLES.items(), key=lambda x: x[1]["rank"]):
        table_rows.append(
            {
                L("Rank", "Sıra"): f"#{meta['rank']}",
                L("Field Code", "Alan Kodu"): meta.get("short_name", code),
                L("Parameter", "Parametre"): meta["short_label"],
                L("Source", "Kaynak"): meta["platform"],
                L("Underlying Product / Collection", "Alttaki Ürün / Koleksiyon"): meta["collection"],
                L("Data Start", "Veri Başlangıcı"): str(meta["satellite_start_year"]) if meta["power_parameter"] else L("Token only", "Yalnızca token ile"),
                L("Team Importance Score / 10 (not used in weights)", "Ekip Önem Puanı / 10 (ağırlıkta kullanılmaz)"): f"{meta.get('importance_score_10', meta['base_weight'] * 10):.1f} / 10",
                L("Unit", "Birim"): meta["unit"],
                L("Valid Surface", "Geçerli Yüzey"): {"any": L("Land + Ocean", "Kara + Okyanus"), "land": L("Land only", "Yalnızca kara"), "ocean": L("Ocean only", "Yalnızca okyanus")}[meta["surface"]],
            }
        )
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    st.info(L(
        "💡 **Team-suggested 4-variable combinations (for the max. 4 selections rule; the team's own choice):**\n"
        "- **Drought Analysis (land):** `T2M` + `PRECTOTCORR` + `GWETROOT` + `EVLAND`\n"
        "- **Polar / Energy Analysis (polar ocean, 2001+):** `T2M` + `FRSEAICE` + `ALBEDO` + `SWGDN`\n"
        "- **Water Cycle Analysis (land):** `PRECTOTCORR` + `GWETROOT` + `EVLAND` + `TQV`",
        "💡 **Ekibin önerdiği 4'lü kombinasyonlar (Maks. 4 Seçim Kuralı İçin; ekibin kendi seçimidir):**\n"
        "- **Kuraklık Analizi (kara):** `T2M` + `PRECTOTCORR` + `GWETROOT` + `EVLAND`\n"
        "- **Kutup / Enerji Analizi (kutup okyanusu, 2001+):** `T2M` + `FRSEAICE` + `ALBEDO` + `SWGDN`\n"
        "- **Su Döngüsü Analizi (kara):** `PRECTOTCORR` + `GWETROOT` + `EVLAND` + `TQV`",
    ))

    st.markdown(L("### 🎓 Variable Descriptions, Why They Matter and Limits of Use", "### 🎓 Değişken Açıklamaları, Neden Kritik Oldukları ve Kullanım Sınırları"))
    for code, meta in sorted(OFFICIAL_VARIABLES.items(), key=lambda x: x[1]["rank"]):
        score_10 = meta.get("importance_score_10", meta["base_weight"] * 10)
        coll = meta["collection"]
        with st.expander(
            L(f"{meta['label']} — Team score: {score_10:.1f}/10 (not used in weights) | {coll}", f"{meta['label']} — Ekip puanı: {score_10:.1f}/10 (ağırlıkta kullanılmaz) | {coll}"),
            expanded=(meta["rank"] <= 3),
        ):
            st.markdown(L(f"**Why It Matters and How to Use It** {meta['student_explanation']}", f"**Neden Kritik ve Nasıl Kullanılır?** {meta['student_explanation']}"))
            coll_link = L(f" — [GES DISC collection page]({meta['collection_url']})", f" — [GES DISC koleksiyon sayfası]({meta['collection_url']})") if meta.get("collection_url") else ""
            st.markdown(L(
                f"- **Field Code (`Short Name`):** `{code}`\n"
                f"- **Actual data source:** [{meta['platform']}]({meta['platform_url']}) — POWER parameter: "
                f"`{meta['power_parameter'] or 'none'}` (data start: {meta['satellite_start_year'] if meta['power_parameter'] else 'token only'})\n"
                f"- **Underlying product / collection:** `{coll}`{coll_link}\n"
                f"- **Giovanni variable ID (token only, untested):** `{meta['satellite_dataset']}`\n"
                f"- **Relationships the team expects (unverified assumption):** `{', '.join(meta['correlated_with'])}`",
                f"- **Alan Kodu (`Short Name`):** `{code}`\n"
                f"- **Gerçek veri kaynağı:** [{meta['platform']}]({meta['platform_url']}) — POWER parametresi: "
                f"`{meta['power_parameter'] or 'yok'}` (veri başlangıcı: {meta['satellite_start_year'] if meta['power_parameter'] else 'yalnızca token ile'})\n"
                f"- **Alttaki ürün / koleksiyon:** `{coll}`{coll_link}\n"
                f"- **Giovanni değişken kimliği (yalnızca token ile, test edilmedi):** `{meta['satellite_dataset']}`\n"
                f"- **Ekibin beklediği ilişkiler (doğrulanmamış varsayım):** `{', '.join(meta['correlated_with'])}`",
            ))

    st.markdown("---")
    st.markdown(L("### 📐 Guide to Statistical Metrics and Trend Analysis Indicators", "### 📐 İstatistiksel Metrikler ve Trend Analizi Göstergeleri Rehberi"))
    st.markdown(L(
        "Below we explain the scientific meaning and mathematical definition of the non-parametric "
        "trend indicators shown in the **Statistics Detail** cards on the analysis pages, and how to interpret their increase/decrease levels:",
        "Analiz sayfalarındaki **İstatistik Detayı** kartlarında yer alan parametrik olmayan (non-parametrik) "
        "trend göstergelerinin bilimsel anlamları, matematiksel tanımları ve yükselme/azalma seviyelerinin nasıl yorumlanacağı aşağıda açıklanmıştır:",
    ))

    _hg = L("Statistical Indicator", "İstatistiksel Gösterge")
    _hm = L("What Does It Measure?", "Ne Ölçer?")
    _hc = L("Numerical Interpretation Criterion", "Sayısal Yorum Kriteri")
    stat_summary_rows = [
        {
            _hg: L("Sen's Slope (Annual / Decadal Rate)", "Sen's Slope (Yıllık / 10 Yıllık Hız)"),
            _hm: L("Median physical rate and direction of change per unit time", "Birim zamandaki ortanca (medyan) fiziksel değişim hızı ve yönü"),
            _hc: L("> 0: Increase | < 0: Decrease | ≈ 0: Stable (robust to outliers)", "> 0: Yükseliş | < 0: Azalış | ≈ 0: Durağan (Aykırı uç değerlere dirençlidir)"),
        },
        {
            _hg: L("Kendall's Tau (τ) and S Statistic", "Kendall's Tau (τ) ve S İstatistiği"),
            _hm: L("Consistency of the monotonic increase or decrease in the time series (between -1 and +1)", "Zaman serisindeki monotonik artış veya azalışın tutarlılığı (-1 ile +1 arası)"),
            _hc: L("|τ| ≥ 0.45: Very strong | 0.25–0.45: Clear | 0.12–0.25: Mild–moderate | < 0.12: Weak/Fluctuating", "|τ| ≥ 0.45: Çok güçlü | 0.25–0.45: Belirgin | 0.12–0.25: Hafif-orta | < 0.12: Zayıf/Dalgalı"),
        },
        {
            _hg: L("95% Confidence Interval", "%95 Güven Aralığı (Confidence Interval)"),
            _hm: L("Lower and upper bounds that contain the true annual slope with 95% probability", "Gerçek yıllık değişim eğiminin %95 olasılıkla bulunduğu alt ve üst sınır"),
            _hc: L("If the interval excludes 0, it is evidence for the slope's direction (not certainty); it is computed from the same corrected variance as p, and 'significant' is declared only if p < 0.05 AND the interval excludes 0", "Aralık 0'ı içermiyorsa eğim yönü için kanıttır (kesinlik değil); p ile aynı düzeltilmiş varyanstan hesaplanır ve 'anlamlı' kararı yalnızca p < 0.05 VE aralık 0'ı dışlıyorsa verilir"),
        },
        {
            _hg: L("Mann-Kendall Z and p-Value (autocorrelation / long-memory corrected)", "Mann-Kendall Z ve p-Değeri (otokorelasyon / uzun hafıza düzeltmeli)"),
            _hm: L("Probability of observing a result at least this extreme if there were no trend at all", "Hiç trend olmasaydı, en az bu kadar uç bir sonucun gözlenme olasılığı"),
            _hc: L("p < 0.05: Inconsistent with the no-trend assumption (significant) | p ≥ 0.05: The no-trend assumption cannot be rejected", "p < 0.05: Trend yok varsayımıyla uyumsuz (anlamlı) | p ≥ 0.05: Trend yok varsayımı reddedilemez"),
        },
        {
            _hg: L("Variance Factor and Effective Sample Size (n_eff)", "Varyans Çarpanı ve Etkin Örneklem (n_eff)"),
            _hm: L("How much Var(S) is inflated because of memory between months (short- and long-term autocorrelation)", "Aylar arasındaki hafıza (kısa ve uzun dönem otokorelasyon) nedeniyle Var(S)'nin büyütülme oranı"),
            _hc: L("If > 1.00, the variance is inflated, the effective sample size shrinks and the p-value increases; for series shorter than 120 months the verdict is flagged as unreliable", "> 1.00 ise varyans büyütülür, etkin örneklem küçülür ve p-değeri artar; 120 aydan kısa serilerde karar güvenilir değil olarak işaretlenir"),
        },
        {
            _hg: L("Spearman ρ and ΔSlope (Between Regions)", "Spearman ρ ve ΔSlope (Bölgeler Arası)"),
            _hm: L("Simultaneous relationship between deseasonalized and linearly detrended series, and difference in annual rate", "Mevsimselliği ve doğrusal trendi çıkarılmış seriler arasındaki eşzamanlı ilişki ve yıllık hız farkı"),
            _hc: L("p < 0.05 (Bretherton effective sample size) and ρ < 0: Opposite | ρ > 0: Same direction | p ≥ 0.05: no significant relationship; at least 60 common months required (a relationship; not proof of causation)", "p < 0.05 (Bretherton etkin örneklemli) ve ρ < 0: Ters yönlü | ρ > 0: Eş yönlü | p ≥ 0.05: anlamlı ilişki yok; en az 60 ortak ay gerekir (ilişkidir; nedensellik kanıtı değildir)"),
        },
    ]
    st.dataframe(pd.DataFrame(stat_summary_rows), width="stretch", hide_index=True)

    with st.expander(L("1️⃣ Sen's Slope (Theil–Sen Slope Estimator — Annual and Decadal Rate of Change)", "1️⃣ Sen's Slope (Theil–Sen Eğim Tahmincisi — Yıllık ve 10 Yıllık Değişim Hızı)"), expanded=True):
        st.markdown(L(
            "- **Definition:** The annual rate of change computed as the **median** of the slopes $\\frac{x_j - x_i}{t_j - t_i}$ between all data pairs $(t_i, t_j)$ in the time series.\n"
            "- **Why Not Ordinary Linear Regression?** In climate data, a single extremely hot/dry year (e.g., a strong El Niño year) can artificially skew the classic ordinary least squares (OLS) slope. **Sen's Slope** is not affected by outliers (robust).\n"
            "- **How to Interpret It?**\n"
            "  - **Positive Value (`+`):** Shows that the variable rises from year to year (e.g., `+0.035 °C/yr` or `+0.35 °C/decade` warming).\n"
            "  - **Negative Value (`-`):** Shows that the variable decreases from year to year (e.g., `-0.004 /yr` for root-zone soil moisture: a decrease of 0.004 units per year on the 0–1 wetness scale).\n"
            "  - **Decadal Rate (`/decade`):** In climate science (in IPCC reports), the rate of change is read on a decadal scale; it is the annual slope multiplied by 10.",
            "- **Tanım:** Zaman serisindeki tüm veri çiftleri $(t_i, t_j)$ arasındaki eğimlerin $\\frac{x_j - x_i}{t_j - t_i}$ **medyanı (ortancası)** alınarak hesaplanan yıllık değişim hızıdır.\n"
            "- **Neden Sıradan Doğrusal Regresyon Değil?** İklim verilerinde tek bir aşırı sıcak/kurak yıl (ör. şiddetli El Niño yılı) klasik en küçük kareler (OLS) eğimini yapay olarak saptırabilir. **Sen's Slope** aykırı değerlerden etkilenmez (robust).\n"
            "- **Nasıl Yorumlanır?**\n"
            "  - **Pozitif Değer (`+`):** Değişkenin yıldan yıla yükseldiğini gösterir (ör. `+0.035 °C/yıl` veya `+0.35 °C / 10 yıl` ısınma).\n"
            "  - **Negatif Değer (`-`):** Değişkenin yıldan yıla azaldığını gösterir (ör. kök bölgesi toprak nemi için `-0.004 /yıl`: 0–1 ıslaklık oranında yılda 0.004 birim azalma).\n"
            "  - **10 Yıllık Hız (`/10 yıl`):** İklim biliminde (IPCC raporlarında) değişim hızı on yıllık (dekadal) ölçekte okunur; yıllık eğimin 10 ile çarpımıdır.",
        ))

    with st.expander(L("2️⃣ Kendall's Tau (τ) Coefficient and Mann–Kendall S Statistic", "2️⃣ Kendall's Tau (τ) Katsayısı ve Mann–Kendall S İstatistiği"), expanded=True):
        st.markdown(L(
            "- **Mann–Kendall `S` Statistic:** The net direction score obtained by subtracting the number of data pairs that decrease over time from the number that increase ($S > 0$ increase, $S < 0$ decrease).\n"
            "- **Kendall's Tau ($\\tau$):** The standard strength coefficient between `-1.00` and `+1.00`, obtained by dividing the `S` statistic by the total number of pairs.\n"
            "- **Increase / Decrease Level Table:**\n"
            "  - $|\\tau| \\ge 0.45$: **Very Strong and Steady** increase or decrease.\n"
            "  - $0.25 \\le |\\tau| < 0.45$: **Clear (Moderate–Strong)** increase or decrease.\n"
            "  - $0.12 \\le |\\tau| < 0.25$: **Mild–Moderate** gradual tendency.\n"
            "  - $|\\tau| < 0.12$: **Very Weak / Stable** (no clear tendency).",
            "- **Mann–Kendall `S` İstatistiği:** Zaman ilerledikçe artan veri çiftlerinin sayısından azalan veri çiftlerinin sayısının çıkarılmasıyla elde edilen net yön skorudur ($S > 0$ artış, $S < 0$ azalış).\n"
            "- **Kendall's Tau ($\\tau$):** `S` istatistiğinin toplam çift sayısına bölünmesiyle elde edilen `-1.00` ile `+1.00` arasındaki standart güç katsayısıdır.\n"
            "- **Yükselme / Azalma Seviyesi Tablosu:**\n"
            "  - $|\\tau| \\ge 0.45$: **Çok Güçlü ve İstikrarlı** yükselme veya azalma.\n"
            "  - $0.25 \\le |\\tau| < 0.45$: **Belirgin (Orta–Güçlü)** yükselme veya azalma.\n"
            "  - $0.12 \\le |\\tau| < 0.25$: **Hafif–Orta Düzeyde** kademeli eğilim.\n"
            "  - $|\\tau| < 0.12$: **Çok Zayıf / Durağan** (belirgin bir yönelim yok).",
        ))

    with st.expander(L("3️⃣ 95% Confidence Interval (Lower and Upper Slope Bounds)", "3️⃣ %95 Güven Aralığı (Confidence Interval — Alt ve Üst Eğim Sınırları)"), expanded=True):
        st.markdown(L(
            "- **Definition:** The `[Lower Bound, Upper Bound]` interval that contains the annual Sen's Slope rate of change **with 95% probability**.\n"
            "- **How to Interpret It?**\n"
            "  - **If both bounds are positive** (e.g., `[+0.021, +0.048]`): The data support an increasing direction (evidence; not certainty).\n"
            "  - **If both bounds are negative** (e.g., `[-0.018, -0.005]`): The data support a decreasing direction.\n"
            "  - **If the interval includes zero (`0`)** (e.g., `[-0.008, +0.014]`): The direction of the slope is uncertain.\n"
            "  - The interval and the p-value are computed from the same corrected variance. When there are many equal (tied) values (e.g., months with zero precipitation or zero sea ice), the interval may touch 0 while p < 0.05; in that case the verdict is cautiously 'not significant' and the reason is written on the card. This way the verdict and the interval never contradict each other.",
            "- **Tanım:** Sen's Slope yıllık değişim hızının **%95 olasılıkla** içinde bulunduğu `[Alt Sınır, Üst Sınır]` aralığıdır.\n"
            "- **Nasıl Yorumlanır?**\n"
            "  - **Her iki sınır da pozitifse** (ör. `[+0.021, +0.048]`): Veri yükseliş yönünü destekler (kanıt; kesinlik değil).\n"
            "  - **Her iki sınır da negatifse** (ör. `[-0.018, -0.005]`): Veri azalış yönünü destekler.\n"
            "  - **Aralık sıfırı (`0`) kapsıyorsa** (ör. `[-0.008, +0.014]`): Eğim yönü belirsizdir.\n"
            "  - Aralık ile p-değeri aynı düzeltilmiş varyanstan hesaplanır. Çok sayıda eşit (bağlı) değer olduğunda (ör. sıfır yağışlı veya sıfır deniz buzlu aylar) p < 0.05 iken aralık 0'a değebilir; bu durumda karar temkinli olarak 'anlamlı değil' verilir ve kartta nedeni yazılır. Böylece karar ile aralık hiçbir zaman çelişmez.",
        ))

    with st.expander(L("4️⃣ Mann-Kendall Z, p-Value (Significance) and Autocorrelation / Long-Memory Correction", "4️⃣ Mann-Kendall Z, p-Değeri (Anlamlılık) ve Otokorelasyon / Uzun Hafıza Düzeltmesi"), expanded=True):
        st.markdown(L(
            "- **p-Value (`p-value`):** The probability of observing a test result at least this extreme if there were actually no trend in the series. It is **not** the 'probability that the trend is due to chance' or the probability that the trend is real.\n"
            "  - `p < 0.05`: The data are considered inconsistent with the 'no trend' assumption (`Statistically significant`).\n"
            "  - `p ≥ 0.05`: The 'no trend' assumption cannot be rejected; this does not prove there is no trend (a longer series may be needed).\n"
            "- **Autocorrelation and long-memory correction:** In monthly climate series, this month's value is not independent of previous months; some series (soil moisture, precipitation, some temperature series) also contain slow fluctuations lasting for years. Because the standard Mann-Kendall test ignores this memory, it more often wrongly concludes 'there is a significant trend'. The engine computes three standard estimates from the residuals of the deseasonalized series and inflates Var(S) by the LARGEST of them: (1) lag-1 AR(1) effective sample size (Yue & Wang 2004), (2) long memory: local Whittle d estimate and ARFIMA(0,d,0) (Robinson 1995), (3) low-frequency variance: equal-weighted cosine (EWC; Lazarus, Lewis, Stock & Watson 2018). The p-value and the 95% confidence interval of the Sen slope use the same variance, taken from a t distribution with B = 0.4·T^(2/3) degrees of freedom (`variance factor`, `effective n`; the card states which estimate is binding). In Monte Carlo validation (scripts/mc_validate_stats.py; 1000 repetitions, 5% threshold), the false alarm rate with no trend was ≤ 4% for AR(1) noise, ≤ 8% for AR(2), and 7–13% for highly persistent long-memory ARFIMA (d = 0.3–0.4) noise (28–54% with the old AR(1)-only correction); the cost is lower power to detect weak trends in 10–20-year series (e.g., +0.03 sd/yr over 20 years: 50% → 33%). The test still assumes stationary noise: breaks such as a source change or a regime shift can look like a trend. For series shorter than 120 months (10 years), the verdict is flagged as 'unreliable'.",
            "- **p-Değeri (`p-value`):** Seride gerçekte hiç trend olmasaydı, en az bu kadar uç bir test sonucunun gözlenme olasılığıdır. Trendin 'şans eseri olma olasılığı' veya trendin gerçek olma olasılığı **değildir**.\n"
            "  - `p < 0.05`: Veri 'trend yok' varsayımıyla uyumsuz kabul edilir (`İstatistiksel olarak anlamlı`).\n"
            "  - `p ≥ 0.05`: 'Trend yok' varsayımı reddedilemez; bu, trendin olmadığını kanıtlamaz (daha uzun seri gerekebilir).\n"
            "- **Otokorelasyon ve uzun hafıza düzeltmesi:** Aylık iklim serilerinde bu ayın değeri geçen aylardan bağımsız değildir; bazı seriler (toprak nemi, yağış, bazı sıcaklık serileri) yıllar boyunca süren yavaş dalgalanmalar da içerir. Standart Mann-Kendall testi bu hafızayı yok saydığı için yanlışlıkla 'anlamlı trend var' sonucuna daha sık varır. Motor, arındırılmış serinin artıklarından üç standart tahmin hesaplar ve Var(S)'yi bunların EN BÜYÜĞÜYLE büyütür: (1) gecikme-1 AR(1) etkin örneklem (Yue & Wang 2004), (2) uzun hafıza: yerel Whittle d tahmini ve ARFIMA(0,d,0) (Robinson 1995), (3) düşük frekans varyansı: eşit ağırlıklı kosinüs (EWC; Lazarus, Lewis, Stock & Watson 2018). p-değeri ve Sen eğiminin %95 güven aralığı aynı varyansla, B = 0.4·T^(2/3) serbestlik dereceli t dağılımından alınır (`varyans çarpanı`, `etkin n`; kartta hangi tahminin bağlayıcı olduğu yazılır). Monte Carlo doğrulamasında (scripts/mc_validate_stats.py; 1000 tekrar, %5 eşik) trend yokken yanlış alarm oranı AR(1) gürültüsünde ≤ %4, AR(2)'de ≤ %8, çok kalıcı uzun hafızalı ARFIMA (d = 0.3–0.4) gürültüsünde %7–13 oldu (eski yalnızca-AR(1) düzeltmesinde %28–54); bedeli, 10–20 yıllık serilerde zayıf trendleri yakalama gücünün düşmesidir (ör. 20 yılda +0.03 sd/yıl: %50 → %33). Test yine de durağan gürültü varsayar: kaynak değişimi veya rejim kayması gibi kırılmalar trend gibi görünebilir. 120 aydan (10 yıl) kısa serilerde karar 'güvenilir değil' olarak işaretlenir.",
        ))

    with st.expander(L("5️⃣ Deseasonalization, Spearman ρ, ΔSlope and 60-Month Statistical Extension", "5️⃣ Mevsimsellikten Arındırma, Spearman ρ, ΔSlope ve 60 Aylık İstatistiksel Uzatma"), expanded=False):
        st.markdown(L(
            "- **Deseasonalization (monthly climatology):** The series is placed on a complete monthly calendar axis and the long-term mean of each calendar month is subtracted; missing months are not filled. The 'Trend (13-month mean)' line in the charts is the centered 13-month moving average of the deseasonalized series and is for visualization only. All trend tests (including the composite index) are run on the deseasonalized series; 'Hide seasonal cycle' only changes the view.\n"
            "- **Spearman Rank Correlation (`ρ`) and `ΔSlope`:** Measures whether two regional series, with seasonality and linear trend removed, fluctuate together (`ρ > 0`) or in opposite directions (`ρ < 0`); the p-value uses the Bretherton et al. (1999) effective sample size to account for autocorrelation, and the class becomes 'Opposite/Same Direction' only if p < 0.05. The best lag is descriptive only. `ΔSlope = Slope A - Slope B` is the difference in annual trend rates. These are measures of association; they are not proof of teleconnection or causation.\n"
            "- **60-Month Statistical Extension and 95% Band:** After the last observation, the series is extended for 60 months with a linear trend + stationary AR(1) model (+ seasonal harmonics in the raw view); the shaded band is the model's own 95% interval. The forecast and band are clipped at the variable's physical limits (ratios and albedo 0–1, precipitation and radiation ≥ 0, open-water temperature ≥ −2 °C); where clipped, the band is no longer a symmetric 95% interval. If the model does not converge or goes outside a reasonable envelope of past data, a fallback model (mean of the last 5 years) is used; if that is also invalid, no forecast is shown and the reason is given. Calendar months never observed in the past (e.g., polar-night albedo) are left empty. Skill is reported by comparing against the last-5-year mean in a backtest over the last 60 months. This is not a climate projection; if real observations exist after the chosen end year, the forecast is not shown.",
            "- **Mevsimsellikten arındırma (aylık klimatoloji):** Seri eksiksiz aylık takvim eksenine yerleştirilir ve her takvim ayının uzun dönem ortalaması çıkarılır; eksik aylar doldurulmaz. Grafiklerdeki 'Trend (13 aylık ortalama)' çizgisi arındırılmış serinin merkezli 13 aylık hareketli ortalamasıdır ve yalnızca görselleştirme içindir. Tüm trend testleri (bütünleşik indeks dahil) arındırılmış seri üzerinde yapılır; 'Mevsimsel döngüyü gizle' yalnızca görünümü değiştirir.\n"
            "- **Spearman Sıra Korelasyonu (`ρ`) ve `ΔSlope`:** Mevsimselliği ve doğrusal trendi çıkarılmış iki bölge serisinin birlikte mi (`ρ > 0`) yoksa zıt yönde mi (`ρ < 0`) dalgalandığını ölçer; p-değeri otokorelasyon için Bretherton vd. (1999) etkin örneklemiyle hesaplanır ve sınıf yalnızca p < 0.05 ise 'Ters/Eş Yönlü' olur. En iyi gecikme yalnızca betimseldir. `ΔSlope = Eğim A - Eğim B` yıllık trend hızı farkıdır. Bunlar ilişki ölçüleridir; uzaktan bağlantı (teleconnection) veya nedensellik kanıtı değildir.\n"
            "- **60 Aylık İstatistiksel Uzatma ve %95 Bandı:** Son gözlemden sonra 60 ay boyunca doğrusal trend + durağan AR(1) modeli (ham görünümde + mevsimsel harmonikler) ile uzatılır; gölgeli bant modelin kendi %95 aralığıdır. Tahmin ve bant değişkenin fiziksel sınırlarında (oranlar ve albedo 0–1, yağış ve ışınım ≥ 0, açık su sıcaklığı ≥ −2 °C) kırpılır; kırpılan yerde bant artık simetrik %95 aralığı değildir. Model yakınsamaz veya geçmiş verinin makul zarfının dışına çıkarsa yedek model (son 5 yılın ortalaması) kullanılır; o da geçersizse tahmin gösterilmez ve gerekçesi yazılır. Geçmişte hiç gözlenmemiş takvim ayları (ör. kutup gecesi albedosu) için tahmin boş kalır. Beceri, son 60 ayın geriye dönük sınamasında son 5 yıl ortalamasıyla karşılaştırılarak raporlanır. Bu bir iklim projeksiyonu değildir; seçilen bitiş yılından sonra gerçek gözlem varsa tahmin gösterilmez.",
        ))


# ---------------------------------------------------------------------------
# ORTAK YAN PANEL & KONTROL MERKEZİ (ANALİZ ROTALARI İÇİN)
# ---------------------------------------------------------------------------
def render_analysis_controls(primary_geo: Optional[Dict[str, Any]] = None) -> None:
    st.sidebar.markdown(L("## ⚙️ Analysis & Data Architecture Controls", "## ⚙️ Analiz & Veri Mimarisi Kontrolleri"))

    st.sidebar.markdown(L("### 📅 1. Time Range & Seasonality", "### 📅 1. Zaman Aralığı & Mevsimsellik"))
    cv = st.session_state.ctrl_version

    def _sync_ctrl(src_key: str, dst: str) -> None:
        val = st.session_state[src_key]
        st.session_state[dst] = (int(val[0]), int(val[1])) if dst == "year_range" else bool(val)

    yr_key = f"yr_slider_{cv}"
    st.sidebar.slider(
        L(f"Historical Analysis Range ({POWER_FIRST_YEAR}–{LATEST_YEAR}):", f"Tarihsel Analiz Aralığı ({POWER_FIRST_YEAR}–{LATEST_YEAR}):"),
        min_value=POWER_FIRST_YEAR,
        max_value=LATEST_YEAR,
        value=tuple(st.session_state.year_range),
        step=1,
        key=yr_key,
        on_change=_sync_ctrl,
        args=(yr_key, "year_range"),
        help=L(
            f"A range of at least {MIN_YEAR_SPAN} years is required; for series shorter than 10 years the trend verdict is flagged as 'unreliable'. NASA POWER data start in {POWER_FIRST_YEAR}-01; "
            f"{LATEST_YEAR} is partial, with published months only. SWGDN and ALBEDO start in 2001: each "
            f"variable's actual data window and coverage are shown on the analysis page.",
            f"En az {MIN_YEAR_SPAN} yıllık aralık gereklidir; 10 yıldan kısa serilerde trend kararı 'güvenilir değil' olarak işaretlenir. NASA POWER verisi {POWER_FIRST_YEAR}-01'de "
            f"başlar; {LATEST_YEAR} yılı yalnızca yayımlanmış aylarla kısmidir. SWGDN ve ALBEDO 2001'de başlar: her "
            f"değişkenin gerçek veri penceresi ve doluluk oranı analiz sayfasında gösterilir.",
        ),
    )
    yr0, yr1 = st.session_state.year_range
    st.sidebar.caption(
        L(f"Selected period: **{yr0}–{yr1}** (at most {yr1 - yr0 + 1} calendar years; the actual number of months depends on the variable). ",
          f"Seçili dönem: **{yr0}–{yr1}** (en fazla {yr1 - yr0 + 1} takvim yılı; gerçek ay sayısı değişkene göre değişir). ")
        + (
            L("The 60-month statistical extension starts after the last observed month.", "60 aylık istatistiksel uzatma son gözlenen aydan sonra başlar.")
            if yr1 >= LATEST_YEAR
            else L(f"Real observations exist after the end year ({yr1}), so the extension (forecast) is not shown.", f"Bitiş yılından ({yr1}) sonra gerçek gözlemler bulunduğu için uzatma (tahmin) gösterilmez.")
        )
    )

    se_key = f"season_toggle_{cv}"
    st.sidebar.toggle(
        L("📉 Hide Seasonal Cycle (Deseasonalized View)", "📉 Mevsimsel Döngüyü Gizle (Arındırılmış Görünüm)"),
        value=st.session_state.hide_seasonality,
        key=se_key,
        on_change=_sync_ctrl,
        args=(se_key, "hide_seasonality"),
        help=L(
            "When on, the charts show the series with the long-term mean of each calendar month removed (deseasonalized); "
            "when off, the raw monthly observations are shown. Trend tests are run on the deseasonalized series in both cases.",
            "Açıkken grafikler, her takvim ayının uzun dönem ortalaması çıkarılmış (mevsimsellikten arındırılmış) "
            "seriyi gösterir; kapalıyken ham aylık gözlemler gösterilir. Trend testleri her iki durumda da arındırılmış seri üzerinde yapılır.",
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 2. NASA Earthdata Login (GES DISC)")
    st.sidebar.text_input(
        L("Earthdata Bearer Token (Optional / GES DISC):", "Earthdata Bearer Token (Opsiyonel / GES DISC):"),
        value=st.session_state.get("nasa_earthdata_token", ""),
        type="password",
        key="nasa_earthdata_token",
        help=L(
            "Optional. If a token is entered, each variable is first requested from the NASA Giovanni Cloud API (MERRA-2); if Giovanni "
            "provides the variable's full series it is used, otherwise NASA POWER is used (months are not mixed). "
            "RUNOFF can only be obtained this way. WARNING: This branch was developed without a token and has not been tested; the response format has not been verified.",
            "İsteğe bağlı. Token girilirse her değişken önce NASA Giovanni Cloud API'den (MERRA-2) istenir; Giovanni "
            "değişkenin serisini tamamen sağlarsa o kullanılır, başarısız olursa NASA POWER kullanılır (aylar karıştırılmaz). "
            "RUNOFF yalnızca bu yolla alınabilir. UYARI: Bu dal token olmadan geliştirildi ve test edilmedi; yanıt biçimi doğrulanmadı.",
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(L("### 🧪 3. Load a Quick Example Combination", "### 🧪 3. Hızlı Örnek Kombinasyon Yükle"))
    if st.sidebar.button(L("1. 🏜️ Drought: Aral Sea (/single)", "1. 🏜️ Kuraklık: Aral Gölü (/single)"), width="stretch", key="case_btn_aral"):
        load_case_study("aral_tuz", use_tuz_golu=False)
        st.rerun()
    if st.sidebar.button(L("1b. 🇹🇷 Drought: Lake Tuz (Tuz Gölü) (/single)", "1b. 🇹🇷 Kuraklık: Tuz Gölü (/single)"), width="stretch", key="case_btn_tuz"):
        load_case_study("aral_tuz", use_tuz_golu=True)
        st.rerun()
    if st.sidebar.button(L("2. 💧 Water Cycle Combination (/dual)", "2. 💧 Su Döngüsü Kombinasyonu (/dual)"), width="stretch", key="case_btn_water"):
        load_case_study("water_cycle")
        st.rerun()
    if st.sidebar.button(L("3. ❄️ Polar / Energy Combination (/single)", "3. ❄️ Kutup / Enerji Kombinasyonu (/single)"), width="stretch", key="case_btn_arctic"):
        load_case_study("arctic_albedo")
        st.rerun()

    st.markdown(L("### 🧬 Variable Recommender by Surface Class and 10-Parameter System Pool (Maximum 4 Selections)", "### 🧬 Yüzey Sınıfına Göre Değişken Önerici ve 10 Parametrelik Sistem Havuzu (Maksimum 4 Seçim)"))

    if primary_geo is not None:
        rec_vars = primary_geo["recommended_vars"]
        rec_labels = [OFFICIAL_VARIABLES[v]["short_label"] for v in rec_vars if v in OFFICIAL_VARIABLES]

        b_col1, b_col2 = st.columns([3.0, 1.3])
        with b_col1:
            st.info(
                L(f"🧭 **Surface Class (NASA POWER land/ocean mask + polar latitude):** `{primary_geo['biome_name']}` "
                  f"({primary_geo['display_label']}; name: {primary_geo['name_source']})\n\n"
                  f"💡 **Variables the Team Recommends for This Class:** ",
                  f"🧭 **Yüzey Sınıfı (NASA POWER kara/okyanus maskesi + kutup enlemi):** `{primary_geo['biome_name']}` "
                  f"({primary_geo['display_label']}; ad: {primary_geo['name_source']})\n\n"
                  f"💡 **Bu Sınıf İçin Ekibin Önerdiği Değişkenler:** ") + (" • ".join([f"**{lbl}**" for lbl in rec_labels]) or "—") +
                f"\n\n*{primary_geo['biome_reason']}*"
            )
        with b_col2:
            st.write("")
            if st.button(
                L("✨ Auto-Select the 4 Recommended Parameters", "✨ Önerilen 4 Parametreyi Otomatik Seç"),
                type="primary",
                width="stretch",
                key="apply_recommended_vars_btn",
            ):
                apply_recommended_biome_vars(rec_vars)
                st.rerun()
    else:
        rec_vars = []
        st.info(L(
            "🧭 **Variable Recommender:** When you select a region on the map, the point's surface class is determined from NASA POWER data "
            "(land / ocean cell; polar if |latitude| ≥ 66.5°) and the 4 variables the team recommends are shown here. "
            "No more detailed biome (desert, forest, city) is determined.",
            "🧭 **Değişken Önerici:** Haritadan bir bölge seçtiğinizde noktanın yüzey sınıfı NASA POWER verisinden "
            "(kara / okyanus hücresi; |enlem| ≥ 66.5° ise kutup) belirlenir ve ekibin önerdiği 4 değişken burada gösterilir. "
            "Daha ayrıntılı bir biyom (çöl, orman, şehir) belirlenmez.",
        ))

    preset_c1, preset_c2, preset_c3 = st.columns(3)
    if preset_c1.button(L("🌵 Drought Set (T2M + PRECTOTCORR + GWETROOT + EVLAND)", "🌵 Kuraklık Seti (T2M + PRECTOTCORR + GWETROOT + EVLAND)"), width="stretch", key="preset_btn_drought"):
        apply_recommended_biome_vars(["T2M", "PRECTOTCORR", "GWETROOT", "EVLAND"])
        st.rerun()
    if preset_c2.button(L("❄️ Polar / Energy Set (T2M + FRSEAICE + ALBEDO + SWGDN; 2001+)", "❄️ Kutup / Enerji Seti (T2M + FRSEAICE + ALBEDO + SWGDN; 2001+)"), width="stretch", key="preset_btn_polar"):
        apply_recommended_biome_vars(["T2M", "FRSEAICE", "ALBEDO", "SWGDN"])
        st.rerun()
    if preset_c3.button(L("💧 Water Cycle Set (PRECTOTCORR + GWETROOT + EVLAND + TQV)", "💧 Su Döngüsü Seti (PRECTOTCORR + GWETROOT + EVLAND + TQV)"), width="stretch", key="preset_btn_water"):
        apply_recommended_biome_vars(["PRECTOTCORR", "GWETROOT", "EVLAND", "TQV"])
        st.rerun()

    selected_count = len(st.session_state.selected_vars)
    st.caption(L(
        f"**Selected Parameters: `{selected_count} / 4`** — "
        f"{'🔒 Maximum of 4 parameters selected; the remaining 6 parameters were automatically disabled.' if selected_count >= 4 else 'You can select up to 4 parameters from the list.'} "
        f"*(Weights are computed automatically from the data itself with the CRITIC method; team importance scores are the team's own choice and are for ranking only).* ",
        f"**Seçili Parametre Sayısı: `{selected_count} / 4`** — "
        f"{'🔒 Maksimum 4 parametre seçildi; kalan 6 parametre otomatik olarak devre dışı (disabled) bırakıldı.' if selected_count >= 4 else 'Listeden en fazla 4 parametre seçebilirsiniz.'} "
        f"*(Ağırlıklar verinin kendisinden CRITIC yöntemiyle otomatik hesaplanır; ekip önem puanları ekibin kendi seçimidir ve yalnızca sıralama içindir).* ",
    ))

    chk_cols = st.columns(2)
    sorted_vars = sorted(OFFICIAL_VARIABLES.items(), key=lambda x: x[1]["rank"])
    for idx, (var_code, meta) in enumerate(sorted_vars):
        is_selected = var_code in st.session_state.selected_vars
        is_disabled = (not is_selected) and (selected_count >= 4)
        is_rec_badge = L(" ⭐ [Recommended]", " ⭐ [Öneri]") if var_code in rec_vars else ""
        chk_key = f"chk_{var_code}_{st.session_state.chk_version}"
        score_10 = meta.get("importance_score_10", meta["base_weight"] * 10)
        coll = meta.get("collection", "")

        with chk_cols[idx % 2]:
            st.checkbox(
                L(f"{meta['label']} — Team score: {score_10:.1f}/10{is_rec_badge}", f"{meta['label']} — Ekip puanı: {score_10:.1f}/10{is_rec_badge}"),
                value=is_selected,
                disabled=is_disabled,
                key=chk_key,
                on_change=_on_param_checkbox_change,
                args=(var_code, chk_key),
                help=L(f"{meta['student_explanation']} | Source: {meta['platform']} — {coll}", f"{meta['student_explanation']} | Kaynak: {meta['platform']} — {coll}"),
            )


FORECAST_NOTE = {
    "en": (
        "📎 **Forecast note:** The dashed lines and bands are a univariate statistical extension (the model is named in "
        "the legend); they contain no physics or scenarios and are not a climate projection. The note under each chart reports whether, in a "
        "backtest over the last 60 months, the model did better than the 'mean of the last 5 years' (skill). If the main model goes outside a "
        "reasonable envelope of past data, a fallback model is used; if that is also invalid, no forecast is shown."
    ),
    "tr": (
    "📎 **Tahmin notu:** Kesikli çizgiler ve bantlar tek değişkenli istatistiksel bir uzatmadır (modeli lejantta "
    "yazılıdır); fizik veya senaryo içermez, iklim projeksiyonu değildir. Her grafiğin altındaki not, son 60 ayın geriye "
    "dönük sınamasında modelin 'son 5 yılın ortalaması'ndan daha iyi olup olmadığını (beceri) bildirir. Ana model geçmiş "
    "verinin makul zarfı dışına çıkarsa yedek model kullanılır; o da geçersizse tahmin gösterilmez."
    ),
}


def _record_fetch_status(label: str, res: Dict[str, Any]) -> None:
    """Bu çalıştırmadaki gerçek veri çekim sonuçlarını kenar çubuğu durum paneli için kaydeder."""
    st.session_state.setdefault("_fetch_log", []).append((label, res.get("status", "?"), res.get("reason", "")))


def render_fetch_status_sidebar() -> None:
    """Statik yeşil ışıklar yerine bu sayfa çalıştırmasındaki gerçek veri çekim sonuçları."""
    log = st.session_state.get("_fetch_log", [])
    with st.sidebar.expander(L("🛰️ Data Fetch Status (this page)", "🛰️ Veri Çekim Durumu (bu sayfa)"), expanded=any(s == "service_error" for _, s, _ in log)):
        if not log:
            st.caption(L("No data has been fetched in this run yet.", "Bu çalıştırmada henüz veri çekilmedi."))
            return
        names = {"ok": L("🟢 Data received", "🟢 Veri alındı"), "service_error": L("🔴 Service error", "🔴 Servis hatası"), "invalid_request": L("🟠 Coordinate rejected", "🟠 Koordinat reddedildi"),
                 "not_applicable": L("⚪ Undefined on this surface", "⚪ Bu yüzeyde tanımsız"),
                 "no_data": L("⚪ No data", "⚪ Veri yok"), "low_coverage": L("🟡 Insufficient coverage", "🟡 Yetersiz doluluk")}
        for label, status, reason in log:
            st.caption(f"- **{label}:** {names.get(status, status)}" + (f" — {reason}" if status in ("service_error", "invalid_request") else ""))
        st.caption(L("Source: NASA POWER monthly point API (Giovanni first if a token was entered). Missing months are not filled.", "Kaynak: NASA POWER aylık nokta API'si (token girildiyse önce Giovanni). Eksik aylar doldurulmaz."))


def composite_direction_text(index_description: str, n_months: Dict[str, int]) -> str:
    """Bütünleşik indeksin anlamı (istatistik motorunun index_description metni) ve kullanılan ortak ay sayısı."""
    months = ", ".join(L(f"{n}: {m} months", f"{n}: {m} ay") for n, m in n_months.items())
    return L(
        f"**How to read the composite anomaly index?** {index_description} The signs are fixed (team rule: index rises = "
        f"warmer, more solar radiation, less water, less sea ice, lower albedo). "
        f"Only the common months in which all selected variables were observed are used ({months}).",
        f"**Bütünleşik anomali indeksi nasıl okunur?** {index_description} İşaretler sabittir (ekibin kuralı: indeks yükselir = "
        f"daha sıcak, daha fazla güneş ışınımı, daha az su, daha az deniz buzu, daha düşük albedo). "
        f"Yalnızca seçili tüm değişkenlerin gözlendiği ortak aylar kullanılır ({months}).",
    )


def fetch_and_analyze_region(
    lat: float,
    lon: float,
    selected_vars: List[str],
    start_year: int,
    end_year: int,
    hide_seasonality: bool,
) -> Dict[str, Any]:
    """
    Belirli bir koordinat için seçilen (1–4) parametreyi çeker, %70 gerçek veri doluluk kuralını denetler,
    mevsimsellikten arındırmayı, Mann-Kendall + Sen testini, CRITIC ağırlıklandırmayı ve 60 aylık istatistiksel uzatmayı
    (korumalarla) yürütür.
    """
    geo_info = reverse_geocode_and_detect_biome(lat, lon)
    param_raw_map: Dict[str, Dict[str, Any]] = {}
    param_full_map: Dict[str, pd.DataFrame] = {}
    param_stl_map: Dict[str, Dict[str, Any]] = {}
    param_mk_map: Dict[str, Dict[str, Any]] = {}
    param_fc_map: Dict[str, pd.DataFrame] = {}
    fc_notes: Dict[str, str] = {}
    missing_warnings: List[str] = []
    data_notes: List[str] = []

    for v in selected_vars:
        res_raw = fetch_hybrid_climate_series(
            lat=lat,
            lon=lon,
            variable=v,
            start_year=start_year,
            end_year=end_year,
        )
        param_raw_map[v] = res_raw
        _record_fetch_status(f"{OFFICIAL_VARIABLES[v]['short_name']} @ {lat:.2f}°, {lon:.2f}°", res_raw)

        if not res_raw["valid"]:
            missing_warnings.append(f"**{OFFICIAL_VARIABLES[v]['short_label']}:** {res_raw['reason']}")
            continue
        data_notes.append(f"**{OFFICIAL_VARIABLES[v]['short_label']}:** {res_raw['reason']}")
        # Dünya haritası işaretçileri için aynı noktanın tüm kaydı (önbellekteki aynı POWER yanıtından)
        param_full_map[v] = fetch_hybrid_climate_series(lat, lon, v, POWER_FIRST_YEAR, LATEST_YEAR)["df"]

        stl_res = stl_decompose_series(res_raw["df"], period=12)
        param_stl_map[v] = stl_res
        if not stl_res["valid"]:
            missing_warnings.append(
                L(f"**{OFFICIAL_VARIABLES[v]['short_label']}:** Real data exist, but deseasonalization could not be performed: {stl_res.get('reason', '')}",
                  f"**{OFFICIAL_VARIABLES[v]['short_label']}:** Gerçek veri var ancak mevsimsellikten arındırma yapılamadı: {stl_res.get('reason', '')}")
            )
            continue

        stl_df = stl_res["df"]
        mk_res = hamed_rao_mann_kendall_test(
            decimal_years(stl_df),
            stl_df["deseasonalized"].to_numpy(dtype=float),
            is_monthly=True,
        )
        param_mk_map[v] = mk_res

        target_col = "deseasonalized" if hide_seasonality else "observed"
        fc_single = generate_5year_sarima_forecast(
            stl_df,
            value_col=target_col,
            hide_seasonality=hide_seasonality,
            steps=60,
            bounds=OFFICIAL_VARIABLES[v].get("bounds"),
        )
        param_fc_map[v], fc_notes[v] = guard_forecast(fc_single, stl_df[target_col].to_numpy(dtype=float), end_year, LATEST_YEAR)

    comp_bundle = compute_objective_weights_and_composite(
        param_stl_map=param_stl_map,
        selected_vars=selected_vars,
        official_variables_meta=OFFICIAL_VARIABLES,
        hide_seasonality=hide_seasonality,
    )

    if not param_mk_map:
        return {
            "valid": False,
            "geo_info": geo_info,
            "missing_warnings": missing_warnings,
        }

    # Bileşik indeks kurulamazsa (ör. ortak aylar < %70) yalnızca değişken bazlı sonuçlar gösterilir.
    comp_df = comp_bundle["composite_df"]
    comp_mk, comp_fc = None, None
    if comp_bundle["valid"]:
        # İşaretler istatistik motorunda OFFICIAL_VARIABLES["polarity"] alanından sabit olarak alınır (seçime bağlı değil)
        comp_mk = hamed_rao_mann_kendall_test(
            decimal_years(comp_df),
            comp_df["value_deseasonalized"].to_numpy(dtype=float),
            is_monthly=True,
        )
        comp_fc_raw = generate_5year_sarima_forecast(
            comp_df,
            value_col="value",
            hide_seasonality=hide_seasonality,
            steps=60,
        )
        comp_fc, fc_notes["COMPOSITE"] = guard_forecast(comp_fc_raw, comp_df["value"].to_numpy(dtype=float), end_year, LATEST_YEAR)
    weights_df = comp_bundle["weights_df"]

    return {
        "valid": True,
        "composite_valid": bool(comp_bundle["valid"]),
        "composite_reason": comp_bundle.get("reason", ""),
        "lat": float(lat),
        "lon": float(lon),
        "geo_info": geo_info,
        "missing_warnings": missing_warnings,
        "data_notes": data_notes,
        "param_raw_map": param_raw_map,
        "param_full_map": param_full_map,
        "param_stl_map": param_stl_map,
        "param_mk_map": param_mk_map,
        "param_fc_map": param_fc_map,
        "fc_notes": fc_notes,
        "weights_df": weights_df,
        "weight_method": comp_bundle.get("weight_method", ""),
        "index_description": comp_bundle.get("index_description", ""),
        "n_months_used": comp_bundle.get("n_months_used", len(comp_df)),
        "vif_warnings": comp_bundle["vif_warnings"],
        "composite_df": comp_df,
        "composite_mk": comp_mk,
        "composite_fc": comp_fc,
    }


def render_weights_and_vif_section(weights_df: pd.DataFrame, vif_warnings: List[str], weight_method: str = "") -> None:
    """İstatistik motorunun döndürdüğü CRITIC ağırlık tablosunu ve yöntem açıklamasını gösterir."""
    with st.expander(L("⚖️ CRITIC Weights (VIF is diagnostic only)", "⚖️ CRITIC Ağırlıkları (VIF yalnızca tanı)"), expanded=True):
        st.caption(L(
            f"Weighting method (as reported by the statistics engine): **{weight_method or 'not specified'}**. "
            "Team importance scores are the team's own choice (there is no external source) and are not used in the weights. "
            "The 'Sign' column shows the fixed direction in the composite index.",
            f"Ağırlık yöntemi (istatistik motorunun bildirdiği): **{weight_method or 'belirtilmedi'}**. "
            "Ekip önem puanları ekibin kendi seçimidir (harici kaynak yoktur) ve ağırlıkta kullanılmaz. "
            "'İşaret' sütunu bütünleşik indeksteki sabit yönü gösterir."
        ))
        if vif_warnings:
            for vw in vif_warnings:
                st.warning(L(f"🛡️ **Multicollinearity Warning (diagnostic):** {vw}", f"🛡️ **Çoklu Doğrusallık Uyarısı (tanı):** {vw}"))
        cols = {
            "rank": L("Rank", "Sıra"),
            "label": L("Parameter", "Parametre"),
            "team_importance_score": L("Team Importance Score (not used in weights)", "Ekip Önem Puanı (ağırlıkta kullanılmaz)"),
            "critic_score": L("CRITIC Score (C_j)", "CRITIC Skoru (C_j)"),
            "vif_value": L("VIF Value", "VIF Değeri"),
            "vif_status": L("Multicollinearity Status", "Çoklu Doğrusallık Durumu"),
            "normalized_weight": L("Normalized Weight (∑=1.00)", "Normalize Ağırlık (∑=1.00)"),
            "weight_pct": L("Share of Influence (%)", "Etki Oranı (%)"),
            "polarity_sign": L("Sign (in index)", "İşaret (indekste)"),
        }
        disp_df = weights_df[[c for c in cols if c in weights_df.columns]].rename(columns=cols)
        st.dataframe(disp_df, width="stretch", hide_index=True)


def _p_txt(disp: str) -> str:
    """Motorun p_value_display metnini ('0.003', '< 0.001') okunur biçime çevirir: 'p = 0.003' / 'p < 0.001'."""
    disp = str(disp or "—").strip()
    return f"p {disp}" if disp[:1] in "<>" else f"p = {disp}"


def _build_short_mk_comment(mk: Dict[str, Any], unit_label: str) -> str:
    """Sen eğimi, Kendall tau, %95 güven aralığı ve p-değerine göre kısa yorum (karar ve GA motorda aynı varyanstan)."""
    slope_dec = float(mk.get("sen_slope_decade", 0.0))
    tau = float(mk.get("kendall_tau", 0.0))
    abs_tau = abs(tau)
    is_sig = bool(mk.get("is_significant", False))
    direction = mk.get("direction", "stable")

    if abs_tau >= 0.45:
        strength_txt = L("very strong and consistent", "çok güçlü ve istikrarlı")
    elif abs_tau >= 0.25:
        strength_txt = L("clear (moderate–strong)", "belirgin (orta–güçlü)")
    elif abs_tau >= 0.12:
        strength_txt = L("weak–moderate", "hafif–orta düzeyde")
    else:
        strength_txt = L("weak", "zayıf")

    if direction == "increasing":
        dir_txt, icon = L(f"{strength_txt} upward trend", f"{strength_txt} yükseliş eğilimi"), "📈"
    elif direction == "decreasing":
        dir_txt, icon = L(f"{strength_txt} downward trend", f"{strength_txt} azalış eğilimi"), "📉"
    else:
        dir_txt, icon = L("flat (zero-slope) behaviour", "durağan (sıfır eğim) seyir"), "➖"

    if mk.get("ci_excludes_zero", False):
        ci_txt = L("the 95% confidence interval excludes zero (supports the direction of the slope)", "%95 güven aralığı sıfırı dışlıyor (eğimin yönünü destekler)")
    else:
        ci_txt = L("the 95% confidence interval includes zero (the direction of the slope is uncertain)", "%95 güven aralığı sıfırı içeriyor (eğim yönü belirsiz)")

    p_txt = _p_txt(mk.get("p_value_display"))
    if is_sig:
        sig_txt = L(f"is inconsistent with the 'no trend' hypothesis ({p_txt}, threshold 0.05; statistically significant).",
                    f"'trend yok' varsayımıyla uyumsuzdur ({p_txt}, eşik 0.05; istatistiksel olarak anlamlı).")
    else:
        sig_txt = L(f"cannot be distinguished from the 'no trend' hypothesis ({p_txt}, threshold 0.05); this does not prove that there is no trend.",
                    f"'trend yok' varsayımından ayırt edilemiyor ({p_txt}, eşik 0.05); bu, trendin olmadığını kanıtlamaz.")
    if mk.get("consistency_note"):
        sig_txt += L(f" Note: {mk['consistency_note']}", f" Not: {mk['consistency_note']}")
    if not mk.get("reliable_verdict", True):
        sig_txt += L(" Because the series is short, this verdict is not reliable.", " Seri kısa olduğu için bu karar güvenilir değildir.")

    return L(
        f"{icon} **Short Statistical Comment:** The series shows a **{dir_txt}** (`τ = {tau:+.3f}`) "
        f"at an average rate of `{slope_dec:+.3f} {unit_label}` per decade; {ci_txt}, and this change {sig_txt}",
        f"{icon} **Kısa İstatistiksel Yorum:** Seride 10 yılda ortalama `{slope_dec:+.3f} {unit_label}` hızında "
        f"**{dir_txt}** (`τ = {tau:+.3f}`) gözlenmektedir; {ci_txt} ve bu değişim {sig_txt}"
    )


def render_mk_significance_card(title: str, mk: Dict[str, Any], unit_label: Optional[str] = None) -> None:
    """Mann-Kendall + Sen (otokorelasyon / uzun hafıza düzeltmeli) anlamlılık kartını ve kısa yorumunu basar."""
    if unit_label is None:
        unit_label = L("Z-score", "Z-Skoru")
    badge_class = "sig-badge-ok" if mk["is_significant"] else "sig-badge-noise"
    st.markdown(
        f"""
        <div class="{badge_class}">
            <b>{title}:</b> {mk['badge_text']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not mk.get("reliable_verdict", True) and mk.get("reliability_note"):
        st.warning(L(f"⚠️ **Unreliable verdict:** {mk['reliability_note']}", f"⚠️ **Güvenilir olmayan karar:** {mk['reliability_note']}"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(L("Sen's Slope (Annual Rate)", "Sen's Slope (Yıllık Hız)"), L(f"{mk['sen_slope_annual']:+.4f} {unit_label}/yr", f"{mk['sen_slope_annual']:+.4f} {unit_label}/yıl"), L(f"{mk['sen_slope_decade']:+.3f} /decade", f"{mk['sen_slope_decade']:+.3f} /10 yıl"))
    m2.metric("Kendall's Tau (τ) & S", f"τ = {mk['kendall_tau']:+.3f}", f"S = {mk['S']}")
    m3.metric(
        L("95% Confidence Interval (Slope)", "%95 Güven Aralığı (Eğim)"),
        f"[{mk['sen_ci_low_annual']:+.4f}, {mk['sen_ci_high_annual']:+.4f}]",
        L(f"variance factor x{mk['autocorr_factor']:.2f} ({mk.get('variance_binding') or '—'}) | effective n ≈ {float(mk.get('n_effective', 0)):.0f}",
          f"varyans çarpanı x{mk['autocorr_factor']:.2f} ({mk.get('variance_binding') or '—'}) | etkin n ≈ {float(mk.get('n_effective', 0)):.0f}"),
    )
    m4.metric(L("Mann-Kendall Z & p (dependence-corrected)", "Mann-Kendall Z & p (bağımlılık düzeltmeli)"), _p_txt(mk.get("p_value_display")), L(f"Z = {mk['z_score']:+.3f} | t df = {mk.get('t_dof', '—')}", f"Z = {mk['z_score']:+.3f} | t sd = {mk.get('t_dof', '—')}"))
    st.caption(L(f"Method: {mk.get('method_short', '')}. ", f"Yöntem: {mk.get('method_short', '')}. ") + _build_short_mk_comment(mk, unit_label))


def year_span_ok() -> bool:
    """Seçilen tarih aralığı analiz için yeterli mi? Değilse açık uyarı verir (sessizce yanlış grafik çizmez)."""
    yr0, yr1 = st.session_state.year_range
    span = yr1 - yr0 + 1
    if span < MIN_YEAR_SPAN:
        st.warning(L(
            f"⏳ The selected range **{yr0}–{yr1}** covers only **{span} years** ({span * 12} months). "
            f"Select a range of at least **{MIN_YEAR_SPAN} years** for deseasonalizing and the Mann-Kendall trend test. "
            f"Ranges of {MIN_YEAR_SPAN}–9 years work, but for series shorter than 10 years the trend verdict is flagged as 'not reliable'.",
            f"⏳ Seçilen aralık **{yr0}–{yr1}** yalnızca **{span} yıl** ({span * 12} ay). "
            f"Mevsimsellikten arındırma ve Mann-Kendall trend testi için en az **{MIN_YEAR_SPAN} yıllık** bir aralık seçin. "
            f"{MIN_YEAR_SPAN}–9 yıllık aralıklar çalışır ancak 10 yıldan kısa serilerde trend kararı 'güvenilir değil' olarak işaretlenir."
        ))
        return False
    return True


def _weight_lookup(weights_df: pd.DataFrame) -> Dict[str, float]:
    if weights_df is None or weights_df.empty:
        return {}
    return {r["var_code"]: float(r["weight_pct"]) for _, r in weights_df.iterrows()}


def _region_label(prefix: str, geo: Dict[str, Any], lat: float, lon: float) -> str:
    return f"{prefix} {geo['country']} ({geo['city_region']}) [{lat:.1f}°, {lon:.1f}°]"


def build_shared_comparison(
    region_results: Dict[str, Dict[str, Any]],
    selected_vars: List[str],
    hide_seasonality: bool,
    end_year: int,
) -> Dict[str, Any]:
    """
    Çiftli/Çoklu kıyas için: ortak ağırlıklar → her bölgenin ortak ağırlıklı bütünleşik serisi,
    Mann-Kendall testi ve istatistiksel uzatma.
    """
    names = list(region_results.keys())
    shared = compute_shared_weights(
        [region_results[n]["param_stl_map"] for n in names],
        selected_vars,
        OFFICIAL_VARIABLES,
        hide_seasonality,
    )
    if not shared["valid"]:
        return {"valid": False, "dropped_vars": shared["dropped_vars"], "reason": shared.get("reason", "")}

    weights_df = shared["weights_df"]
    composites, forecasts, fc_notes = {}, {}, {}
    for n in names:
        comp = apply_weights_to_composite(region_results[n]["param_stl_map"], weights_df, hide_seasonality)
        composites[n] = comp
        forecasts[n], fc_notes[n] = guard_forecast(
            generate_5year_sarima_forecast(comp, "value", hide_seasonality, steps=60),  # z-indeks: fiziksel sınır yok
            comp["value"].to_numpy(dtype=float),
            end_year,
            LATEST_YEAR,
        )
    return {
        "valid": True,
        "fc_notes": fc_notes,
        "weights_df": weights_df,
        "weight_method": shared.get("weight_method", ""),
        "index_description": shared.get("index_description", ""),
        "vif_warnings": shared["vif_warnings"],
        "common_vars": shared["common_vars"],
        "dropped_vars": shared["dropped_vars"],
        "composites": composites,
        "forecasts": forecasts,
    }


REL_CLASS_TR = {
    "opposite": {"en": "Opposite direction", "tr": "Ters Yönlü"},
    "same": {"en": "Same direction", "tr": "Eş Yönlü"},
    "none": {"en": "No significant relationship", "tr": "Anlamlı ilişki yok"},
    "identical": {"en": "Same POWER cell (identical series)", "tr": "Aynı POWER hücresi (özdeş seri)"},
}


def render_without_composite(region_results: Dict[str, Dict[str, Any]], shared: Dict[str, Any], hide: bool) -> None:
    """Ortak bileşik indeks kurulamadığında gerekçeyi söyler ve parametre bazlı kıyasları yine gösterir."""
    st.warning(f"⚖️ {shared.get('reason') or L('A shared composite index could not be built.', 'Ortak bütünleşik indeks oluşturulamadı.')}")
    common = [
        v for v in st.session_state.selected_vars
        if all(v in r["param_mk_map"] for r in region_results.values())
    ]
    if common:
        render_per_parameter_comparisons(region_results, common, pd.DataFrame(), hide)


def same_cell_warning(points: Dict[str, Tuple[float, float]]) -> None:
    """Aynı NASA POWER hücresine düşen bölge çiftleri için uyarı (seriler özdeş olur)."""
    names = list(points)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ca, cb = power_cell(*points[names[i]]), power_cell(*points[names[j]])
            shared_src = [lbl for key, lbl in (("merra2", L("MERRA-2 (meteorology)", "MERRA-2 (meteoroloji)")), ("radiation", "CERES 1° (ALBEDO, SWGDN)"))
                          if ca[key] == cb[key]]
            if shared_src:
                st.warning(L(
                    f"📍 **{names[i]}** and **{names[j]}** are in the same NASA POWER cell ({', '.join(shared_src)}): for "
                    f"variables from this source the series are exactly identical, so no relationship verdict is given for these variables.",
                    f"📍 **{names[i]}** ve **{names[j]}** aynı NASA POWER hücresinde ({', '.join(shared_src)}): bu "
                    f"kaynaktan gelen değişkenlerde seriler birebir aynıdır; bu değişkenler için ilişki kararı verilmez."
                ))


def render_world_map(var_code: str, points: List[Dict[str, Any]], key: str) -> None:
    """NASA POWER 5° ızgara haritası (erken / geç / değişim) + açıklama; ızgarada olmayan değişkenler için harita yok."""
    meta = OFFICIAL_VARIABLES[var_code]
    try:
        fig = plot_world_period_change_map(var_code, meta["short_label"], meta["unit"], points, st.session_state.year_range)
    except FileNotFoundError:
        st.warning(L("World map data file not found (data/global_grid_power_5deg.csv); regenerate it with the command in the README.",
                     "Dünya haritası veri dosyası bulunamadı (data/global_grid_power_5deg.csv); README'deki komutla yeniden üretin."))
        return
    if fig is None:
        st.caption(L(f"🗺️ No global grid data for {meta['short_label']} (not available in NASA POWER); no map drawn.",
                     f"🗺️ {meta['short_label']} için küresel ızgara verisi yok (NASA POWER'da bulunmuyor); harita çizilmedi."))
        return
    st.plotly_chart(fig, width="stretch", key=key)
    st.caption(world_map_caption(var_code, st.session_state.year_range))


def render_per_parameter_comparisons(
    region_results: Dict[str, Dict[str, Any]],
    common_vars: List[str],
    weights_df: pd.DataFrame,
    hide_seasonality: bool,
) -> None:
    """Bütünleşik grafiğin ALTINDA: her parametre için AYRI bölgeler arası kıyas grafiği + NASA POWER 5° dünya haritası + ilişki özeti."""
    st.markdown(L("### 📈 Separate Comparison Charts per Parameter and NASA POWER World Maps", "### 📈 Parametre Bazlı Ayrık Kıyas Grafikleri ve NASA POWER Dünya Haritaları"))
    st.caption(L(
        "Each parameter is shown in its own chart in its physical unit; below it is an early-period / late-period / change map "
        "from NASA POWER 5° point samples. Relationship statistics are computed on series with seasonality and the linear trend "
        "removed; they are measures of association, not proof of causation.",
        "Her parametre kendi fiziksel biriminde ayrı bir grafikte gösterilir; altında NASA POWER 5° nokta örneklerinden "
        "erken dönem / geç dönem / değişim haritası yer alır. İlişki istatistikleri mevsimselliği ve doğrusal trendi "
        "çıkarılmış seriler üzerinden hesaplanır; ilişki ölçüleridir, nedensellik kanıtı değildir."
    ))
    w_map = _weight_lookup(weights_df)
    rel_rows = []

    for v in common_vars:
        meta = OFFICIAL_VARIABLES[v]
        region_stl = {n: r["param_stl_map"][v] for n, r in region_results.items()}
        region_mk = {n: r["param_mk_map"][v] for n, r in region_results.items()}
        rel = compare_parameter_across_regions(region_stl, use_col="deseasonalized")

        fig = plot_parameter_regions_comparison(
            var_code=v,
            region_stl=region_stl,
            region_mk=region_mk,
            meta=meta,
            weight_pct=w_map.get(v),
            rel_stats=rel,
            hide_seasonality=hide_seasonality,
        )
        st.plotly_chart(fig, width="stretch", key=f"param_cmp_{v}")

        render_world_map(
            v,
            [{"name": n, "lat": r["lat"], "lon": r["lon"], "df": r["param_full_map"].get(v)} for n, r in region_results.items()],
            key=f"world_map_cmp_{v}_{len(region_results)}",
        )

        with st.expander(L(f"📐 {meta['short_label']} — Statistical Details by Region (Sen's Slope, Kendall τ, 95% CI, p-value and Comment)",
                           f"📐 {meta['short_label']} — Bölgelerin İstatistik Detayı (Sen's Slope, Kendall τ, %95 GA, p-Değeri ve Yorum)")):
            for n, mk_v in region_mk.items():
                render_mk_significance_card(f"{n} — {meta['short_label']}", mk_v, unit_label=meta["unit"])

        if rel.get("relationship") == "identical":
            st.info(f"⚪ {rel['explanation']}")
            rel_rows.append({L("Parameter", "Parametre"): meta["short_label"], L("Weight (%)", "Ağırlık (%)"): w_map.get(v), "Spearman ρ (1.–2.)": None,
                             L("p (effective n)", "p (etkin n)"): "—", L("Common months", "Ortak ay"): rel["n_common_months"], "ΔSlope (1.–2.)": rel["delta_slope"],
                             L("Result", "Sonuç"): rel["verdict_badge"]})
        elif rel.get("valid", False):
            if len(region_results) == 2:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(L("Spearman ρ (deseasonalized + detrended)", "Spearman ρ (arındırılmış + trendsiz)"), f"{rel['spearman_rho']:+.3f}", L(f"{_p_txt(rel['p_value_display'])} | {rel['n_common_months']} common months", f"{_p_txt(rel['p_value_display'])} | {rel['n_common_months']} ortak ay"))
                c2.metric("ΔSlope (A − B)", f"{rel['delta_slope']:+.4f}", L(f"{meta['unit']}/yr", f"{meta['unit']}/yıl"))
                c3.metric(L("Max. Lagged Correlation (descriptive)", "Maks. Gecikmeli Korelasyon (betimsel)"), f"{rel['best_xcorr']:+.3f}", L(f"k = {rel['best_lag_months']} months", f"k = {rel['best_lag_months']} ay"))
                c4.metric(L("Relationship Class (association, not causation)", "İlişki Sınıfı (ilişki, nedensellik değil)"), pick(REL_CLASS_TR.get(rel["relationship"], rel["relationship"])))
                st.caption(L(f"Lag convention: {rel['lag_convention']}", f"Gecikme kuralı: {rel['lag_convention']}"))
            else:
                with st.expander(L(f"🔎 {meta['short_label']} — Relationship table for region pairs", f"🔎 {meta['short_label']} — Bölge çiftleri ilişki tablosu")):
                    st.dataframe(rel["pairwise_df"], width="stretch", hide_index=True)
            rel_rows.append(
                {
                    L("Parameter", "Parametre"): meta["short_label"],
                    L("Weight (%)", "Ağırlık (%)"): w_map.get(v),
                    "Spearman ρ (1.–2.)": rel["spearman_rho"],
                    L("p (effective n)", "p (etkin n)"): rel["p_value_display"],
                    L("Common months", "Ortak ay"): rel["n_common_months"],
                    "ΔSlope (1.–2.)": rel["delta_slope"],
                    L("Result", "Sonuç"): rel["verdict_badge"],
                }
            )
        st.markdown("---")

    if rel_rows:
        st.markdown(L("#### 🧾 Opposition / Relationship Summary per Parameter", "#### 🧾 Parametre Bazlı Zıtlık / İlişki Özeti"))
        st.dataframe(pd.DataFrame(rel_rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# ROTA 3: `/analyze/single` — TEKLİ BÖLGE ANALİZİ (HARİTADA SADECE 1 İKON)
# ---------------------------------------------------------------------------
def render_single_analyze_route() -> None:
    st.subheader(L("📍 Single-Region Climate Trend, Deseasonalizing and Statistical Extension (/analyze/single)", "📍 Tekli Bölge İklim Trendi, Mevsimsel Arındırma ve İstatistiksel Uzatma (/analyze/single)"))
    if st.session_state.active_case_title:
        st.caption(L(f"📌 **Active Case Study:** {pick(st.session_state.active_case_title)}", f"📌 **Aktif Vaka Çalışması:** {pick(st.session_state.active_case_title)}"))

    mk_s = st.session_state.marker_single
    cver = st.session_state.coord_version

    if mk_s is not None:
        lat_s = float(mk_s["lat"])
        lon_s = float(mk_s["lon"])
        geo_s = reverse_geocode_and_detect_biome(lat_s, lon_s)

        c_info, c_lat_in, c_lon_in, c_clr = st.columns([2.1, 0.65, 0.65, 0.6])
        with c_info:
            st.markdown(
                f"""
                <div class="route-card" style="font-size:14px;padding:12px 16px;">
                    <b>{L("🎯 Selected Region (Single Icon on Map):", "🎯 Seçili Bölge (Haritada Tek İkon):")}</b> {geo_s['country']} — {geo_s['city_region']}
                    &nbsp;|&nbsp; <b>{L("Surface:", "Yüzey:")}</b> {geo_s['biome_name']} &nbsp;|&nbsp; <code>{lat_s:.2f}°, {lon_s:.2f}°</code>
                    <br><small style="color:#94A3B8;">{L("Name:", "Ad:")} {geo_s['name_source']}</small>
                    <br><small style="color:#94A3B8;">{geo_s['power_cell_note']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_lat_in:
            new_lat = st.number_input(L("Latitude (Lat):", "Enlem (Lat):"), -85.0, 85.0, value=lat_s, step=0.25, format="%.2f", key=f"in_single_lat_{cver}")
        with c_lon_in:
            new_lon = st.number_input(L("Longitude (Lon):", "Boylam (Lon):"), -180.0, 180.0, value=lon_s, step=0.25, format="%.2f", key=f"in_single_lon_{cver}")
        with c_clr:
            st.write("")
            if st.button(L("❌ Remove Selection", "❌ Seçimi Kaldır"), key="clr_single_btn", width="stretch"):
                st.session_state.marker_single = None
                st.session_state.active_case_title = None
                st.session_state.coord_version += 1
                st.rerun()

        if round(new_lat, 2) != round(lat_s, 2) or round(new_lon, 2) != round(lon_s, 2):
            st.session_state.marker_single = {"lat": float(new_lat), "lon": float(new_lon)}
            st.rerun()
        if geo_s["mixed_cell_note"]:
            st.warning(geo_s["mixed_cell_note"])

        m = _create_base_folium_map(lat_s, lon_s, zoom=4)
        folium.Marker(
            location=[lat_s, lon_s],
            tooltip=folium.Tooltip(geo_s["tooltip_html"], sticky=True),
            popup=folium.Popup(geo_s["tooltip_html"], max_width=280),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
    else:
        geo_s = None
        c_info, c_lat_in, c_lon_in, c_add = st.columns([2.0, 0.65, 0.65, 0.7])
        with c_info:
            st.info(L("👆 **No region selected yet.** Click any country/point on the map, or enter coordinates on the right and press **📍 Select Region**.",
                      "👆 **Henüz bölge seçilmedi.** Harita üzerinde istediğiniz ülkeye/noktaya tıklayın veya sağdan koordinat girip **📍 Bölgeyi Seç** butonuna basın."))
        with c_lat_in:
            in_lat = st.number_input(L("Latitude (Lat):", "Enlem (Lat):"), -85.0, 85.0, value=0.0, step=0.5, format="%.2f", key=f"empty_single_lat_{cver}")
        with c_lon_in:
            in_lon = st.number_input(L("Longitude (Lon):", "Boylam (Lon):"), -180.0, 180.0, value=0.0, step=0.5, format="%.2f", key=f"empty_single_lon_{cver}")
        with c_add:
            st.write("")
            if st.button(L("📍 Select Region", "📍 Bölgeyi Seç"), type="primary", width="stretch", key="set_single_btn"):
                st.session_state.marker_single = {"lat": float(in_lat), "lon": float(in_lon)}
                st.session_state.coord_version += 1
                st.rerun()
        m = _create_base_folium_map(25.0, 15.0, zoom=2)

    folium.LayerControl().add_to(m)

    map_out = st_folium(m, use_container_width=True, height=440, key="nasa_map_single", returned_objects=["last_clicked"])
    if _clicked(map_out):
        c_lat, c_lon = _clicked(map_out)
        click_pt = (c_lat, c_lon)
        if click_pt != st.session_state.get("single_last_click"):
            st.session_state.single_last_click = click_pt
            st.session_state.marker_single = {"lat": c_lat, "lon": c_lon}
            st.session_state.coord_version += 1
            st.session_state.active_case_title = None
            st.rerun()

    render_analysis_controls(geo_s)
    if mk_s is None:
        return
    if not year_span_ok():
        return

    start_yr, end_yr = st.session_state.year_range
    hide = st.session_state.hide_seasonality
    res = fetch_and_analyze_region(
        lat_s, lon_s, st.session_state.selected_vars, start_yr, end_yr, hide,
    )
    render_fetch_status_sidebar()

    for mw in res["missing_warnings"]:
        st.error(f"⚠️ {mw}")
    if not res["valid"]:
        return
    with st.expander(L("📡 Data source, actual data window and completeness (per variable)", "📡 Veri kaynağı, gerçek veri penceresi ve doluluk (değişken başına)")):
        for note in res["data_notes"]:
            st.markdown(f"- {note}")

    # 1. Bütünleşik grafik (seçilen parametrelerin ağırlıklı sentezi)
    st.markdown(L("### 🌍 Composite Anomaly Index (z-score)", "### 🌍 Bütünleşik Anomali İndeksi (z-skoru)"))
    if res["composite_valid"]:
        render_weights_and_vif_section(res["weights_df"], res["vif_warnings"], res.get("weight_method", ""))
        st.caption(composite_direction_text(res["index_description"], {L("This region", "Bu bölge"): res["n_months_used"]}))
        render_mk_significance_card(L(f"{geo_s['display_label']} — Composite Trend", f"{geo_s['display_label']} — Bütünleşik Trend"), res["composite_mk"])
        st.plotly_chart(
            plot_single_region_composite_and_forecast(
                geo_s["display_label"], res["composite_df"], res["composite_mk"],
                res["composite_fc"], res["weights_df"], hide,
            ),
            width="stretch",
            key="single_composite",
        )
        st.caption(L(f"{res['fc_notes'].get('COMPOSITE', '')} No world map is drawn for the composite index because there is no global field for it.",
                     f"{res['fc_notes'].get('COMPOSITE', '')} Bütünleşik indeks için küresel alan olmadığından dünya haritası çizilmez."))
    else:
        st.warning(f"⚖️ {res['composite_reason']}")
    st.info(pick(FORECAST_NOTE))

    # 2. HER PARAMETRE İÇİN AYRI GRAFİK + ALTINDA NASA POWER DÜNYA HARİTASI
    st.markdown(L("### 📊 Separate Charts per Parameter and NASA POWER World Maps", "### 📊 Parametre Bazlı Ayrı Grafikler ve NASA POWER Dünya Haritaları"))
    w_map = _weight_lookup(res["weights_df"])
    valid_vars = [v for v in st.session_state.selected_vars if v in res["param_mk_map"]]
    for idx, v in enumerate(valid_vars):
        meta = OFFICIAL_VARIABLES[v]
        st.plotly_chart(
            plot_single_parameter_chart(
                region_name=geo_s["display_label"],
                var_code=v,
                stl_res=res["param_stl_map"][v],
                mk=res["param_mk_map"][v],
                fc_df=res["param_fc_map"].get(v),
                meta=meta,
                weight_pct=w_map.get(v),
                hide_seasonality=hide,
                color=PARAM_PALETTE[idx % len(PARAM_PALETTE)],
            ),
            width="stretch",
            key=f"single_param_{v}",
        )
        st.caption(res["fc_notes"].get(v, ""))
        render_world_map(
            v,
            [{"name": geo_s["display_label"], "lat": lat_s, "lon": lon_s, "df": res["param_full_map"].get(v)}],
            key=f"single_param_world_map_{v}",
        )
        with st.expander(L(f"📐 {meta['short_label']} — Statistical Details (Sen's Slope, Kendall τ, 95% CI, p-value and Comment)",
                           f"📐 {meta['short_label']} — İstatistik Detayı (Sen's Slope, Kendall τ, %95 GA, p-Değeri ve Yorum)"), expanded=True):
            render_mk_significance_card(meta["short_label"], res["param_mk_map"][v], unit_label=meta["unit"])

    with st.expander(L("🔗 Comparison links in NASA Giovanni (unverified)", "🔗 NASA Giovanni'de karşılaştırma bağlantıları (doğrulanmamış)")):
        st.caption(L(
            "These links try to open the 1°×1° area average of the MERRA-2 variable in Giovanni (the link format "
            "has not been verified). The values in the app are NASA POWER point data; for ALBEDO/SWGDN, POWER uses CERES while "
            "Giovanni uses MERRA-2. So the numbers will not match exactly.",
            "Bu bağlantılar Giovanni'de MERRA-2 değişkeninin 1°×1° alan ortalamasını açmaya çalışır (bağlantı biçimi "
            "doğrulanmadı). Uygulamadaki değerler NASA POWER nokta verisidir; ALBEDO/SWGDN için POWER CERES kullanırken "
            "Giovanni MERRA-2 kullanır. Bu nedenle sayılar birebir aynı çıkmaz."
        ))
        for v in st.session_state.selected_vars:
            g_url = build_giovanni_direct_url(lat_s, lon_s, v, start_yr, end_yr)
            st.markdown(L(f"- **{OFFICIAL_VARIABLES[v]['label']}:** [Open Giovanni link]({g_url})", f"- **{OFFICIAL_VARIABLES[v]['label']}:** [Giovanni bağlantısını aç]({g_url})"))


# ---------------------------------------------------------------------------
# ROTA 4: `/analyze/dual` — ÇİFTLİ BÖLGE KARŞILAŞTIRMASI (TELECONNECTION)
# ---------------------------------------------------------------------------
def render_dual_analyze_route() -> None:
    st.subheader(L("⚡ Two-Region Opposition / Relationship Analysis and Statistical Extension (/analyze/dual)", "⚡ Çiftli Bölge Zıtlık / İlişki Analizi ve İstatistiksel Uzatma (/analyze/dual)"))
    if st.session_state.active_case_title:
        st.caption(L(f"📌 **Active Case Study:** {pick(st.session_state.active_case_title)}", f"📌 **Aktif Vaka Çalışması:** {pick(st.session_state.active_case_title)}"))

    m_a, m_b = st.session_state.markers_dual
    geo_a = reverse_geocode_and_detect_biome(float(m_a["lat"]), float(m_a["lon"])) if m_a else None
    geo_b = reverse_geocode_and_detect_biome(float(m_b["lat"]), float(m_b["lon"])) if m_b else None
    cver = st.session_state.coord_version

    top_c1, top_c2, top_c3 = st.columns([1.25, 1.37, 1.37])
    with top_c1:
        default_idx = 0 if m_a is None else (1 if m_b is None else 0)
        target_pin = st.radio(
            L("🖱️ Region to Select / Update When the Map Is Clicked:", "🖱️ Haritaya Tıklanınca Seçilecek / Güncellenecek Bölge:"),
            options=[0, 1],
            format_func=lambda v: L("🔴 Region A (1st Pin)", "🔴 Bölge A (1. Pin)") if v == 0 else L("🔵 Region B (2nd Pin)", "🔵 Bölge B (2. Pin)"),
            index=default_idx,
            horizontal=True,
            key="dual_target_pin",
        )
        if st.button(L("🗑️ Clear Both Region Selections", "🗑️ Her İki Bölge Seçimini Temizle"), disabled=(m_a is None and m_b is None), width="stretch", key="clr_dual_both"):
            st.session_state.markers_dual = [None, None]
            st.session_state.active_case_title = None
            st.session_state.coord_version += 1
            st.rerun()

    with top_c2:
        if m_a is not None and geo_a is not None:
            st.markdown(L(f"**🔴 Region A:** `{geo_a['display_label']}`  \n*Surface:* {geo_a['biome_name']}  \n<small>Name: {geo_a['name_source']}</small>  \n<small>{geo_a['power_cell_note']}</small>",
                          f"**🔴 Bölge A:** `{geo_a['display_label']}`  \n*Yüzey:* {geo_a['biome_name']}  \n<small>Ad: {geo_a['name_source']}</small>  \n<small>{geo_a['power_cell_note']}</small>"), unsafe_allow_html=True)
            if geo_a["mixed_cell_note"]:
                st.warning(geo_a["mixed_cell_note"])
            if st.button(L("❌ Remove Region A Selection", "❌ Bölge A Seçimini Kaldır"), key="rm_dual_a", width="stretch"):
                st.session_state.markers_dual[0] = None
                st.session_state.coord_version += 1
                st.rerun()
        else:
            st.markdown(L("**🔴 Region A:** *Not selected yet (click on the map or enter coordinates)*", "**🔴 Bölge A:** *Henüz seçilmedi (Haritadan tıklayın veya koordinat girin)*"))
            ca1, ca2, ca3 = st.columns([1, 1, 0.9])
            na_lat = ca1.number_input(L("Latitude A:", "Enlem A:"), -85.0, 85.0, value=0.0, step=0.5, key=f"dual_a_lat_{cver}")
            na_lon = ca2.number_input(L("Longitude A:", "Boylam A:"), -180.0, 180.0, value=0.0, step=0.5, key=f"dual_a_lon_{cver}")
            ca3.write("")
            if ca3.button(L("➕ Select A", "➕ Seç A"), key="set_dual_a", width="stretch"):
                st.session_state.markers_dual[0] = {"lat": float(na_lat), "lon": float(na_lon)}
                st.session_state.coord_version += 1
                st.rerun()

    with top_c3:
        if m_b is not None and geo_b is not None:
            st.markdown(L(f"**🔵 Region B:** `{geo_b['display_label']}`  \n*Surface:* {geo_b['biome_name']}  \n<small>Name: {geo_b['name_source']}</small>  \n<small>{geo_b['power_cell_note']}</small>",
                          f"**🔵 Bölge B:** `{geo_b['display_label']}`  \n*Yüzey:* {geo_b['biome_name']}  \n<small>Ad: {geo_b['name_source']}</small>  \n<small>{geo_b['power_cell_note']}</small>"), unsafe_allow_html=True)
            if geo_b["mixed_cell_note"]:
                st.warning(geo_b["mixed_cell_note"])
            if st.button(L("❌ Remove Region B Selection", "❌ Bölge B Seçimini Kaldır"), key="rm_dual_b", width="stretch"):
                st.session_state.markers_dual[1] = None
                st.session_state.coord_version += 1
                st.rerun()
        else:
            st.markdown(L("**🔵 Region B:** *Not selected yet (click on the map or enter coordinates)*", "**🔵 Bölge B:** *Henüz seçilmedi (Haritadan tıklayın veya koordinat girin)*"))
            cb1, cb2, cb3 = st.columns([1, 1, 0.9])
            nb_lat = cb1.number_input(L("Latitude B:", "Enlem B:"), -85.0, 85.0, value=0.0, step=0.5, key=f"dual_b_lat_{cver}")
            nb_lon = cb2.number_input(L("Longitude B:", "Boylam B:"), -180.0, 180.0, value=0.0, step=0.5, key=f"dual_b_lon_{cver}")
            cb3.write("")
            if cb3.button(L("➕ Select B", "➕ Seç B"), key="set_dual_b", width="stretch"):
                st.session_state.markers_dual[1] = {"lat": float(nb_lat), "lon": float(nb_lon)}
                st.session_state.coord_version += 1
                st.rerun()

    m = _create_base_folium_map(25.0, 15.0, zoom=2)
    if m_a is not None and geo_a is not None:
        folium.Marker(
            location=[float(m_a["lat"]), float(m_a["lon"])],
            tooltip=folium.Tooltip(L("<b>🔴 REGION A</b><br>", "<b>🔴 BÖLGE A</b><br>") + geo_a["tooltip_html"], sticky=True),
            popup=folium.Popup(L("<b>🔴 REGION A</b><br>", "<b>🔴 BÖLGE A</b><br>") + geo_a["tooltip_html"], max_width=280),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
    if m_b is not None and geo_b is not None:
        folium.Marker(
            location=[float(m_b["lat"]), float(m_b["lon"])],
            tooltip=folium.Tooltip(L("<b>🔵 REGION B</b><br>", "<b>🔵 BÖLGE B</b><br>") + geo_b["tooltip_html"], sticky=True),
            popup=folium.Popup(L("<b>🔵 REGION B</b><br>", "<b>🔵 BÖLGE B</b><br>") + geo_b["tooltip_html"], max_width=280),
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)
    if m_a is not None and m_b is not None:
        folium.PolyLine(
            locations=[[float(m_a["lat"]), float(m_a["lon"])], [float(m_b["lat"]), float(m_b["lon"])]],
            color="#EF4444",
            weight=3.0,
            dash_array="6, 8",
            tooltip=L("Region A – Region B", "Bölge A – Bölge B"),
        ).add_to(m)
    folium.LayerControl().add_to(m)

    map_out = st_folium(m, use_container_width=True, height=440, key="nasa_map_dual", returned_objects=["last_clicked"])
    if _clicked(map_out):
        c_lat, c_lon = _clicked(map_out)
        click_pt = (c_lat, c_lon)
        if click_pt != st.session_state.get("dual_last_click"):
            st.session_state.dual_last_click = click_pt
            if m_a is None:
                idx = 0
            elif m_b is None:
                idx = 1
            else:
                idx = 0 if target_pin == 0 else 1
            st.session_state.markers_dual[idx] = {"lat": c_lat, "lon": c_lon}
            st.session_state.coord_version += 1
            st.session_state.active_case_title = None
            st.rerun()

    render_analysis_controls(geo_a or geo_b)

    if m_a is None or m_b is None:
        st.info(L("👆 **To start the two-region comparison, select both 🔴 Region A and 🔵 Region B on the map.**", "👆 **Çiftli bölge karşılaştırmasını başlatmak için harita üzerinden hem 🔴 Bölge A hem de 🔵 Bölge B noktasını seçin.**"))
        return
    if not year_span_ok():
        return

    lat_a, lon_a = float(m_a["lat"]), float(m_a["lon"])
    lat_b, lon_b = float(m_b["lat"]), float(m_b["lon"])
    start_yr, end_yr = st.session_state.year_range
    hide = st.session_state.hide_seasonality
    name_a = _region_label("🔴 A:", geo_a, lat_a, lon_a)
    name_b = _region_label("🔵 B:", geo_b, lat_b, lon_b)
    same_cell_warning({name_a: (lat_a, lon_a), name_b: (lat_b, lon_b)})

    region_results = {}
    for name, lt, ln in [(name_a, lat_a, lon_a), (name_b, lat_b, lon_b)]:
        r = fetch_and_analyze_region(lt, ln, st.session_state.selected_vars, start_yr, end_yr, hide)
        for w_msg in r["missing_warnings"]:
            st.error(f"⚠️ **{name}:** {w_msg}")
        if r["valid"]:
            region_results[name] = r
    render_fetch_status_sidebar()
    with st.expander(L("📡 Data source, actual data window and completeness (per region and variable)", "📡 Veri kaynağı, gerçek veri penceresi ve doluluk (bölge ve değişken başına)")):
        for name, r in region_results.items():
            for note in r["data_notes"]:
                st.markdown(f"- {name} — {note}")

    if len(region_results) < 2:
        st.warning(L("For a comparison, both regions must have valid data for at least one parameter.", "Karşılaştırma için her iki bölgede de en az bir parametrenin geçerli verisi olmalıdır."))
        return

    shared = build_shared_comparison(region_results, st.session_state.selected_vars, hide, end_yr)
    if not shared["valid"]:
        render_without_composite(region_results, shared, hide)
        return
    if shared["dropped_vars"]:
        st.info(
            L("ℹ️ These parameters were removed from the composite comparison because one of the regions has no data for them: ",
              "ℹ️ Şu parametreler bölgelerden birinde veri içermediği için bütünleşik kıyastan çıkarıldı: ")
            + ", ".join(OFFICIAL_VARIABLES[v]["short_label"] for v in shared["dropped_vars"])
        )

    tele_res = compare_regions_teleconnection(
        {name_a: shared["composites"][name_a], name_b: shared["composites"][name_b]}, value_col="value"
    )

    # 1. EN ÜSTTE: BÜTÜNLEŞİK ZITLIK / İLİŞKİ
    st.markdown(L("### 🌐 Composite Opposition / Relationship Chart", "### 🌐 Bütünleşik Zıtlık / İlişki Grafiği"))
    st.markdown(
        f"""
        <div class="tele-banner">
            <h4 style="margin-top:0;color:#FBBF24;">{tele_res['verdict_badge']}</h4>
            <p style="margin-bottom:0;font-size:14.5px;color:#E2E8F0;">{tele_res['explanation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(L(
        "This class is based only on the significance of the Spearman correlation between the series with seasonality and the "
        "linear trend removed (p < 0.05, corrected for autocorrelation with an effective sample size); slope signs are not part of the verdict. "
        "It is a measure of association; it is not proof of a teleconnection or of causation.",
        "Bu sınıf yalnızca mevsimselliği ve doğrusal trendi çıkarılmış seriler arasındaki Spearman korelasyonunun "
        "anlamlılığına (otokorelasyon için etkin örneklem düzeltmeli p < 0.05) dayanır; eğim işaretleri karara katılmaz. "
        "Bu bir ilişki ölçüsüdür; uzaktan bağlantı (teleconnection) veya nedensellik kanıtı değildir."
    ))
    st.caption(composite_direction_text(shared["index_description"], {n: len(c) for n, c in shared["composites"].items()}))
    st.plotly_chart(
        plot_dual_region_integrated_composite(
            name_a, name_b, shared["composites"][name_a], shared["composites"][name_b],
            shared["forecasts"][name_a], shared["forecasts"][name_b], tele_res, shared["weights_df"], hide,
        ),
        width="stretch",
        key="dual_integrated",
    )
    st.caption(" ".join(f"{n}: {t}" for n, t in shared["fc_notes"].items()) + L(" No world map is drawn for the composite index.", " Bütünleşik indeks için dünya haritası çizilmez."))
    st.info(pick(FORECAST_NOTE))
    render_weights_and_vif_section(shared["weights_df"], shared["vif_warnings"], shared.get("weight_method", ""))
    col_a, col_b = st.columns(2)
    with col_a:
        render_mk_significance_card(name_a, tele_res["region_stats"][name_a])
    with col_b:
        render_mk_significance_card(name_b, tele_res["region_stats"][name_b])

    # 2. ALTTA: PARAMETRE BAZLI AYRIK KIYAS
    st.markdown("---")
    render_per_parameter_comparisons(region_results, shared["common_vars"], shared["weights_df"], hide)


# ---------------------------------------------------------------------------
# ROTA 5: `/analyze/multi` — ÇOKLU BÖLGE KARŞILAŞTIRMASI (MAKSİMUM 10 BÖLGE)
# ---------------------------------------------------------------------------
def render_multi_analyze_route() -> None:
    st.subheader(L("🌐 Multi-Region Opposition / Relationship Analysis and Statistical Extension (/analyze/multi — Max. 10 Regions)", "🌐 Çoklu Bölge Zıtlık / İlişki Analizi ve İstatistiksel Uzatma (/analyze/multi — Maks. 10 Bölge)"))

    markers = st.session_state.markers_multi
    n_markers = len(markers)
    geo_list = [reverse_geocode_and_detect_biome(mk["lat"], mk["lon"]) for mk in markers]
    cver = st.session_state.coord_version

    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns([1.8, 0.65, 0.65, 0.7, 0.8])
    with b_col1:
        st.markdown(L(f"#### 📌 Selected Regions (`{n_markers} / 10`)", f"#### 📌 Seçili Bölgeler (`{n_markers} / 10`)"))
        st.caption(L("Add regions by clicking the map or entering coordinates on the right.", "Haritaya tıklayarak veya sağdan koordinat girerek istediğiniz bölgeleri ekleyin."))
    with b_col2:
        add_lat = st.number_input(L("Latitude:", "Enlem:"), -85.0, 85.0, value=0.0, step=0.5, key=f"multi_in_lat_{cver}")
    with b_col3:
        add_lon = st.number_input(L("Longitude:", "Boylam:"), -180.0, 180.0, value=0.0, step=0.5, key=f"multi_in_lon_{cver}")
    with b_col4:
        st.write("")
        if st.button(L("➕ Add Region", "➕ Bölge Ekle"), disabled=(n_markers >= 10), width="stretch", key="add_multi_btn"):
            st.session_state.markers_multi.append({"lat": round(float(add_lat), 2), "lon": round(float(add_lon), 2)})
            st.session_state.coord_version += 1
            st.rerun()
    with b_col5:
        st.write("")
        if st.button(L("🗑️ Clear All Selections", "🗑️ Tüm Seçimleri Temizle"), disabled=(n_markers == 0), width="stretch", key="clr_multi_all"):
            st.session_state.markers_multi = []
            st.session_state.multi_last_click = None
            st.session_state.coord_version += 1
            st.rerun()

    m = _create_base_folium_map(25.0, 15.0, zoom=2)
    folium_colors = ["red", "blue", "green", "orange", "purple", "pink", "darkblue", "lightgreen", "beige", "cadetblue"]
    for idx, (mk, g) in enumerate(zip(markers, geo_list)):
        folium.Marker(
            location=[mk["lat"], mk["lon"]],
            tooltip=folium.Tooltip(L(f"<b>Region #{idx + 1}</b><br>", f"<b>Bölge #{idx + 1}</b><br>") + g["tooltip_html"], sticky=True),
            popup=folium.Popup(L(f"<b>Region #{idx + 1}</b><br>", f"<b>Bölge #{idx + 1}</b><br>") + g["tooltip_html"], max_width=280),
            icon=folium.Icon(color=folium_colors[idx % len(folium_colors)], icon="info-sign"),
        ).add_to(m)
    folium.LayerControl().add_to(m)

    map_out = st_folium(m, use_container_width=True, height=440, key="nasa_map_multi", returned_objects=["last_clicked"])
    if _clicked(map_out):
        c_lat, c_lon = _clicked(map_out)
        click = (c_lat, c_lon)
        if click != st.session_state.get("multi_last_click"):
            st.session_state.multi_last_click = click
            exists = any(abs(c_lat - x["lat"]) < 0.05 and abs(c_lon - x["lon"]) < 0.05 for x in markers)
            if not exists:
                if len(markers) < 10:
                    st.session_state.markers_multi.append({"lat": c_lat, "lon": c_lon})
                    st.rerun()
                else:
                    st.warning(L("🔒 The maximum of 10 regions has been reached. Remove a region from the list to add a new one.", "🔒 Maksimum 10 bölge sınırına ulaşıldı. Yeni eklemek için listeden bir bölgeyi silin."))

    # Seçili bölgeleri listele ve HER BİRİNİN yanına bağımsız '❌ Kaldır' butonu koy (hiçbir bölge kilitli kalmaz)
    if markers:
        st.markdown(L("##### 📍 Added Regions (you can remove any region individually):", "##### 📍 Eklenen Bölgeler (İstediğiniz bölgeyi tek tek kaldırabilirsiniz):"))
        for idx, (mk, g) in enumerate(zip(markers, geo_list)):
            r_col1, r_col2 = st.columns([4.2, 0.8])
            with r_col1:
                st.markdown(
                    f"**#{idx + 1}** — {g['country']} — {g['city_region']} (`{mk['lat']:.2f}°, {mk['lon']:.2f}°`) | *{g['biome_name']}*"
                    f"  \n<small>{g['power_cell_note']}</small>", unsafe_allow_html=True,
                )
                if g["mixed_cell_note"]:
                    st.caption(g["mixed_cell_note"])
            with r_col2:
                if st.button(L(f"❌ Remove #{idx + 1}", f"❌ Kaldır #{idx + 1}"), key=f"rm_multi_item_{idx}", width="stretch"):
                    st.session_state.markers_multi.pop(idx)
                    st.session_state.multi_last_click = None
                    st.rerun()

    if markers:
        st.caption(L(f"Region names: {OSM_ATTRIBUTION} (Nominatim; if unreachable, 'Name unavailable').", f"Bölge adları: {OSM_ATTRIBUTION} (Nominatim; erişilemezse 'Ad alınamadı')."))
    render_analysis_controls(geo_list[0] if geo_list else None)

    if len(markers) < 2:
        st.info(L(
            f"👆 **To start the multi-region comparison, select at least 2 regions by clicking the map "
            f"(currently selected: `{len(markers)} / 10`).**",
            f"👆 **Çoklu bölge karşılaştırmasını başlatmak için harita üzerine tıklayarak en az 2 bölge seçin "
            f"(Şu an seçili: `{len(markers)} / 10`).**"
        ))
        return
    if not year_span_ok():
        return

    start_yr, end_yr = st.session_state.year_range
    hide = st.session_state.hide_seasonality
    same_cell_warning({
        _region_label(f"#{i + 1}", g, mk["lat"], mk["lon"]): (mk["lat"], mk["lon"]) for i, (mk, g) in enumerate(zip(markers, geo_list))
    })

    region_results: Dict[str, Dict[str, Any]] = {}
    region_geo: Dict[str, Dict[str, Any]] = {}
    for idx, (mk, g) in enumerate(zip(markers, geo_list)):
        r_label = _region_label(f"#{idx + 1}", g, mk["lat"], mk["lon"])
        r = fetch_and_analyze_region(mk["lat"], mk["lon"], st.session_state.selected_vars, start_yr, end_yr, hide)
        for mw in r["missing_warnings"]:
            st.error(f"⚠️ **{r_label}:** {mw}")
        if r["valid"]:
            region_results[r_label] = r
            region_geo[r_label] = g
    render_fetch_status_sidebar()
    with st.expander(L("📡 Data source, actual data window and completeness (per region and variable)", "📡 Veri kaynağı, gerçek veri penceresi ve doluluk (bölge ve değişken başına)")):
        for name, r in region_results.items():
            for note in r["data_notes"]:
                st.markdown(f"- {name} — {note}")

    if len(region_results) < 2:
        st.warning(L("For a comparison, at least 2 regions must have valid data.", "Karşılaştırma için en az 2 bölgede geçerli veri bulunmalıdır."))
        return

    shared = build_shared_comparison(region_results, st.session_state.selected_vars, hide, end_yr)
    if not shared["valid"]:
        render_without_composite(region_results, shared, hide)
        return
    if shared["dropped_vars"]:
        st.info(
            L("ℹ️ These parameters were removed from the comparison because some regions have no data for them: ",
              "ℹ️ Şu parametreler bazı bölgelerde veri içermediği için kıyastan çıkarıldı: ")
            + ", ".join(OFFICIAL_VARIABLES[v]["short_label"] for v in shared["dropped_vars"])
        )

    tele_multi = compare_regions_teleconnection(shared["composites"], value_col="value")

    # 1. EN ÜSTTE: BÜTÜNLEŞİK ZITLIK / İLİŞKİ
    st.markdown(L("### 🌐 Composite Opposition / Relationship Chart", "### 🌐 Bütünleşik Zıtlık / İlişki Grafiği"))
    fig_ts, fig_heat = plot_multi_region_comparison(shared["composites"], shared["forecasts"], tele_multi)
    st.caption(composite_direction_text(shared["index_description"], {n: len(c) for n, c in shared["composites"].items()}))
    st.plotly_chart(fig_ts, width="stretch", key="multi_integrated")
    st.caption(" ".join(f"{n}: {t}" for n, t in shared["fc_notes"].items()) + L(" No world map is drawn for the composite index.", " Bütünleşik indeks için dünya haritası çizilmez."))
    st.info(pick(FORECAST_NOTE))
    st.caption(L(
        "Region-pair classes are based on the significance of the Spearman correlation of the series with seasonality and the linear trend removed "
        "(p < 0.05, effective-sample-size corrected); they are not proof of a teleconnection or of causation.",
        "Bölge çifti sınıfları, mevsimselliği ve doğrusal trendi çıkarılmış serilerin Spearman korelasyonunun anlamlılığına "
        "(etkin örneklem düzeltmeli p < 0.05) dayanır; uzaktan bağlantı (teleconnection) veya nedensellik kanıtı değildir."
    ))

    with st.expander(L("📐 Composite Index — Statistical Details by Region (Sen's Slope, Kendall τ, 95% CI, p-value and Comment)", "📐 Bütünleşik İndeks — Bölgelerin İstatistik Detayı (Sen's Slope, Kendall τ, %95 GA, p-Değeri ve Yorum)"), expanded=True):
        for name in shared["composites"]:
            render_mk_significance_card(name, tele_multi["region_stats"][name])

    c_heat, c_tbl = st.columns([1.15, 1.35])
    with c_heat:
        st.plotly_chart(fig_heat, width="stretch", key="multi_heat")
    with c_tbl:
        summary_rows = []
        for name in shared["composites"]:
            mk_stat = tele_multi["region_stats"][name]
            summary_rows.append(
                {
                    L("Region", "Bölge"): name,
                    L("Surface", "Yüzey"): region_geo[name]["biome_name"],
                    L("Sen's Slope (Annual)", "Sen's Slope (Yıllık)"): mk_stat["sen_slope_annual"],
                    "Kendall τ": mk_stat["kendall_tau"],
                    L("p-value (dependence-corrected)", "p-Değeri (bağımlılık düzeltmeli)"): mk_stat["p_value_display"],
                    L("Verdict reliable?", "Karar güvenilir mi?"): L("yes", "evet") if mk_stat.get("reliable_verdict", True) else L("no (short series)", "hayır (seri kısa)"),
                    L("Significance", "Anlamlılık"): mk_stat["badge_text"],
                }
            )
        st.markdown(L("#### 📋 Composite Trend Summary", "#### 📋 Bütünleşik Trend Özeti"))
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
        st.markdown(L("#### ⚡ Region Pairs (Composite Index)", "#### ⚡ Bölge Çiftleri (Bütünleşik İndeks)"))
        st.dataframe(tele_multi["pairwise_df"], width="stretch", hide_index=True)

    render_weights_and_vif_section(shared["weights_df"], shared["vif_warnings"], shared.get("weight_method", ""))

    # 2. ALTTA: PARAMETRE BAZLI AYRIK KIYAS
    st.markdown("---")
    render_per_parameter_comparisons(region_results, shared["common_vars"], shared["weights_df"], hide)


# ---------------------------------------------------------------------------
# ANA ROTA YÖNLENDİRİCİSİ (MAIN ROUTER DISPATCHER)
# ---------------------------------------------------------------------------
render_top_navbar()

current_route = st.session_state.route
if current_route == "/":
    render_landing_route()
elif current_route == "/explore":
    render_explore_route()
elif current_route == "/analyze/single":
    render_single_analyze_route()
elif current_route == "/analyze/dual":
    render_dual_analyze_route()
elif current_route == "/analyze/multi":
    render_multi_analyze_route()
elif current_route == "/parameters":
    render_parameters_route()
else:
    render_dual_analyze_route()
