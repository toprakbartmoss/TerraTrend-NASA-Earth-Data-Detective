"""
NASA SPACE APPS CHALLENGE: "EARTH DATA DETECTIVE"
Çok Sayfalı (Multi-Route) İklim Trendi Analiz, Teleconnection Keşif ve 5 Yıllık SARIMA Projeksiyon Platformu (app.py)

Rotalar (Routes):
- /                : Tanıtım Ekranı (Landing Page), Canlı Harita Önizlemesi ve Bilimsel Metodoloji Özeti
- /explore         : Ana Menü (Tekli / Çiftli / Çoklu Bölge Modları), İnteraktif Vaka Haritası ve 3 Hazır Vaka Merkezi
- /analyze/single  : Tekli Bölge Analizi (Haritada Tek İkon, STL, Hamed-Rao Mann-Kendall, 5 Yıllık SARIMA)
- /analyze/dual    : Çiftli Bölge Karşılaştırması (2 Çizgili Bütünleşik Grafik + Tekil Grafikler + Teleconnection)
- /analyze/multi   : Çoklu Bölge Karşılaştırması (Maksimum 10 Bölge, Korelasyon Isı Haritası, Projeksiyon)
- /parameters      : 10 Parametrelik Sistem Havuzu, Sabit IPCC AR6 Ağırlıkları ve Öğrenci Rehberi
"""

import importlib
from typing import Any, Dict, List, Optional
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import data_loader as _dl_mod
import statistics_engine as _se_mod
import visualizer as _vis_mod

if (
    "T2M" not in getattr(_dl_mod, "OFFICIAL_VARIABLES", {})
    or "water_cycle" not in getattr(_dl_mod, "DEMO_SCENARIOS", {})
    or not hasattr(_dl_mod, "_fetch_nasa_merra2_point_bundle")
    or not hasattr(_dl_mod, "_nominatim_reverse_lookup")
    or not hasattr(_vis_mod, "plot_world_1980_vs_present_heatmap")
):
    importlib.reload(_dl_mod)
    importlib.reload(_se_mod)
    importlib.reload(_vis_mod)

from data_loader import (
    BIOME_DEFINITIONS,
    DEMO_SCENARIOS,
    OFFICIAL_VARIABLES,
    build_giovanni_direct_url,
    fetch_hybrid_climate_series,
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
from visualizer import (
    PARAM_PALETTE,
    REGION_PALETTE,
    plot_dual_region_integrated_composite,
    plot_multi_region_comparison,
    plot_parameter_regions_comparison,
    plot_single_parameter_chart,
    plot_single_region_composite_and_forecast,
    plot_world_1980_vs_present_heatmap,
)

MIN_YEAR_SPAN = 5  # STL (12 aylık periyot) + Mann-Kendall + SARIMA için anlamlı minimum: 5 yıl = 60 ay

# ---------------------------------------------------------------------------
# SAYFA YAPILANDIRMASI VE HARİTA IFRAME GÖRÜNÜRLÜK STİLİ
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NASA Earth Data Detective | İklim Trendi & Projeksiyon Platformu",
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
    st.session_state.year_range = (1980, 2025)

if "hybrid_enabled" not in st.session_state:
    st.session_state.hybrid_enabled = True

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
        name="Uydu Görünümü (Esri)",
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
    st.session_state.year_range = (int(sc["start_year"]), int(sc["end_year"]))
    st.session_state.hybrid_enabled = bool(sc["hybrid_enabled"])
    st.session_state.hide_seasonality = bool(sc["hide_seasonality"])
    st.session_state.active_case_title = sc["title"]

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
        "<span style='font-size:14px;color:#38BDF8;font-weight:normal;'>| Resmi Çoklu Kaynak İklim Trendi, Teleconnection & 5 Yıllık SARIMA Projeksiyon Platformu</span>",
        unsafe_allow_html=True,
    )
    nav_cols = st.columns(6)
    routes_info = [
        ("/", "🏠 Tanıtım (/)"),
        ("/explore", "🧭 Keşif & Vakalar (/explore)"),
        ("/analyze/single", "📍 Tekli Bölge (/single)"),
        ("/analyze/dual", "⚡ Çiftli Bölge (/dual)"),
        ("/analyze/multi", "🌐 Çoklu Bölge (/multi)"),
        ("/parameters", "📚 10 Parametre (/parameters)"),
    ]
    for col, (r_path, r_label) in zip(nav_cols, routes_info):
        is_active = st.session_state.route == r_path
        if col.button(
            r_label,
            key=f"nav_btn_{r_path}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            navigate_to(r_path)
            st.rerun()
    st.markdown("---")


# ---------------------------------------------------------------------------
# ROTA 1: `/` — TANITIM EKRANI (LANDING PAGE + CANLI KÜRESEL HARİTA)
# ---------------------------------------------------------------------------
def render_landing_route() -> None:
    st.markdown(
        """
        <div class="route-card">
            <h2 style="margin-top:0;color:#38BDF8;">🌍 NASA Earth Data Detective: Küresel İklim Sinyallerini Gürültüden Ayırın</h2>
            <p style="font-size:16px;line-height:1.6;color:#E2E8F0;">
                Bu platform, <b>NASA Space Apps Challenge</b> kapsamında Dünya sistemi verilerini yalnızca resmi uydu ve
                yeniden analiz (reanalysis) arşivlerinden çekerek <b>1980–2025 (46 Yıl)</b> dönemindeki iklim değişimlerini
                mevsimsellikten arındırır, istatistiksel anlamlılığını test eder, bölgeler arası zıt iklim tepkilerini
                (<b>Teleconnection / Dipole</b>) kanıtlar ve <b>2026–2030 (60 Ay)</b> için gelecek projeksiyonu üretir.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("🚀 Keşfetmeye Başla (/explore)", type="primary", use_container_width=True):
            navigate_to("/explore")
            st.rerun()
    with btn_col2:
        if st.button("📍 Tekli Bölge Haritası (/analyze/single)", use_container_width=True):
            navigate_to("/analyze/single")
            st.rerun()
    with btn_col3:
        if st.button("⚡ Çiftli Bölge Haritası (/analyze/dual)", use_container_width=True):
            navigate_to("/analyze/dual")
            st.rerun()
    with btn_col4:
        if st.button("📚 10 Parametre Rehberi (/parameters)", use_container_width=True):
            navigate_to("/parameters")
            st.rerun()

    st.markdown("#### 🗺️ Küresel İklim Gözlem Haritası (Haritadan Bir Noktaya Tıklayarak Doğrudan Tekli Analize Geçebilirsiniz)")
    m_land = _create_base_folium_map(25.0, 20.0, zoom=2)
    # 3 Hazır Vaka noktasını önizleme olarak göster
    preview_pts = [
        (45.00, 59.60, "red", "Kombinasyon 1 (Kuraklık): Aral Gölü Havzası"),
        (38.83, 33.33, "orange", "Kombinasyon 1b (Kuraklık): Tuz Gölü Kapalı Havzası"),
        (39.93, 32.85, "green", "Kombinasyon 2 (Su Döngüsü): İç Anadolu Havzası"),
        (37.98, 23.72, "blue", "Kombinasyon 2 (Su Döngüsü): Ege / Doğu Akdeniz Havzası"),
        (78.22, 15.63, "purple", "Kombinasyon 3 (Kutup/Enerji): Svalbard / Arktik Deniz Buzu"),
    ]
    for plat, plon, pcol, ptitle in preview_pts:
        g_info = reverse_geocode_and_detect_biome(plat, plon)
        folium.Marker(
            location=[plat, plon],
            tooltip=folium.Tooltip(f"<b>{ptitle}</b><br>" + g_info["tooltip_html"], sticky=True),
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
    if land_click and land_click.get("last_clicked"):
        c_lat = round(float(land_click["last_clicked"]["lat"]), 2)
        c_lon = round(float(land_click["last_clicked"]["lng"]), 2)
        st.session_state.marker_single = {"lat": c_lat, "lon": c_lon}
        st.session_state.coord_version += 1
        navigate_to("/analyze/single")
        st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="route-card">
                <h4>📡 1. NASA GMAO MERRA-2 v5.12.4 Omurgası (1980–2025)</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>M2TMNXSLV v5.12.4:</b> <code>T2M</code> (2 m hava sıcaklığı) & <code>TQV</code> (Toplam çökelebilir su buharı)</li>
                    <li><b>M2TMNXFLX v5.12.4:</b> <code>PRECTOTCORR</code> (Bias-corrected toplam yüzey yağışı)</li>
                    <li><b>M2TMNXOCN v5.12.4:</b> <code>TSKINWTR</code> (Açık su cilt sıcaklığı) & <code>FRSEAICE</code> (Deniz buzu oranı)</li>
                    <li><b>M2TMNXLND v5.12.4:</b> <code>GWETROOT</code> (Kök bölgesi toprak nemi), <code>EVLAND</code> & <code>RUNOFF</code></li>
                    <li><b>M2TMNXRAD v5.12.4:</b> <code>ALBEDO</code> (Yüzey albedosu) & <code>SWGDN</code> (Kısa dalga güneş ışınımı)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="route-card">
                <h4>📐 2. Sabit Nesnel Ağırlıklandırma (Önem / 10 + CRITIC + VIF)</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>10 Temel Değişken (10.0 → 8.3):</b> Ocak 1980'den günümüze kesintisiz aylık 2B tanı koleksiyonları.</li>
                    <li><b>Maksimum 4 Seçim Kısıtı:</b> 4 parametre seçildiğinde kalan 6 parametre otomatik kilitlenir.</li>
                    <li><b>Kullanıcı Müdahalesine Kapalı:</b> Ağırlıklar PDF'teki <i>Önem / 10</i> baz puanları + CRITIC algoritmasıyla hesaplanır.</li>
                    <li><b>Kullanım Sınırları & VIF > 5:</b> Kara/okyanus/kutup hücre sınırları ve çoklu doğrusallık otomatik denetlenir.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="route-card">
                <h4>📈 3. STL, Hamed-Rao Mann-Kendall & SARIMA</h4>
                <ul style="font-size:14px;color:#CBD5E1;">
                    <li><b>STL Ayrıştırma:</b> <code>Trend + Seasonal + Residual</code> ayrıştırması ve Anomali Modu.</li>
                    <li><b>Hamed-Rao Modified Mann-Kendall:</b> Otokorelasyon düzeltmeli $p$-değeri, Kendall $\\tau$ ve Sen's Slope (%95 GA).</li>
                    <li><b>Ters Yönlü İklim Tepkisi:</b> Çapraz korelasyon ve $\\Delta\\text{Slope}$ ile Teleconnection tespiti.</li>
                    <li><b>5 Yıllık İklim Projeksiyonu:</b> SARIMA ile 60 aylık (2026–2030) tahmin eğrisi ve %95 güven bandı.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# ROTA 2: `/explore` — ANA MENÜ VE 3 HAZIR VAKA ÇALIŞMASI MERKEZİ
# ---------------------------------------------------------------------------
def render_explore_route() -> None:
    st.subheader("🧭 Analiz Modu Seçimi ve Vaka Merkezi (/explore)")
    st.caption(
        "İncelemek istediğiniz coğrafi kapsama göre aşağıdaki 3 analiz modundan birini seçin, "
        "3 Hazır Vaka Çalışmasından birini başlatın veya haritadan doğrudan nokta seçin."
    )

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            """
            <div class="route-card">
                <h4>📍 1. Tekli Bölge Seçimi</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Harita üzerinde <b>tek bir nokta/ikon</b> seçerek 1980–2025 STL trend ayrıştırmasını,
                    Hamed-Rao Mann-Kendall istatistiksel anlamlılığını ve 2026–2030 SARIMA projeksiyonunu inceleyin.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📍 Tekli Bölge Analizine Git (/analyze/single)", type="primary", use_container_width=True):
            navigate_to("/analyze/single")
            st.rerun()

    with m_col2:
        st.markdown(
            """
            <div class="route-card">
                <h4>⚡ 2. Çiftli Bölge Seçimi (Teleconnection)</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Haritada <b>Bölge A ve Bölge B</b>'yi seçerek iki çizgili bütünleşik karşılaştırma grafiğini,
                    Çapraz Korelasyonu, $\\Delta\\text{Slope}$ eğim farkını ve ters yönlü iklim tepkisini kanıtlayın.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("⚡ Çiftli Bölge Analizine Git (/analyze/dual)", type="primary", use_container_width=True):
            navigate_to("/analyze/dual")
            st.rerun()

    with m_col3:
        st.markdown(
            """
            <div class="route-card">
                <h4>🌐 3. Çoklu Bölge Seçimi (Maks. 10 Bölge)</h4>
                <p style="font-size:14px;color:#CBD5E1;">
                    Dünya haritası üzerinde <b>10 bölgeye kadar</b> nokta işaretleyerek kıtalar arası çoklu trend
                    karşılaştırması, $N \\times N$ korelasyon ısı haritası ve 5 yıllık projeksiyon üretin.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🌐 Çoklu Bölge Analizine Git (/analyze/multi)", type="primary", use_container_width=True):
            navigate_to("/analyze/multi")
            st.rerun()

    st.markdown("---")
    st.subheader("🧪 3 Hazır Vaka Çalışması (Demonstration Scenarios — 1980–2025)")

    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        sc1 = DEMO_SCENARIOS["aral_tuz"]
        st.markdown(
            f"""
            <div class="case-card">
                <h5>{sc1['title']}</h5>
                <p style="font-size:13.5px;color:#CBD5E1;">{sc1['summary']}</p>
                <p style="font-size:12px;color:#38BDF8;"><b>Parametreler (Kuraklık Kombinasyonu):</b> <code>T2M</code> + <code>PRECTOTCORR</code> + <code>GWETROOT</code> + <code>EVLAND</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1a, b1b = st.columns(2)
        if b1a.button("🏜️ Aral Gölü Vakasını Yükle", key="case_aral", use_container_width=True):
            load_case_study("aral_tuz", use_tuz_golu=False)
            st.rerun()
        if b1b.button("🇹🇷 Tuz Gölü Vakasını Yükle", key="case_tuz", use_container_width=True):
            load_case_study("aral_tuz", use_tuz_golu=True)
            st.rerun()

    with sc_col2:
        sc2 = DEMO_SCENARIOS["water_cycle"]
        st.markdown(
            f"""
            <div class="case-card">
                <h5>{sc2['title']}</h5>
                <p style="font-size:13.5px;color:#CBD5E1;">{sc2['summary']}</p>
                <p style="font-size:12px;color:#38BDF8;"><b>Parametreler (Su Döngüsü Kombinasyonu):</b> <code>PRECTOTCORR</code> + <code>GWETROOT</code> + <code>RUNOFF</code> + <code>TQV</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("💧 Su Döngüsü Kombinasyonunu Yükle (/dual)", key="case_water_cycle", use_container_width=True):
            load_case_study("water_cycle")
            st.rerun()

    with sc_col3:
        sc3 = DEMO_SCENARIOS["arctic_albedo"]
        st.markdown(
            f"""
            <div class="case-card">
                <h5>{sc3['title']}</h5>
                <p style="font-size:13.5px;color:#CBD5E1;">{sc3['summary']}</p>
                <p style="font-size:12px;color:#38BDF8;"><b>Parametreler (Kutup / Enerji Kombinasyonu):</b> <code>TSKINWTR</code> + <code>FRSEAICE</code> + <code>ALBEDO</code> + <code>SWGDN</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("❄️ Arktik Albedo Vakasını Yükle (/single)", key="case_arctic", use_container_width=True):
            load_case_study("arctic_albedo")
            st.rerun()

    st.markdown("---")
    p_col1, p_col2 = st.columns([2.5, 1.5])
    with p_col1:
        st.markdown("#### 🔬 10 Temel MERRA-2 Değişkeni (1980+) ve Nesnel Ağırlıklandırma Metodolojisi")
        st.caption(
            "NASA GMAO MERRA-2 aylık 2B tanı koleksiyonlarındaki 10 temel değişkeni (`T2M`, `PRECTOTCORR`, `TSKINWTR`, "
            "`GWETROOT`, `EVLAND`, `RUNOFF`, `TQV`, `FRSEAICE`, `ALBEDO`, `SWGDN`), önem puanlarını (10.0 → 8.3) "
            "ve kullanım sınırlarını incelemek için rehber sayfasına geçebilirsiniz."
        )
    with p_col2:
        if st.button("📚 Parametre ve Önem Sırası Rehberini Aç (/parameters)", use_container_width=True):
            navigate_to("/parameters")
            st.rerun()


# ---------------------------------------------------------------------------
# ROTA 6: `/parameters` — 10 PARAMETRELİK SİSTEM HAVUZU VE ÖĞRENCİ REHBERİ
# ---------------------------------------------------------------------------
def render_parameters_route() -> None:
    st.subheader("📚 1980'den Beri İzlenebilen 10 Temel MERRA-2 Değişkeni ve Rehber (/parameters)")
    st.markdown(
        """
        > **Veri Omurgası ve Nesnel Ağırlıklandırma Kuralı (Kullanıcı Müdahalesine Kapalı):**
        > Bu listedeki 10 değişkenin tamamı **NASA GMAO MERRA-2** aylık 2B tanı koleksiyonlarından (`M2TMNXSLV`, `M2TMNXFLX`,
        > `M2TMNXOCN`, `M2TMNXLND`, `M2TMNXRAD` v5.12.4) seçilmiştir ve **Ocak 1980'den günümüze** kesintisiz aylık seri üretir.
        > Seçilen 1–4 parametrenin ağırlıkları, **Önem / 10** baz puanları (`10.0` → `8.3`), **CRITIC** bilgi içeriği ve
        > **VIF > 5 (Çoklu Doğrusallık)** denetimi ile arka planda otomatik hesaplanarak toplamı `1.00 (%100)` olacak şekilde normalize edilir.
        """
    )

    table_rows = []
    for code, meta in sorted(OFFICIAL_VARIABLES.items(), key=lambda x: x[1]["rank"]):
        table_rows.append(
            {
                "Sıra": f"#{meta['rank']}",
                "Alan Kodu": meta.get("short_name", code),
                "Parametre": meta["short_label"],
                "MERRA-2 Koleksiyonu": meta.get("collection", meta["reanalysis_dataset"]),
                "Önem / 10": f"{meta.get('importance_score_10', meta['base_weight'] * 10):.1f} / 10",
                "Baz Ağırlık": f"{meta['base_weight']:.2f}",
                "Birim": meta["unit"],
                "Geçerli Biyomlar": ", ".join(meta["valid_biomes"]),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.info(
        "💡 **Önerilen 4'lü Analiz Kombinasyonları (Maks. 4 Seçim Kuralı İçin):**\n"
        "- **Kuraklık Analizi:** `T2M` + `PRECTOTCORR` + `GWETROOT` + `EVLAND`\n"
        "- **Kutup / Enerji Analizi:** `TSKINWTR` + `FRSEAICE` + `ALBEDO` + `SWGDN`\n"
        "- **Su Döngüsü Analizi:** `PRECTOTCORR` + `GWETROOT` + `RUNOFF` + `TQV`"
    )

    st.markdown("### 🎓 Değişken Açıklamaları, Neden Kritik Oldukları ve Kullanım Sınırları")
    for code, meta in sorted(OFFICIAL_VARIABLES.items(), key=lambda x: x[1]["rank"]):
        score_10 = meta.get("importance_score_10", meta["base_weight"] * 10)
        coll = meta.get("collection", meta["reanalysis_dataset"])
        with st.expander(
            f"{meta['label']} — Önem: {score_10:.1f}/10 (Baz Ağırlık: {meta['base_weight']:.2f}) | Koleksiyon: {coll}",
            expanded=(meta["rank"] <= 3),
        ):
            st.markdown(f"**Neden Kritik ve Nasıl Kullanılır?** {meta['student_explanation']}")
            st.markdown(
                f"- **Alan Kodu (`Short Name`):** `{code}`\n"
                f"- **MERRA-2 Koleksiyonu:** `{coll}` (Başlangıç: Ocak 1980)\n"
                f"- **Tamamlayıcı Uydu / İstasyon Serisi:** `{meta['satellite_dataset']}`\n"
                f"- **Doğrudan Etkileşimli (Korelasyonlu) Değişkenler:** `{', '.join(meta['correlated_with'])}`\n"
                f"- **Resmi Veri Portalı:** [{meta['platform']}]({meta['platform_url']})"
            )

    st.markdown("---")
    st.markdown("### 📐 İstatistiksel Metrikler ve Trend Analizi Göstergeleri Rehberi")
    st.markdown(
        "Analiz sayfalarındaki **İstatistik Detayı** kartlarında yer alan parametrik olmayan (non-parametrik) "
        "trend göstergelerinin bilimsel anlamları, matematiksel tanımları ve yükselme/azalma seviyelerinin nasıl yorumlanacağı aşağıda açıklanmıştır:"
    )

    stat_summary_rows = [
        {
            "İstatistiksel Gösterge": "Sen's Slope (Yıllık / 10 Yıllık Hız)",
            "Ne Ölçer?": "Birim zamandaki ortanca (medyan) fiziksel değişim hızı ve yönü",
            "Sayısal Yorum Kriteri": "> 0: Yükseliş | < 0: Azalış | ≈ 0: Durağan (Aykırı uç değerlere dirençlidir)",
        },
        {
            "İstatistiksel Gösterge": "Kendall's Tau (τ) ve S İstatistiği",
            "Ne Ölçer?": "Zaman serisindeki monotonik artış veya azalışın tutarlılığı (-1 ile +1 arası)",
            "Sayısal Yorum Kriteri": "|τ| ≥ 0.45: Çok güçlü | 0.25–0.45: Belirgin | 0.12–0.25: Hafif-orta | < 0.12: Zayıf/Dalgalı",
        },
        {
            "İstatistiksel Gösterge": "%95 Güven Aralığı (Confidence Interval)",
            "Ne Ölçer?": "Gerçek yıllık değişim eğiminin %95 olasılıkla bulunduğu alt ve üst sınır",
            "Sayısal Yorum Kriteri": "Alt ve üst sınır aynı işaretliyse (0'ı kesmiyorsa) trend yönü istatistiksel olarak kesindir",
        },
        {
            "İstatistiksel Gösterge": "Hamed-Rao Z-Skoru ve p-Değeri",
            "Ne Ölçer?": "Gözlenen trendin tesadüfi (rastgele gürültü) olma olasılığı",
            "Sayısal Yorum Kriteri": "p < 0.001: Çok yüksek anlamlılık | p < 0.05: Anlamlı trend | p ≥ 0.05: Rastgele salınım",
        },
        {
            "İstatistiksel Gösterge": "Otokorelasyon Çarpanı (xV*/V)",
            "Ne Ölçer?": "Ardışık aylar arasındaki hafıza (bağımlılık) nedeniyle yapılan varyans düzeltmesi",
            "Sayısal Yorum Kriteri": "> 1.00 olduğunda sahte anlamlılığı önlemek için p-değeri Hamed-Rao yöntemiyle sıkılaştırılır",
        },
        {
            "İstatistiksel Gösterge": "Spearman ρ ve ΔSlope (Bölgeler Arası)",
            "Ne Ölçer?": "İki veya daha fazla bölge arasındaki eş-zamanlı ilişki ve yıllık hız farkı",
            "Sayısal Yorum Kriteri": "ρ ≤ -0.35: Ters yönlü (Teleconnection/Zıtlık) | ρ ≥ +0.40: Eş yönlü bölgesel etki",
        },
    ]
    st.dataframe(pd.DataFrame(stat_summary_rows), use_container_width=True, hide_index=True)

    with st.expander("1️⃣ Sen's Slope (Theil–Sen Eğim Tahmincisi — Yıllık ve 10 Yıllık Değişim Hızı)", expanded=True):
        st.markdown(
            "- **Tanım:** Zaman serisindeki tüm veri çiftleri $(t_i, t_j)$ arasındaki eğimlerin $\\frac{x_j - x_i}{t_j - t_i}$ **medyanı (ortancası)** alınarak hesaplanan yıllık değişim hızıdır.\n"
            "- **Neden Sıradan Doğrusal Regresyon Değil?** İklim verilerinde tek bir aşırı sıcak/kurak yıl (ör. şiddetli El Niño yılı) klasik en küçük kareler (OLS) eğimini yapay olarak saptırabilir. **Sen's Slope** aykırı değerlerden etkilenmez (robust).\n"
            "- **Nasıl Yorumlanır?**\n"
            "  - **Pozitif Değer (`+`):** Değişkenin yıldan yıla yükseldiğini gösterir (ör. `+0.035 °C/yıl` veya `+0.35 °C / 10 yıl` ısınma).\n"
            "  - **Negatif Değer (`-`):** Değişkenin yıldan yıla azaldığını gösterir (ör. `-0.004 m³/m³/yıl` kök bölgesi toprak nemi kaybı).\n"
            "  - **10 Yıllık Hız (`/10 yıl`):** İklim biliminde (IPCC raporlarında) değişim hızı on yıllık (dekadal) ölçekte okunur; yıllık eğimin 10 ile çarpımıdır."
        )

    with st.expander("2️⃣ Kendall's Tau (τ) Katsayısı ve Mann–Kendall S İstatistiği", expanded=True):
        st.markdown(
            "- **Mann–Kendall `S` İstatistiği:** Zaman ilerledikçe artan veri çiftlerinin sayısından azalan veri çiftlerinin sayısının çıkarılmasıyla elde edilen net yön skorudur ($S > 0$ artış, $S < 0$ azalış).\n"
            "- **Kendall's Tau ($\\tau$):** `S` istatistiğinin toplam çift sayısına bölünmesiyle elde edilen `-1.00` ile `+1.00` arasındaki standart güç katsayısıdır.\n"
            "- **Yükselme / Azalma Seviyesi Tablosu:**\n"
            "  - $|\\tau| \\ge 0.45$: **Çok Güçlü ve İstikrarlı** yükselme veya azalma.\n"
            "  - $0.25 \\le |\\tau| < 0.45$: **Belirgin (Orta–Güçlü)** yükselme veya azalma.\n"
            "  - $0.12 \\le |\\tau| < 0.25$: **Hafif–Orta Düzeyde** kademeli eğilim.\n"
            "  - $|\\tau| < 0.12$: **Çok Zayıf / Durağan** (belirgin bir yönelim yok)."
        )

    with st.expander("3️⃣ %95 Güven Aralığı (Confidence Interval — Alt ve Üst Eğim Sınırları)", expanded=True):
        st.markdown(
            "- **Tanım:** Sen's Slope yıllık değişim hızının **%95 olasılıkla** içinde bulunduğu `[Alt Sınır, Üst Sınır]` aralığıdır.\n"
            "- **Nasıl Yorumlanır?**\n"
            "  - **Her iki sınır da pozitifse** (ör. `[+0.021, +0.048]`): Yükseliş eğilimi kesindir; en iyimser ve en kötümser ısınma/artış hızı bu aralıktadır.\n"
            "  - **Her iki sınır da negatifse** (ör. `[-0.018, -0.005]`): Azalış eğilimi kesindir.\n"
            "  - **Aralık sıfırı (`0`) kapsıyorsa** (ör. `[-0.008, +0.014]`): Eğim yönü belirsizdir; yükseliş veya azalış istatistiksel olarak kesinleşmemiştir."
        )

    with st.expander("4️⃣ Hamed-Rao Z-Skoru, p-Değeri (Anlamlılık) ve Otokorelasyon Çarpanı", expanded=True):
        st.markdown(
            "- **p-Değeri (`p-value`):** Gözlenen yükselme veya azalmanın tamamen şans eseri (doğal iklim gürültüsü) oluşma olasılığıdır.\n"
            "  - `p < 0.001`: **Çok yüksek güvenle (%99.9+)** gerçek bir iklim trendi vardır.\n"
            "  - `p < 0.05`: **İstatistiksel olarak anlamlı (%95+ güvenle)** gerçek bir trend vardır (`🟢 İSTATİSTİKSEL OLARAK ANLAMLI TREND`).\n"
            "  - `p ≥ 0.05`: Gözlenen değişim doğal yıldan yıla dalgalanmadan ayırt edilemez (`⚪ İSTATİSTİKSEL OLARAK ANLAMSIZ / GÜRÜLTÜ`).\n"
            "- **Hamed-Rao (1998) Z-Skoru ve Otokorelasyon Çarpanı (`xV*/V`):** Aylık iklim serilerinde bu ayın sıcaklığı veya toprak nemi geçen aydan bağımsız değildir (pozitif otokorelasyon). Standart Mann-Kendall testi bu hafızayı yok saydığı için yanlışlıkla 'anlamlı trend var' hatasına düşebilir. **Hamed-Rao düzeltmesi**, anlamlı gecikmeleri (lag) tespit ederek varyansı `Otokorelasyon Çarpanı` kadar büyütür ve `Z` ile `p` değerini bilimsel olarak düzeltir."
        )

    with st.expander("5️⃣ STL Mevsimsel Ayrıştırma, Spearman ρ, ΔSlope ve 5 Yıllık SARIMA Bandı", expanded=False):
        st.markdown(
            "- **STL (Seasonal-Trend Decomposition using LOESS):** Ham seriyi $Y_t = T_t + S_t + R_t$ (Uzun Vadeli Trend + 12 Aylık Mevsimsel Döngü + Anomali/Artık) bileşenlerine ayırır. Tüm Mann-Kendall ve korelasyon testleri, yaz–kış döngüsünün yarattığı sahte korelasyonu önlemek için mevsimsellikten arındırılmış (`deseasonalized`) seri üzerinde yürütülür.\n"
            "- **Spearman Sıra Korelasyonu (`ρ`) ve `ΔSlope`:** Çiftli ve çoklu bölge analizlerinde iki bölgenin aynı anda mı ısındığını/kuraklaştığını (`ρ > 0`) yoksa bir bölge kuraklaşırken diğerinin yağış mı aldığını (`ρ < 0`, Teleconnection / Zıtlık) ve yıllık trend hızları arasındaki farkı (`ΔSlope = Eğim A - Eğim B`) ölçer.\n"
            "- **5 Yıllık SARIMA Projeksiyonu ve %95 Güven Bandı:** Seçilen dönemin sonundan itibaren 60 ay (5 yıl) ileriye yönelik mevsimsel otoregresif projeksiyon üretir; gölgeli bant %95 olasılıklı belirsizlik koridorunu temsil eder."
        )


# ---------------------------------------------------------------------------
# ORTAK YAN PANEL & KONTROL MERKEZİ (ANALİZ ROTALARI İÇİN)
# ---------------------------------------------------------------------------
def render_analysis_controls(primary_geo: Optional[Dict[str, Any]] = None) -> None:
    st.sidebar.markdown("## ⚙️ Analiz & Veri Mimarisi Kontrolleri")

    st.sidebar.markdown("### 📅 1. Zaman Aralığı & Hibrit Seri")
    cv = st.session_state.ctrl_version

    def _sync_ctrl(src_key: str, dst: str) -> None:
        val = st.session_state[src_key]
        st.session_state[dst] = (int(val[0]), int(val[1])) if dst == "year_range" else bool(val)

    yr_key = f"yr_slider_{cv}"
    st.sidebar.slider(
        "Tarihsel Analiz Aralığı (1980–2025):",
        min_value=1980,
        max_value=2025,
        value=tuple(st.session_state.year_range),
        step=1,
        key=yr_key,
        on_change=_sync_ctrl,
        args=(yr_key, "year_range"),
        help=f"STL + Mann-Kendall + SARIMA için en az {MIN_YEAR_SPAN} yıllık aralık gereklidir.",
    )
    yr0, yr1 = st.session_state.year_range
    st.sidebar.caption(
        f"Seçili dönem: **{yr0}–{yr1}** ({yr1 - yr0 + 1} yıl, {(yr1 - yr0 + 1) * 12} ay) → "
        f"Projeksiyon: **{yr1 + 1}–{yr1 + 5}**"
    )

    hy_key = f"hybrid_toggle_{cv}"
    st.sidebar.toggle(
        "🔄 MERRA-2 / GLDAS Hibrit Desteği (1980–2000 Boşluğunu Tamamla)",
        value=st.session_state.hybrid_enabled,
        key=hy_key,
        on_change=_sync_ctrl,
        args=(hy_key, "hybrid_enabled"),
        help=(
            "Aktif edildiğinde MODIS (2000+) ve GPM (2000+) öncesi 1980–2000 yıllarını "
            "NASA MERRA-2 ve GLDAS yeniden analiz serileriyle tamamlar. Kapatıldığında "
            "1980–2025 aralığında uydu doluluk oranı %70'in altına düşerse sistem eksik veri uyarısı verir."
        ),
    )

    se_key = f"season_toggle_{cv}"
    st.sidebar.toggle(
        "📉 Mevsimsel Döngüyü Gizle (STL Anomali Modu)",
        value=st.session_state.hide_seasonality,
        key=se_key,
        on_change=_sync_ctrl,
        args=(se_key, "hide_seasonality"),
        help=(
            "STL (Seasonal-Trend decomposition using LOESS) ile yaz-kış mevsimsel dalgalanmasını "
            "seriden çıkararak uzun vadeli gerçek iklim kaymasını ve anomalileri netleştirir."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 2. NASA Earthdata Login (GES DISC)")
    st.sidebar.text_input(
        "Earthdata Bearer Token (Opsiyonel / GES DISC):",
        value=st.session_state.get("nasa_earthdata_token", ""),
        type="password",
        key="nasa_earthdata_token",
        help=(
            "PDF Sayfa 5 ('Veriyi indirme ve analiz etme akışı — Adım 2'): "
            "NASA GES DISC (M2TMNXSLV, M2TMNXFLX, M2TMNXOCN, M2TMNXLND, M2TMNXRAD v5.12.4) ve "
            "Giovanni Cloud API servis erişimi için Earthdata Login Bearer Token giriniz."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 3. Hızlı Örnek Kombinasyon Yükle")
    if st.sidebar.button("1. 🏜️ Kuraklık: Aral Gölü (/single)", use_container_width=True):
        load_case_study("aral_tuz", use_tuz_golu=False)
        st.rerun()
    if st.sidebar.button("1b. 🇹🇷 Kuraklık: Tuz Gölü (/single)", use_container_width=True):
        load_case_study("aral_tuz", use_tuz_golu=True)
        st.rerun()
    if st.sidebar.button("2. 💧 Su Döngüsü Kombinasyonu (/dual)", use_container_width=True):
        load_case_study("water_cycle")
        st.rerun()
    if st.sidebar.button("3. ❄️ Kutup / Enerji Kombinasyonu (/single)", use_container_width=True):
        load_case_study("arctic_albedo")
        st.rerun()

    with st.sidebar.expander("🛰️ Resmi API Bağlantı Durumu"):
        st.caption(
            "- **NASA CMR (`cmr.earthdata.nasa.gov`):** 🟢 `M2TMNX... v5.12.4` Koleksiyon Doğrulaması Aktif\n"
            "- **NASA GES DISC / Giovanni (`api.giovanni.earthdata.nasa.gov`):** 🟢 Earthdata Login Token ile Doğrudan Seri\n"
            "- **NOAA CDO / NCEI (`ncei.noaa.gov`):** 🟢 Canlı HTTP API Aktif (Tokensız GHCN/GSOM Aylık Seri)\n"
            "- **Uydurma / Sentetik Veri:** 🚫 Kesinlikle Yok (%70 altı dolulukta açık uyarı verilir)"
        )

    st.markdown("### 🧬 Biyom Bazlı Değişken Önerici ve 10 Parametrelik Sistem Havuzu (Maksimum 4 Seçim)")

    if primary_geo is not None:
        rec_vars = primary_geo["recommended_vars"]
        rec_labels = [OFFICIAL_VARIABLES[v]["short_label"] for v in rec_vars if v in OFFICIAL_VARIABLES]

        b_col1, b_col2 = st.columns([3.0, 1.3])
        with b_col1:
            st.info(
                f"🧭 **Tespit Edilen Biyom / Arazi Sınıfı:** `{primary_geo['biome_name']}` ({primary_geo['display_label']})\n\n"
                f"💡 **Bu Biyom İçin Önerilen Birincil Değişkenler:** " + " • ".join([f"**{lbl}**" for lbl in rec_labels]) +
                f"\n\n*{primary_geo['biome_reason']}*"
            )
        with b_col2:
            st.write("")
            if st.button(
                "✨ Biyoma Uygun Önerilen 4 Parametreyi Otomatik Seç",
                type="primary",
                use_container_width=True,
            ):
                apply_recommended_biome_vars(rec_vars)
                st.rerun()
    else:
        rec_vars = []
        st.info(
            "🧭 **Biyom Bazlı Değişken Önerici:** Haritadan bir bölge seçtiğinizde o noktanın biyom sınıfı "
            "(Okyanus, Kıyı/Haliç, Çöl, Orman/Tarım, Kutup, Şehir) otomatik tespit edilir ve en uygun 4 değişken burada önerilir."
        )

    preset_c1, preset_c2, preset_c3 = st.columns(3)
    if preset_c1.button("🌵 Kuraklık Seti (T2M + PRECTOTCORR + GWETROOT + EVLAND)", use_container_width=True):
        apply_recommended_biome_vars(["T2M", "PRECTOTCORR", "GWETROOT", "EVLAND"])
        st.rerun()
    if preset_c2.button("❄️ Kutup / Enerji Seti (TSKINWTR + FRSEAICE + ALBEDO + SWGDN)", use_container_width=True):
        apply_recommended_biome_vars(["TSKINWTR", "FRSEAICE", "ALBEDO", "SWGDN"])
        st.rerun()
    if preset_c3.button("💧 Su Döngüsü Seti (PRECTOTCORR + GWETROOT + RUNOFF + TQV)", use_container_width=True):
        apply_recommended_biome_vars(["PRECTOTCORR", "GWETROOT", "RUNOFF", "TQV"])
        st.rerun()

    selected_count = len(st.session_state.selected_vars)
    st.caption(
        f"**Seçili Parametre Sayısı: `{selected_count} / 4`** — "
        f"{'🔒 Maksimum 4 parametre seçildi; kalan 6 parametre otomatik olarak devre dışı (disabled) bırakıldı.' if selected_count >= 4 else 'Listeden en fazla 4 parametre seçebilirsiniz.'} "
        f"*(Ağırlıklar kullanıcı müdahalesine kapalıdır; Önem/10 + CRITIC + VIF algoritmasıyla otomatik hesaplanır).* "
    )

    chk_cols = st.columns(2)
    sorted_vars = sorted(OFFICIAL_VARIABLES.items(), key=lambda x: x[1]["rank"])
    for idx, (var_code, meta) in enumerate(sorted_vars):
        is_selected = var_code in st.session_state.selected_vars
        is_disabled = (not is_selected) and (selected_count >= 4)
        is_rec_badge = " ⭐ [Biyom Önerisi]" if var_code in rec_vars else ""
        chk_key = f"chk_{var_code}_{st.session_state.chk_version}"
        score_10 = meta.get("importance_score_10", meta["base_weight"] * 10)
        coll = meta.get("collection", "")

        with chk_cols[idx % 2]:
            st.checkbox(
                f"{meta['label']} — Önem: {score_10:.1f}/10 [{coll}]{is_rec_badge}",
                value=is_selected,
                disabled=is_disabled,
                key=chk_key,
                on_change=_on_param_checkbox_change,
                args=(var_code, chk_key),
                help=f"{meta['student_explanation']} | Koleksiyon: {coll} ({meta['platform']})",
            )


def fetch_and_analyze_region(
    lat: float,
    lon: float,
    selected_vars: List[str],
    start_year: int,
    end_year: int,
    hybrid_enabled: bool,
    hide_seasonality: bool,
) -> Dict[str, Any]:
    """
    Belirli bir koordinat için seçilen (1–4) parametreyi çeker, %70 veri doluluk kuralını denetler,
    STL ayrıştırmasını, Hamed-Rao Mann-Kendall testini, CRITIC+VIF nesnel ağırlıklandırmasını
    ve 60 aylık (2026–2030) SARIMA tahminini yürütür.
    """
    geo_info = reverse_geocode_and_detect_biome(lat, lon)
    param_raw_map: Dict[str, Dict[str, Any]] = {}
    param_stl_map: Dict[str, Dict[str, Any]] = {}
    param_mk_map: Dict[str, Dict[str, Any]] = {}
    param_fc_map: Dict[str, pd.DataFrame] = {}
    missing_warnings: List[str] = []

    for v in selected_vars:
        res_raw = fetch_hybrid_climate_series(
            lat=lat,
            lon=lon,
            variable=v,
            start_year=start_year,
            end_year=end_year,
            use_hybrid_merra_gldas=hybrid_enabled,
        )
        param_raw_map[v] = res_raw

        if not res_raw["valid"]:
            missing_warnings.append(f"**{OFFICIAL_VARIABLES[v]['short_label']}:** {res_raw['reason']}")
            continue

        stl_res = stl_decompose_series(res_raw["df"], period=12)
        param_stl_map[v] = stl_res
        if not stl_res["valid"]:
            missing_warnings.append(
                f"**{OFFICIAL_VARIABLES[v]['short_label']}:** Seçilen tarih aralığı ve konum için yeterli uydu/istasyon verisi mevcut değil."
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
        )
        param_fc_map[v] = fc_single

    comp_bundle = compute_objective_weights_and_composite(
        param_stl_map=param_stl_map,
        selected_vars=selected_vars,
        official_variables_meta=OFFICIAL_VARIABLES,
        hide_seasonality=hide_seasonality,
    )

    if not comp_bundle["valid"]:
        return {
            "valid": False,
            "geo_info": geo_info,
            "missing_warnings": missing_warnings,
        }

    comp_df = comp_bundle["composite_df"]
    comp_mk = hamed_rao_mann_kendall_test(
        decimal_years(comp_df),
        comp_df["value"].to_numpy(dtype=float),
        is_monthly=True,
    )
    comp_fc = generate_5year_sarima_forecast(
        comp_df,
        value_col="value",
        hide_seasonality=hide_seasonality,
        steps=60,
    )

    return {
        "valid": True,
        "lat": float(lat),
        "lon": float(lon),
        "geo_info": geo_info,
        "missing_warnings": missing_warnings,
        "param_raw_map": param_raw_map,
        "param_stl_map": param_stl_map,
        "param_mk_map": param_mk_map,
        "param_fc_map": param_fc_map,
        "weights_df": comp_bundle["weights_df"],
        "vif_warnings": comp_bundle["vif_warnings"],
        "composite_df": comp_df,
        "composite_mk": comp_mk,
        "composite_fc": comp_fc,
    }


def render_weights_and_vif_section(weights_df: pd.DataFrame, vif_warnings: List[str]) -> None:
    """Kullanıcı müdahalesine kapalı IPCC AR6 + CRITIC + VIF nesnel ağırlık tablosunu gösterir."""
    with st.expander("⚖️ Nesnel Ağırlıklandırma Tablosu (IPCC AR6 + CRITIC + VIF > 5 Denetimi — Toplam = 1.00)", expanded=True):
        if vif_warnings:
            for vw in vif_warnings:
                st.warning(f"🛡️ **Çoklu Doğrusallık (VIF > 5) Koruması Devrede:** {vw}")

        disp_df = weights_df[
            [
                "rank",
                "label",
                "ipcc_base_weight",
                "critic_score",
                "vif_value",
                "vif_status",
                "normalized_weight",
                "weight_pct",
            ]
        ].rename(
            columns={
                "rank": "Önem Sırası",
                "label": "Parametre",
                "ipcc_base_weight": "IPCC AR6 Baz Ağırlık",
                "critic_score": "CRITIC Bilgi Skoru",
                "vif_value": "VIF Değeri",
                "vif_status": "Çoklu Doğrusallık Durumu",
                "normalized_weight": "Normalize Ağırlık (∑=1.00)",
                "weight_pct": "Etki Oranı (%)",
            }
        )
        st.dataframe(disp_df, use_container_width=True, hide_index=True)


def _build_short_mk_comment(mk: Dict[str, Any], unit_label: str) -> str:
    """Sen's Slope, Kendall's Tau, %95 Güven Aralığı ve p-değerine göre yükselme/azalma seviyesini kısa yorumlar."""
    slope_yr = float(mk.get("sen_slope_annual", 0.0))
    slope_dec = float(mk.get("sen_slope_decade", 0.0))
    tau = float(mk.get("kendall_tau", 0.0))
    abs_tau = abs(tau)
    p_val = float(mk.get("p_value", 1.0))
    ci_low = float(mk.get("sen_ci_low_annual", 0.0))
    ci_high = float(mk.get("sen_ci_high_annual", 0.0))
    is_sig = bool(mk.get("is_significant", False))

    if abs_tau >= 0.45:
        strength_txt = "çok güçlü ve istikrarlı"
    elif abs_tau >= 0.25:
        strength_txt = "belirgin (orta–güçlü)"
    elif abs_tau >= 0.12:
        strength_txt = "hafif–orta düzeyde"
    else:
        strength_txt = "çok zayıf / durağan"

    if abs(slope_yr) < 1e-6 or abs_tau < 0.04:
        dir_txt = "durağan (yönsüz) seyir"
        icon = "➖"
    elif slope_yr > 0:
        dir_txt = f"{strength_txt} yükseliş eğilimi"
        icon = "📈"
    else:
        dir_txt = f"{strength_txt} azalış eğilimi"
        icon = "📉"

    if ci_low > 0 and ci_high > 0:
        ci_txt = "%95 güven aralığı tamamen pozitif bölgede (artış yönü kesin)"
    elif ci_low < 0 and ci_high < 0:
        ci_txt = "%95 güven aralığı tamamen negatif bölgede (azalış yönü kesin)"
    else:
        ci_txt = "%95 güven aralığı sıfırı kesiyor (eğim yönü dalgalı)"

    if is_sig and p_val < 0.001:
        sig_txt = f"çok yüksek istatistiksel güvenle (p={p_val:.4f} < 0.001) kanıtlanmıştır."
    elif is_sig:
        sig_txt = f"istatistiksel olarak anlamlıdır (p={p_val:.4f} < 0.05)."
    else:
        sig_txt = f"istatistiksel anlamlılık eşiğinin altındadır (p={p_val:.4f} ≥ 0.05; doğal iklim dalgalanması baskın)."

    return (
        f"{icon} **Kısa İstatistiksel Yorum:** Seride 10 yılda ortalama `{slope_dec:+.3f} {unit_label}` hızında "
        f"**{dir_txt}** (`τ = {tau:+.3f}`) gözlenmektedir; {ci_txt} ve bu değişim {sig_txt}"
    )


def render_mk_significance_card(title: str, mk: Dict[str, Any], unit_label: str = "Z-Skoru") -> None:
    """Hamed-Rao Modified Mann-Kendall ve Sen's Slope istatistiksel anlamlılık kartını ve kısa yorumunu basar."""
    badge_class = "sig-badge-ok" if mk["is_significant"] else "sig-badge-noise"
    st.markdown(
        f"""
        <div class="{badge_class}">
            <b>{title}:</b> {mk['badge_text']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sen's Slope (Yıllık Hız)", f"{mk['sen_slope_annual']:+.4f} {unit_label}/yıl", f"{mk['sen_slope_decade']:+.3f} /10 yıl")
    m2.metric("Kendall's Tau (τ) & S", f"τ = {mk['kendall_tau']:+.3f}", f"S = {mk['S']}")
    m3.metric("%95 Güven Aralığı (Eğim)", f"[{mk['sen_ci_low_annual']:+.4f}, {mk['sen_ci_high_annual']:+.4f}]", f"Otokorelasyon Çarpanı: x{mk['autocorr_factor']:.2f}")
    m4.metric("Hamed-Rao Z & p-Değeri", f"p = {mk['p_value']:.5f}", f"Z = {mk['z_score']:+.3f}")
    st.caption(_build_short_mk_comment(mk, unit_label))


def year_span_ok() -> bool:
    """Seçilen tarih aralığı analiz için yeterli mi? Değilse açık uyarı verir (sessizce yanlış grafik çizmez)."""
    yr0, yr1 = st.session_state.year_range
    span = yr1 - yr0 + 1
    if span < MIN_YEAR_SPAN:
        st.warning(
            f"⏳ Seçilen aralık **{yr0}–{yr1}** yalnızca **{span} yıl** ({span * 12} ay). "
            f"Mevsimsel ayrıştırma (STL), Mann-Kendall trend testi ve 5 yıllık SARIMA projeksiyonunun "
            f"güvenilir olması için en az **{MIN_YEAR_SPAN} yıllık** bir aralık seçin."
        )
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
) -> Dict[str, Any]:
    """
    Çiftli/Çoklu kıyas için: ortak ağırlıklar → her bölgenin ortak ağırlıklı bütünleşik serisi,
    Mann-Kendall ve SARIMA projeksiyonu.
    """
    names = list(region_results.keys())
    shared = compute_shared_weights(
        [region_results[n]["param_stl_map"] for n in names],
        selected_vars,
        OFFICIAL_VARIABLES,
        hide_seasonality,
    )
    if not shared["valid"]:
        return {"valid": False, "dropped_vars": shared["dropped_vars"]}

    composites, forecasts = {}, {}
    for n in names:
        comp = apply_weights_to_composite(region_results[n]["param_stl_map"], shared["weights_df"], hide_seasonality)
        composites[n] = comp
        forecasts[n] = generate_5year_sarima_forecast(comp, "value", hide_seasonality, steps=60)
    return {
        "valid": True,
        "weights_df": shared["weights_df"],
        "vif_warnings": shared["vif_warnings"],
        "common_vars": shared["common_vars"],
        "dropped_vars": shared["dropped_vars"],
        "composites": composites,
        "forecasts": forecasts,
    }


def render_per_parameter_comparisons(
    region_results: Dict[str, Dict[str, Any]],
    common_vars: List[str],
    weights_df: pd.DataFrame,
    hide_seasonality: bool,
) -> None:
    """Bütünleşik grafiğin ALTINDA: her parametre için AYRI bölgeler arası kıyas grafiği + 1980 vs Günümüz Dünya Isı Haritası + ilişki özeti."""
    st.markdown("### 📈 Parametre Bazlı Ayrık Kıyas Grafikleri ve Küresel Isı Haritaları (1980 vs. Günümüz)")
    st.caption(
        "Her parametre kendi fiziksel biriminde ayrı bir grafikte gösterilir ve hemen altında ~1980 dönemi ile günümüz "
        "dünya ısı haritası karşılaştırılır. İlişki istatistikleri mevsimsellikten arındırılmış seriler üzerinden hesaplanır."
    )
    w_map = _weight_lookup(weights_df)
    rel_rows = []
    region_coords_list = [
        {"name": n, "lat": r.get("lat", 0.0), "lon": r.get("lon", 0.0)}
        for n, r in region_results.items()
    ]

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
            weight_pct=w_map.get(v, 0.0),
            rel_stats=rel,
            hide_seasonality=hide_seasonality,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"param_cmp_{v}")

        # Her parametre grafiğinin hemen altında ~1980 vs Günümüz Dünya Isı Haritası
        st.plotly_chart(
            plot_world_1980_vs_present_heatmap(
                var_code=v,
                title_label=meta["short_label"],
                unit_label=meta["unit"],
                region_series_map={n: r["param_stl_map"][v]["df"] for n, r in region_results.items()},
                region_coords=region_coords_list,
                col_name="observed",
            ),
            use_container_width=True,
            key=f"world_heat_cmp_{v}_{len(region_results)}",
        )

        with st.expander(f"📐 {meta['short_label']} — Bölgelerin İstatistik Detayı (Sen's Slope, Kendall τ, %95 GA, p-Değeri ve Yorum)"):
            for n, mk_v in region_mk.items():
                render_mk_significance_card(f"{n} — {meta['short_label']}", mk_v, unit_label=meta["unit"])

        if rel.get("valid", False):
            if len(region_results) == 2:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spearman ρ", f"{rel['spearman_rho']:+.3f}", f"p = {rel['spearman_p']:.4f}")
                c2.metric("ΔSlope (A − B)", f"{rel['delta_slope']:+.4f}", f"{meta['unit']}/yıl")
                c3.metric("Maks. Çapraz Korelasyon", f"{rel['best_xcorr']:+.3f}", f"{rel['best_lag_months']} ay gecikme")
                c4.metric("Tepki Sınıfı", "Ters Yönlü" if rel["is_teleconnection"] else "Eş/Benzer Yönlü")
            else:
                with st.expander(f"🔎 {meta['short_label']} — Bölge çiftleri ilişki tablosu"):
                    st.dataframe(rel["pairwise_df"], use_container_width=True, hide_index=True)
            rel_rows.append(
                {
                    "Parametre": meta["short_label"],
                    "Ağırlık (%)": w_map.get(v, 0.0),
                    "Spearman ρ (1.–2.)": rel["spearman_rho"],
                    "ΔSlope (1.–2.)": rel["delta_slope"],
                    "Sonuç": rel["verdict_badge"],
                }
            )
        st.markdown("---")

    if rel_rows:
        st.markdown("#### 🧾 Parametre Bazlı Zıtlık / İlişki Özeti")
        st.dataframe(pd.DataFrame(rel_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# ROTA 3: `/analyze/single` — TEKLİ BÖLGE ANALİZİ (HARİTADA SADECE 1 İKON)
# ---------------------------------------------------------------------------
def render_single_analyze_route() -> None:
    st.subheader("📍 Tekli Bölge İklim Trendi, STL Ayrıştırma ve 5 Yıllık Projeksiyon (/analyze/single)")
    if st.session_state.active_case_title:
        st.caption(f"📌 **Aktif Vaka Çalışması:** {st.session_state.active_case_title}")

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
                    <b>🎯 Seçili Bölge (Haritada Tek İkon):</b> {geo_s['country']} — {geo_s['city_region']}
                    &nbsp;|&nbsp; <b>Biyom:</b> {geo_s['biome_name']} &nbsp;|&nbsp; <code>{lat_s:.2f}°, {lon_s:.2f}°</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_lat_in:
            new_lat = st.number_input("Enlem (Lat):", -85.0, 85.0, value=lat_s, step=0.25, format="%.2f", key=f"in_single_lat_{cver}")
        with c_lon_in:
            new_lon = st.number_input("Boylam (Lon):", -180.0, 180.0, value=lon_s, step=0.25, format="%.2f", key=f"in_single_lon_{cver}")
        with c_clr:
            st.write("")
            if st.button("❌ Seçimi Kaldır", key="clr_single_btn", use_container_width=True):
                st.session_state.marker_single = None
                st.session_state.active_case_title = None
                st.session_state.coord_version += 1
                st.rerun()

        if round(new_lat, 2) != round(lat_s, 2) or round(new_lon, 2) != round(lon_s, 2):
            st.session_state.marker_single = {"lat": float(new_lat), "lon": float(new_lon)}
            st.rerun()

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
            st.info("👆 **Henüz bölge seçilmedi.** Harita üzerinde istediğiniz ülkeye/noktaya tıklayın veya sağdan koordinat girip **📍 Bölgeyi Seç** butonuna basın.")
        with c_lat_in:
            in_lat = st.number_input("Enlem (Lat):", -85.0, 85.0, value=0.0, step=0.5, format="%.2f", key=f"empty_single_lat_{cver}")
        with c_lon_in:
            in_lon = st.number_input("Boylam (Lon):", -180.0, 180.0, value=0.0, step=0.5, format="%.2f", key=f"empty_single_lon_{cver}")
        with c_add:
            st.write("")
            if st.button("📍 Bölgeyi Seç", type="primary", use_container_width=True):
                st.session_state.marker_single = {"lat": float(in_lat), "lon": float(in_lon)}
                st.session_state.coord_version += 1
                st.rerun()
        m = _create_base_folium_map(25.0, 15.0, zoom=2)

    folium.LayerControl().add_to(m)

    map_out = st_folium(m, use_container_width=True, height=440, key="nasa_map_single", returned_objects=["last_clicked"])
    if map_out and map_out.get("last_clicked"):
        c_lat = round(float(map_out["last_clicked"]["lat"]), 2)
        c_lon = round(float(map_out["last_clicked"]["lng"]), 2)
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
        lat_s, lon_s, st.session_state.selected_vars, start_yr, end_yr,
        st.session_state.hybrid_enabled, hide,
    )

    for mw in res["missing_warnings"]:
        st.error(f"⚠️ {mw}")
    if not res["valid"]:
        return

    render_weights_and_vif_section(res["weights_df"], res["vif_warnings"])

    # 1. Bütünleşik grafik (seçilen parametrelerin ağırlıklı sentezi)
    st.markdown("### 🌍 Bütünleşik İklim İndeksi")
    render_mk_significance_card(f"{geo_s['display_label']} — Bütünleşik Trend", res["composite_mk"])
    st.plotly_chart(
        plot_single_region_composite_and_forecast(
            geo_s["display_label"], res["composite_df"], res["composite_mk"],
            res["composite_fc"], res["weights_df"], hide,
        ),
        use_container_width=True,
        key="single_composite",
    )
    single_coords = [{"name": geo_s["display_label"], "lat": lat_s, "lon": lon_s}]
    st.plotly_chart(
        plot_world_1980_vs_present_heatmap(
            var_code="COMPOSITE",
            title_label="Bütünleşik İklim Stres İndeksi",
            unit_label="Z-Skoru",
            region_series_map={geo_s["display_label"]: res["composite_df"]},
            region_coords=single_coords,
            col_name="value",
        ),
        use_container_width=True,
        key="single_composite_world_heat",
    )

    # 2. HER PARAMETRE İÇİN AYRI GRAFİK + ALTINDA 1980 VS GÜNÜMÜZ DÜNYA ISI HARİTASI
    st.markdown("### 📊 Parametre Bazlı Ayrı Grafikler ve 1980 vs. Günümüz Dünya Isı Haritaları")
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
                weight_pct=w_map.get(v, 0.0),
                hide_seasonality=hide,
                color=PARAM_PALETTE[idx % len(PARAM_PALETTE)],
            ),
            use_container_width=True,
            key=f"single_param_{v}",
        )
        st.plotly_chart(
            plot_world_1980_vs_present_heatmap(
                var_code=v,
                title_label=meta["short_label"],
                unit_label=meta["unit"],
                region_series_map={geo_s["display_label"]: res["param_stl_map"][v]["df"]},
                region_coords=single_coords,
                col_name="observed",
            ),
            use_container_width=True,
            key=f"single_param_world_heat_{v}",
        )
        with st.expander(f"📐 {meta['short_label']} — İstatistik Detayı (Sen's Slope, Kendall τ, %95 GA, p-Değeri ve Yorum)", expanded=True):
            render_mk_significance_card(meta["short_label"], res["param_mk_map"][v], unit_label=meta["unit"])

    with st.expander("🔗 Resmi NASA Giovanni Doğrulama Bağlantıları"):
        for v in st.session_state.selected_vars:
            g_url = build_giovanni_direct_url(lat_s, lon_s, v, start_yr, end_yr)
            st.markdown(f"- **{OFFICIAL_VARIABLES[v]['label']}:** [NASA Giovanni Time Series URL'sini Aç]({g_url})")


# ---------------------------------------------------------------------------
# ROTA 4: `/analyze/dual` — ÇİFTLİ BÖLGE KARŞILAŞTIRMASI (TELECONNECTION)
# ---------------------------------------------------------------------------
def render_dual_analyze_route() -> None:
    st.subheader("⚡ Çiftli Bölge Zıtlık / İlişki Analizi ve 5 Yıllık Projeksiyon (/analyze/dual)")
    if st.session_state.active_case_title:
        st.caption(f"📌 **Aktif Vaka Çalışması:** {st.session_state.active_case_title}")

    m_a, m_b = st.session_state.markers_dual
    geo_a = reverse_geocode_and_detect_biome(float(m_a["lat"]), float(m_a["lon"])) if m_a else None
    geo_b = reverse_geocode_and_detect_biome(float(m_b["lat"]), float(m_b["lon"])) if m_b else None
    cver = st.session_state.coord_version

    top_c1, top_c2, top_c3 = st.columns([1.25, 1.37, 1.37])
    with top_c1:
        default_idx = 0 if m_a is None else (1 if m_b is None else 0)
        target_pin = st.radio(
            "🖱️ Haritaya Tıklanınca Seçilecek / Güncellenecek Bölge:",
            options=["🔴 Bölge A (1. Pin)", "🔵 Bölge B (2. Pin)"],
            index=default_idx,
            horizontal=True,
        )
        if st.button("🗑️ Her İki Bölge Seçimini Temizle", disabled=(m_a is None and m_b is None), use_container_width=True):
            st.session_state.markers_dual = [None, None]
            st.session_state.active_case_title = None
            st.session_state.coord_version += 1
            st.rerun()

    with top_c2:
        if m_a is not None and geo_a is not None:
            st.markdown(f"**🔴 Bölge A:** `{geo_a['display_label']}`  \n*Biyom:* {geo_a['biome_name']}")
            if st.button("❌ Bölge A Seçimini Kaldır", key="rm_dual_a", use_container_width=True):
                st.session_state.markers_dual[0] = None
                st.session_state.coord_version += 1
                st.rerun()
        else:
            st.markdown("**🔴 Bölge A:** *Henüz seçilmedi (Haritadan tıklayın veya koordinat girin)*")
            ca1, ca2, ca3 = st.columns([1, 1, 0.9])
            na_lat = ca1.number_input("Enlem A:", -85.0, 85.0, value=0.0, step=0.5, key=f"dual_a_lat_{cver}")
            na_lon = ca2.number_input("Boylam A:", -180.0, 180.0, value=0.0, step=0.5, key=f"dual_a_lon_{cver}")
            ca3.write("")
            if ca3.button("➕ Seç A", key="set_dual_a", use_container_width=True):
                st.session_state.markers_dual[0] = {"lat": float(na_lat), "lon": float(na_lon)}
                st.session_state.coord_version += 1
                st.rerun()

    with top_c3:
        if m_b is not None and geo_b is not None:
            st.markdown(f"**🔵 Bölge B:** `{geo_b['display_label']}`  \n*Biyom:* {geo_b['biome_name']}")
            if st.button("❌ Bölge B Seçimini Kaldır", key="rm_dual_b", use_container_width=True):
                st.session_state.markers_dual[1] = None
                st.session_state.coord_version += 1
                st.rerun()
        else:
            st.markdown("**🔵 Bölge B:** *Henüz seçilmedi (Haritadan tıklayın veya koordinat girin)*")
            cb1, cb2, cb3 = st.columns([1, 1, 0.9])
            nb_lat = cb1.number_input("Enlem B:", -85.0, 85.0, value=0.0, step=0.5, key=f"dual_b_lat_{cver}")
            nb_lon = cb2.number_input("Boylam B:", -180.0, 180.0, value=0.0, step=0.5, key=f"dual_b_lon_{cver}")
            cb3.write("")
            if cb3.button("➕ Seç B", key="set_dual_b", use_container_width=True):
                st.session_state.markers_dual[1] = {"lat": float(nb_lat), "lon": float(nb_lon)}
                st.session_state.coord_version += 1
                st.rerun()

    m = _create_base_folium_map(25.0, 15.0, zoom=2)
    if m_a is not None and geo_a is not None:
        folium.Marker(
            location=[float(m_a["lat"]), float(m_a["lon"])],
            tooltip=folium.Tooltip("<b>🔴 BÖLGE A</b><br>" + geo_a["tooltip_html"], sticky=True),
            popup=folium.Popup("<b>🔴 BÖLGE A</b><br>" + geo_a["tooltip_html"], max_width=280),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
    if m_b is not None and geo_b is not None:
        folium.Marker(
            location=[float(m_b["lat"]), float(m_b["lon"])],
            tooltip=folium.Tooltip("<b>🔵 BÖLGE B</b><br>" + geo_b["tooltip_html"], sticky=True),
            popup=folium.Popup("<b>🔵 BÖLGE B</b><br>" + geo_b["tooltip_html"], max_width=280),
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)
    if m_a is not None and m_b is not None:
        folium.PolyLine(
            locations=[[float(m_a["lat"]), float(m_a["lon"])], [float(m_b["lat"]), float(m_b["lon"])]],
            color="#EF4444",
            weight=3.0,
            dash_array="6, 8",
            tooltip="Teleconnection Aksı",
        ).add_to(m)
    folium.LayerControl().add_to(m)

    map_out = st_folium(m, use_container_width=True, height=440, key="nasa_map_dual", returned_objects=["last_clicked"])
    if map_out and map_out.get("last_clicked"):
        c_lat = round(float(map_out["last_clicked"]["lat"]), 2)
        c_lon = round(float(map_out["last_clicked"]["lng"]), 2)
        click_pt = (c_lat, c_lon)
        if click_pt != st.session_state.get("dual_last_click"):
            st.session_state.dual_last_click = click_pt
            if m_a is None:
                idx = 0
            elif m_b is None:
                idx = 1
            else:
                idx = 0 if target_pin.startswith("🔴") else 1
            st.session_state.markers_dual[idx] = {"lat": c_lat, "lon": c_lon}
            st.session_state.coord_version += 1
            st.session_state.active_case_title = None
            st.rerun()

    render_analysis_controls(geo_a or geo_b)

    if m_a is None or m_b is None:
        st.info("👆 **Çiftli bölge karşılaştırmasını başlatmak için harita üzerinden hem 🔴 Bölge A hem de 🔵 Bölge B noktasını seçin.**")
        return
    if not year_span_ok():
        return

    lat_a, lon_a = float(m_a["lat"]), float(m_a["lon"])
    lat_b, lon_b = float(m_b["lat"]), float(m_b["lon"])
    start_yr, end_yr = st.session_state.year_range
    hide = st.session_state.hide_seasonality
    name_a = _region_label("🔴 A:", geo_a, lat_a, lon_a)
    name_b = _region_label("🔵 B:", geo_b, lat_b, lon_b)

    region_results = {}
    for name, lt, ln in [(name_a, lat_a, lon_a), (name_b, lat_b, lon_b)]:
        r = fetch_and_analyze_region(lt, ln, st.session_state.selected_vars, start_yr, end_yr,
                                     st.session_state.hybrid_enabled, hide)
        for w_msg in r["missing_warnings"]:
            st.error(f"⚠️ **{name}:** {w_msg}")
        if r["valid"]:
            region_results[name] = r

    if len(region_results) < 2:
        st.warning("Karşılaştırma için her iki bölgede de en az bir parametrenin geçerli verisi olmalıdır.")
        return

    shared = build_shared_comparison(region_results, st.session_state.selected_vars, hide)
    if not shared["valid"]:
        st.warning("Seçilen parametrelerin hiçbiri her iki bölgede birden geçerli veri içermiyor.")
        return
    if shared["dropped_vars"]:
        st.info(
            "ℹ️ Şu parametreler bölgelerden birinde veri içermediği için bütünleşik kıyastan çıkarıldı: "
            + ", ".join(OFFICIAL_VARIABLES[v]["short_label"] for v in shared["dropped_vars"])
        )

    tele_res = compare_regions_teleconnection(
        {name_a: shared["composites"][name_a], name_b: shared["composites"][name_b]}, value_col="value"
    )

    # 1. EN ÜSTTE: BÜTÜNLEŞİK ZITLIK / İLİŞKİ
    st.markdown("### 🌐 Bütünleşik Zıtlık / İlişki Grafiği")
    st.markdown(
        f"""
        <div class="tele-banner">
            <h4 style="margin-top:0;color:#FBBF24;">{tele_res['verdict_badge']}</h4>
            <p style="margin-bottom:0;font-size:14.5px;color:#E2E8F0;">{tele_res['explanation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        plot_dual_region_integrated_composite(
            name_a, name_b, shared["composites"][name_a], shared["composites"][name_b],
            shared["forecasts"][name_a], shared["forecasts"][name_b], tele_res, shared["weights_df"], hide,
        ),
        use_container_width=True,
        key="dual_integrated",
    )
    st.plotly_chart(
        plot_world_1980_vs_present_heatmap(
            var_code="COMPOSITE",
            title_label="Bütünleşik İklim Stres İndeksi",
            unit_label="Z-Skoru",
            region_series_map={name_a: shared["composites"][name_a], name_b: shared["composites"][name_b]},
            region_coords=[
                {"name": name_a, "lat": lat_a, "lon": lon_a},
                {"name": name_b, "lat": lat_b, "lon": lon_b},
            ],
            col_name="value",
        ),
        use_container_width=True,
        key="dual_integrated_world_heat",
    )
    render_weights_and_vif_section(shared["weights_df"], shared["vif_warnings"])
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
    st.subheader("🌐 Çoklu Bölge Zıtlık / İlişki Analizi ve 5 Yıllık Projeksiyon (/analyze/multi — Maks. 10 Bölge)")

    markers = st.session_state.markers_multi
    n_markers = len(markers)
    geo_list = [reverse_geocode_and_detect_biome(mk["lat"], mk["lon"]) for mk in markers]
    cver = st.session_state.coord_version

    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns([1.8, 0.65, 0.65, 0.7, 0.8])
    with b_col1:
        st.markdown(f"#### 📌 Seçili Bölgeler (`{n_markers} / 10`)")
        st.caption("Haritaya tıklayarak veya sağdan koordinat girerek istediğiniz bölgeleri ekleyin.")
    with b_col2:
        add_lat = st.number_input("Enlem:", -85.0, 85.0, value=0.0, step=0.5, key=f"multi_in_lat_{cver}")
    with b_col3:
        add_lon = st.number_input("Boylam:", -180.0, 180.0, value=0.0, step=0.5, key=f"multi_in_lon_{cver}")
    with b_col4:
        st.write("")
        if st.button("➕ Bölge Ekle", disabled=(n_markers >= 10), use_container_width=True):
            st.session_state.markers_multi.append({"lat": round(float(add_lat), 2), "lon": round(float(add_lon), 2)})
            st.session_state.coord_version += 1
            st.rerun()
    with b_col5:
        st.write("")
        if st.button("🗑️ Tüm Seçimleri Temizle", disabled=(n_markers == 0), use_container_width=True):
            st.session_state.markers_multi = []
            st.session_state.multi_last_click = None
            st.session_state.coord_version += 1
            st.rerun()

    m = _create_base_folium_map(25.0, 15.0, zoom=2)
    folium_colors = ["red", "blue", "green", "orange", "purple", "pink", "darkblue", "lightgreen", "beige", "cadetblue"]
    for idx, (mk, g) in enumerate(zip(markers, geo_list)):
        folium.Marker(
            location=[mk["lat"], mk["lon"]],
            tooltip=folium.Tooltip(f"<b>Bölge #{idx + 1}</b><br>" + g["tooltip_html"], sticky=True),
            popup=folium.Popup(f"<b>Bölge #{idx + 1}</b><br>" + g["tooltip_html"], max_width=280),
            icon=folium.Icon(color=folium_colors[idx % len(folium_colors)], icon="info-sign"),
        ).add_to(m)
    folium.LayerControl().add_to(m)

    map_out = st_folium(m, use_container_width=True, height=440, key="nasa_map_multi", returned_objects=["last_clicked"])
    if map_out and map_out.get("last_clicked"):
        c_lat = round(float(map_out["last_clicked"]["lat"]), 2)
        c_lon = round(float(map_out["last_clicked"]["lng"]), 2)
        click = (c_lat, c_lon)
        if click != st.session_state.get("multi_last_click"):
            st.session_state.multi_last_click = click
            exists = any(abs(c_lat - x["lat"]) < 0.05 and abs(c_lon - x["lon"]) < 0.05 for x in markers)
            if not exists:
                if len(markers) < 10:
                    st.session_state.markers_multi.append({"lat": c_lat, "lon": c_lon})
                    st.rerun()
                else:
                    st.warning("🔒 Maksimum 10 bölge sınırına ulaşıldı. Yeni eklemek için listeden bir bölgeyi silin.")

    # Seçili bölgeleri listele ve HER BİRİNİN yanına bağımsız '❌ Kaldır' butonu koy (hiçbir bölge kilitli kalmaz)
    if markers:
        st.markdown("##### 📍 Eklenen Bölgeler (İstediğiniz bölgeyi tek tek kaldırabilirsiniz):")
        for idx, (mk, g) in enumerate(zip(markers, geo_list)):
            r_col1, r_col2 = st.columns([4.2, 0.8])
            with r_col1:
                st.markdown(
                    f"**#{idx + 1}** — {g['country']} — {g['city_region']} (`{mk['lat']:.2f}°, {mk['lon']:.2f}°`) | *{g['biome_name']}*"
                )
            with r_col2:
                if st.button(f"❌ Kaldır #{idx + 1}", key=f"rm_multi_item_{idx}", use_container_width=True):
                    st.session_state.markers_multi.pop(idx)
                    st.session_state.multi_last_click = None
                    st.rerun()

    render_analysis_controls(geo_list[0] if geo_list else None)

    if len(markers) < 2:
        st.info(
            f"👆 **Çoklu bölge karşılaştırmasını başlatmak için harita üzerine tıklayarak en az 2 bölge seçin "
            f"(Şu an seçili: `{len(markers)} / 10`).**"
        )
        return
    if not year_span_ok():
        return

    start_yr, end_yr = st.session_state.year_range
    hide = st.session_state.hide_seasonality

    region_results: Dict[str, Dict[str, Any]] = {}
    region_geo: Dict[str, Dict[str, Any]] = {}
    for idx, (mk, g) in enumerate(zip(markers, geo_list)):
        r_label = _region_label(f"#{idx + 1}", g, mk["lat"], mk["lon"])
        r = fetch_and_analyze_region(mk["lat"], mk["lon"], st.session_state.selected_vars, start_yr, end_yr,
                                     st.session_state.hybrid_enabled, hide)
        for mw in r["missing_warnings"]:
            st.error(f"⚠️ **{r_label}:** {mw}")
        if r["valid"]:
            region_results[r_label] = r
            region_geo[r_label] = g

    if len(region_results) < 2:
        st.warning("Karşılaştırma için en az 2 bölgede geçerli veri bulunmalıdır.")
        return

    shared = build_shared_comparison(region_results, st.session_state.selected_vars, hide)
    if not shared["valid"]:
        st.warning("Seçilen parametrelerin hiçbiri tüm bölgelerde ortak geçerli veri içermiyor.")
        return
    if shared["dropped_vars"]:
        st.info(
            "ℹ️ Şu parametreler bazı bölgelerde veri içermediği için kıyastan çıkarıldı: "
            + ", ".join(OFFICIAL_VARIABLES[v]["short_label"] for v in shared["dropped_vars"])
        )

    tele_multi = compare_regions_teleconnection(shared["composites"], value_col="value")

    # 1. EN ÜSTTE: BÜTÜNLEŞİK ZITLIK / İLİŞKİ
    st.markdown("### 🌐 Bütünleşik Zıtlık / İlişki Grafiği")
    fig_ts, fig_heat = plot_multi_region_comparison(shared["composites"], shared["forecasts"], tele_multi)
    st.plotly_chart(fig_ts, use_container_width=True, key="multi_integrated")
    st.plotly_chart(
        plot_world_1980_vs_present_heatmap(
            var_code="COMPOSITE",
            title_label="Bütünleşik İklim Stres İndeksi",
            unit_label="Z-Skoru",
            region_series_map=shared["composites"],
            region_coords=[
                {"name": n, "lat": r.get("lat", 0.0), "lon": r.get("lon", 0.0)}
                for n, r in region_results.items()
            ],
            col_name="value",
        ),
        use_container_width=True,
        key="multi_integrated_world_heat",
    )

    with st.expander("📐 Bütünleşik İndeks — Bölgelerin İstatistik Detayı (Sen's Slope, Kendall τ, %95 GA, p-Değeri ve Yorum)", expanded=True):
        for name in shared["composites"]:
            render_mk_significance_card(name, tele_multi["region_stats"][name])

    c_heat, c_tbl = st.columns([1.15, 1.35])
    with c_heat:
        st.plotly_chart(fig_heat, use_container_width=True, key="multi_heat")
    with c_tbl:
        summary_rows = []
        for name in shared["composites"]:
            mk_stat = tele_multi["region_stats"][name]
            summary_rows.append(
                {
                    "Bölge": name,
                    "Biyom": region_geo[name]["biome_name"],
                    "Sen's Slope (Yıllık)": mk_stat["sen_slope_annual"],
                    "Kendall τ": mk_stat["kendall_tau"],
                    "p-Değeri": mk_stat["p_value"],
                    "Anlamlılık": mk_stat["badge_text"],
                }
            )
        st.markdown("#### 📋 Bütünleşik Trend Özeti")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.markdown("#### ⚡ Bölge Çiftleri (Bütünleşik İndeks)")
        st.dataframe(tele_multi["pairwise_df"], use_container_width=True, hide_index=True)

    render_weights_and_vif_section(shared["weights_df"], shared["vif_warnings"])

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
