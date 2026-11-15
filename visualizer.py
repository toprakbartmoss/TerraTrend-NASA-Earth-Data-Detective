"""
NASA Earth Data Detective (EDD) - Bilimsel Görselleştirme Modülü (visualizer.py)

Grafik Düzeni:
- Tekli Bölge  : 1 bütünleşik grafik + HER PARAMETRE İÇİN AYRI grafik (kendi biriminde)
- Çiftli Bölge : 1 bütünleşik zıtlık/ilişki grafiği (2 çizgi) + HER PARAMETRE İÇİN AYRI kıyas grafiği
- Çoklu Bölge  : 1 bütünleşik grafik (N çizgi) + korelasyon ısı haritası + HER PARAMETRE İÇİN AYRI kıyas grafiği
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from i18n import L

REGION_PALETTE = [
    "#FF4B4B",  # 1. Bölge / Bölge A (Kırmızı)
    "#00B4D8",  # 2. Bölge / Bölge B (Mavi)
    "#10B981",  # 3. Zümrüt
    "#F59E0B",  # 4. Kehribar
    "#A855F7",  # 5. Mor
    "#EC4899",  # 6. Pembe
    "#3B82F6",  # 7. Kraliyet Mavisi
    "#84CC16",  # 8. Limon
    "#F97316",  # 9. Turuncu
    "#14B8A6",  # 10. Turkuaz
]

PARAM_PALETTE = ["#38BDF8", "#F97316", "#10B981", "#A855F7"]

_LAYOUT_BASE = dict(
    template="plotly_dark",
    hovermode="x unified",
    # Legend sits at the bottom of the figure (container coordinates) so it can never collide with the
    # title block; plotly grows the bottom margin to fit however many rows it needs.
    legend=dict(
        orientation="h", yref="container", y=0.0, yanchor="bottom", xanchor="left", x=0.0,
        font=dict(size=11), bgcolor="rgba(0,0,0,0)", tracegroupgap=4,
    ),
    paper_bgcolor="#0E1117",
    plot_bgcolor="#161B22",
)


def _title(main: str, *sub_lines: str) -> dict:
    """Title block: bold main line + grey subtitle lines; automargin pushes the plot down to fit it."""
    subs = "".join(f"<br><span style='font-size:12px;color:#94A3B8;'>{t}</span>" for t in sub_lines if t)
    return dict(
        # Anchored to the top of the plot area and growing upwards; automargin enlarges the top margin to fit
        # every line, so the title block never overlaps the plot or gets cut off.
        text=f"{main}{subs}", x=0.0, xref="paper", xanchor="left", y=1.0, yref="paper", yanchor="bottom",
        pad=dict(b=12), automargin=True, font=dict(size=16),
    )


def _wrap_join(items: List[str], sep: str = " + ", per_line: int = 3) -> List[str]:
    """Join items into several lines (per_line items each) so long weight lists do not run off the chart."""
    return [sep.join(items[i:i + per_line]) for i in range(0, len(items), per_line)] or [""]


def _hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def forecast_period_label(fc_df: Optional[pd.DataFrame]) -> str:
    """Tahmin döneminin gerçek yıllarını döndürür (seçilen bitiş yılına göre dinamik)."""
    if fc_df is None or fc_df.empty:
        return "—"
    return f"{pd.Timestamp(fc_df['date'].iloc[0]).year}–{pd.Timestamp(fc_df['date'].iloc[-1]).year}"


def _p_label(disp: Any) -> str:
    """Motorun p_value_display metni için: 'p = 0.003' veya 'p < 0.001'."""
    d = str(disp or "—").strip()
    return f"p {d}" if d[:1] in "<>" else f"p = {d}"


def _weight_txt(weight_pct: Optional[float]) -> str:
    """Bileşik indeks kurulamadıysa (weight_pct None) ağırlık gösterilmez."""
    if weight_pct is None:
        return L("Weight: — (no composite index)", "Ağırlık: — (bütünleşik indeks yok)")
    return L(f"Weight: {weight_pct:.1f}%", f"Ağırlık: %{weight_pct:.1f}")


def _hist_period_label(df: pd.DataFrame) -> str:
    return f"{int(df['year'].min())}–{int(df['year'].max())}"


def guard_forecast(fc_df: Optional[pd.DataFrame], hist: np.ndarray, end_year: int, latest_year: int) -> tuple:
    """
    Gösterilecek tahmini ve kullanıcıya söylenecek notu belirler. Model seçimi, geçmiş veri zarfı denetimi ve
    yedek model istatistik motorunda yapılır (generate_5year_sarima_forecast); burada yalnızca:
    (1) seçilen bitiş yılından sonra gerçek gözlem varsa tahmin gösterilmez;
    (2) motor boş tablo döndürdüyse motorun gerekçesi (attrs["reason"]) iletilir;
    (3) aksi halde model adı, tarih aralığı ve motorun beceri notu (attrs["skill_note"]) döner.
    hist bilgi amaçlı tutulur (imza uyumu). Dönüş: (fc_df veya None, açıklama).
    """
    if end_year < latest_year:
        return None, L(
            f"Real observations exist after the selected end year ({end_year}); no forecast is shown.",
            f"Seçilen bitiş yılından ({end_year}) sonra gerçek gözlemler mevcut; tahmin gösterilmiyor.",
        )
    if fc_df is None or fc_df.empty:
        reason = fc_df.attrs.get("reason", "") if fc_df is not None else ""
        return None, L(f"No forecast shown{': ' + reason if reason else '.'}", f"Tahmin gösterilmiyor{': ' + reason if reason else '.'}")
    model = fc_df.attrs.get("model_name") or str(fc_df["model_name"].iloc[0])
    f0, f1 = pd.Timestamp(fc_df['date'].iloc[0]), pd.Timestamp(fc_df['date'].iloc[-1])
    note = L(f"Forecast model: {model} ({f0:%Y-%m} – {f1:%Y-%m}). ", f"Tahmin modeli: {model} ({f0:%Y-%m} – {f1:%Y-%m}). ") + (
        f"{fc_df.attrs.get('skill_note', '')}"
    )
    if fc_df["forecast"].isna().any():
        note += L(" No forecast is given for calendar months never observed in the record.",
                  " Geçmişte hiç gözlenmemiş takvim ayları için tahmin boş bırakıldı.")
    if fc_df.attrs.get("clipped"):
        note += L(
            " The forecast and its 95% band were clipped at the variable's physical limit (e.g. fraction 0–1, "
            "precipitation/radiation ≥ 0); where clipped, the band is no longer the model's symmetric 95% interval.",
            " Tahmin ve %95 bandı değişkenin fiziksel sınırında (ör. oran 0–1, yağış/ışınım ≥ 0) kırpıldı; "
            "kırpılan kısımda bant artık modelin simetrik %95 aralığı değildir.",
        )
    return fc_df, note.strip()


def _add_forecast(fig: go.Figure, fc_df: Optional[pd.DataFrame], color: str, name: str, show_band_legend: bool = True) -> None:
    """Tahmin çizgisi + %95 bandı. Bant, NaN (gözlenmemiş takvim ayı) boşluklarında ayrı parçalar halinde çizilir."""
    if fc_df is None or fc_df.empty:
        return
    model = fc_df["model_name"].iloc[0]
    skill = fc_df.attrs.get("skill_note", "")
    ok = (fc_df["lower_95"].notna() & fc_df["upper_95"].notna()).to_numpy()
    seg_id = np.cumsum(~ok)
    first = True
    for s in np.unique(seg_id[ok]):
        part = fc_df[ok & (seg_id == s)]
        fig.add_trace(
            go.Scatter(
                x=pd.concat([part["date"], part["date"][::-1]]),
                y=pd.concat([part["upper_95"], part["lower_95"][::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.16),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=L(f"{name} 95% band ({model})", f"{name} %95 Bandı ({model})"),
                legendgroup=f"band_{name}",
                showlegend=show_band_legend and first,
            )
        )
        first = False
    fig.add_trace(
        go.Scatter(
            x=fc_df["date"],
            y=fc_df["forecast"],
            mode="lines",
            connectgaps=False,
            name=L(f"{name} — statistical extrapolation: {model} ({forecast_period_label(fc_df)})",
                   f"{name} — istatistiksel uzatma: {model} ({forecast_period_label(fc_df)})"),
            line=dict(color=color, width=2.6, dash="dot"),
            hovertemplate=f"%{{x|%Y-%m}}: %{{y:.3f}}<br>{model}<br>{skill}<extra></extra>",
        )
    )


# ---------------------------------------------------------------------------
# TEKLİ BÖLGE
# ---------------------------------------------------------------------------
def plot_single_region_composite_and_forecast(
    region_name: str,
    comp_df: pd.DataFrame,
    mk_stats: Dict[str, Any],
    forecast_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    hide_seasonality: bool = True,
) -> go.Figure:
    """Tekli bölgenin ağırlıklı bütünleşik anomali indeksi (z) + Sen's Slope + korumalı 60 aylık istatistiksel uzatma."""
    fig = go.Figure()
    y_hist = comp_df["value"].to_numpy(dtype=float)
    rolling_12 = pd.Series(y_hist).rolling(window=12, min_periods=1).mean()
    w_items = [L(f"{r['label']} ({r['weight_pct']:.1f}%)", f"{r['label']} (%{r['weight_pct']:.1f})") for _, r in weights_df.iterrows()]

    fig.add_trace(go.Scatter(x=comp_df["date"], y=y_hist, mode="lines", name=L("Monthly composite index", "Aylık Bütünleşik İndeks"),
                             line=dict(color="rgba(56,189,248,0.45)", width=1.2)))
    fig.add_trace(go.Scatter(x=comp_df["date"], y=rolling_12, mode="lines", name=L("12-month moving average", "12 Aylık Hareketli Ortalama"),
                             line=dict(color="#38BDF8", width=2.8)))
    if len(mk_stats.get("trend_line", [])) == len(comp_df):
        fig.add_trace(go.Scatter(
            x=comp_df["date"], y=mk_stats["trend_line"], mode="lines",
            name=L(f"Sen's slope ({mk_stats['sen_slope_annual']:+.4f} z/yr | {_p_label(mk_stats.get('p_value_display'))})",
                   f"Sen's Slope ({mk_stats['sen_slope_annual']:+.4f} z/yıl | {_p_label(mk_stats.get('p_value_display'))})"),
            line=dict(color="#FACC15", width=2.4, dash="dash"),
        ))
    _add_forecast(fig, forecast_df, "#10B981", L("Composite index", "Bütünleşik İndeks"))

    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(148,163,184,0.4)")
    w_lines = _wrap_join(w_items)
    w_lines[0] = L("Weights: ", "Ağırlıklar: ") + w_lines[0]
    fig.update_layout(
        title=_title(
            L(f"🌍 <b>{region_name} — Composite Anomaly Index ({_hist_period_label(comp_df)})</b>",
              f"🌍 <b>{region_name} — Bütünleşik Anomali İndeksi ({_hist_period_label(comp_df)})</b>"),
            L(f"+ statistical extrapolation ({forecast_period_label(forecast_df)})",
              f"+ İstatistiksel Uzatma ({forecast_period_label(forecast_df)})"),
            *w_lines,
        ),
        height=500,
        xaxis_title=L("Date", "Tarih"),
        yaxis_title=L("Weighted z-score", "Ağırlıklı Z-Skoru"),
        margin=dict(l=55, r=30, t=40, b=30),
        **_LAYOUT_BASE,
    )
    return fig


def plot_single_parameter_chart(
    region_name: str,
    var_code: str,
    stl_res: Dict[str, Any],
    mk: Dict[str, Any],
    fc_df: Optional[pd.DataFrame],
    meta: Dict[str, Any],
    weight_pct: float,
    hide_seasonality: bool = True,
    color: str = "#38BDF8",
) -> go.Figure:
    """
    Tekli bölgede TEK BİR parametrenin kendi fiziksel birimindeki AYRI grafiği:
    gözlem (ham veya mevsimsellikten arındırılmış) + 13 aylık merkezli ortalama (görsel) + Sen eğimi + istatistiksel uzatma.
    """
    stl_df = stl_res["df"]
    fig = go.Figure()
    y_col = "deseasonalized" if hide_seasonality else "observed"

    fig.add_trace(go.Scatter(
        x=stl_df["date"], y=stl_df[y_col], mode="lines",
        name=(L("Deseasonalized series (monthly climatology removed)", "Mevsimsellikten Arındırılmış Seri (aylık klimatoloji çıkarıldı)")
              if hide_seasonality else L("Raw monthly observation", "Ham Aylık Gözlem")),
        line=dict(color=_hex_to_rgba(color, 0.55), width=1.3),
        hovertemplate=f"%{{x|%Y-%m}}: %{{y:.3f}} {meta['unit']}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=stl_df["date"], y=stl_df["trend"], mode="lines",
        name=L("Trend (13-month centred mean, visual only)", "Trend (13 aylık merkezli ortalama, yalnızca görsel)"),
        line=dict(color=color, width=2.8),
    ))
    if len(mk.get("trend_line", [])) == len(stl_df):
        fig.add_trace(go.Scatter(
            x=stl_df["date"], y=mk["trend_line"], mode="lines",
            name=L(f"Sen's slope ({mk['sen_slope_annual']:+.4f} {meta['unit']}/yr)", f"Sen's Slope ({mk['sen_slope_annual']:+.4f} {meta['unit']}/yıl)"),
            line=dict(color="#FACC15", width=2.2, dash="dash"),
        ))
    _add_forecast(fig, fc_df, color, meta["short_label"])

    fig.update_layout(
        title=_title(
            f"<b>{meta['label']}</b> — {region_name}",
            L(f"{_weight_txt(weight_pct)} | Sen's slope: {mk['sen_slope_annual']:+.4f} {meta['unit']}/yr "
              f"({mk['sen_slope_decade']:+.3f} per decade) | τ={mk['kendall_tau']:+.3f} | {_p_label(mk.get('p_value_display'))}",
              f"{_weight_txt(weight_pct)} | Sen's Slope: {mk['sen_slope_annual']:+.4f} {meta['unit']}/yıl "
              f"(10 yılda {mk['sen_slope_decade']:+.3f}) | τ={mk['kendall_tau']:+.3f} | {_p_label(mk.get('p_value_display'))}"),
            f"{mk['badge_text']}",
        ),
        height=450,
        xaxis_title=L("Date", "Tarih"),
        yaxis_title=meta["unit"],
        margin=dict(l=60, r=30, t=40, b=30),
        **_LAYOUT_BASE,
    )
    return fig


# ---------------------------------------------------------------------------
# ÇİFTLİ / ÇOKLU BÖLGE — BÜTÜNLEŞİK GRAFİKLER
# ---------------------------------------------------------------------------
def plot_dual_region_integrated_composite(
    name_a: str,
    name_b: str,
    comp_a: pd.DataFrame,
    comp_b: pd.DataFrame,
    fc_a: pd.DataFrame,
    fc_b: pd.DataFrame,
    tele_res: Dict[str, Any],
    weights_df: pd.DataFrame,
    hide_seasonality: bool = True,
) -> go.Figure:
    """Çiftli modda EN ÜSTTEKİ bütünleşik zıtlık/ilişki grafiği: tam olarak 2 ana çizgi (A ve B)."""
    fig = go.Figure()
    w_items = [L(f"{r['label']} ({r['weight_pct']:.1f}%)", f"{r['label']} (%{r['weight_pct']:.1f})") for _, r in weights_df.iterrows()]
    mk_a = tele_res["region_stats"][name_a]
    mk_b = tele_res["region_stats"][name_b]

    for name, comp, mk, fc, color in [
        (name_a, comp_a, mk_a, fc_a, REGION_PALETTE[0]),
        (name_b, comp_b, mk_b, fc_b, REGION_PALETTE[1]),
    ]:
        roll = pd.Series(comp["value"]).rolling(window=12, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=comp["date"], y=roll, mode="lines",
            name=L(f"{name} (Sen's slope: {mk['sen_slope_annual']:+.4f}/yr)", f"{name} (Sen's Eğim: {mk['sen_slope_annual']:+.4f}/yıl)"),
            line=dict(color=color, width=3.2),
        ))
        if len(mk.get("trend_line", [])) == len(comp):
            fig.add_trace(go.Scatter(
                x=comp["date"], y=mk["trend_line"], mode="lines",
                name=f"{name} — Theil-Sen Trend", line=dict(color=color, width=1.8, dash="dash"),
                showlegend=False,
            ))
        _add_forecast(fig, fc, color, name, show_band_legend=False)

    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(148,163,184,0.45)")
    rho_txt = "" if tele_res.get("relationship") == "identical" else f"Spearman ρ = {tele_res['spearman_rho']:+.3f} | "
    w_lines = _wrap_join(w_items)
    w_lines[0] = L("Shared weights: ", "Ortak Ağırlıklar: ") + w_lines[0]
    fig.update_layout(
        title=_title(
            L(f"🌐 <b>Two-Region Composite Comparison ({_hist_period_label(comp_a)})</b>",
              f"🌐 <b>BÜTÜNLEŞİK ÇİFT BÖLGE ZITLIK / İLİŞKİ GRAFİĞİ ({_hist_period_label(comp_a)})</b>"),
            L(f"+ statistical extrapolation ({forecast_period_label(fc_a)})", f"+ İstatistiksel Uzatma ({forecast_period_label(fc_a)})"),
            f"{tele_res['verdict_badge']} | {rho_txt}" + L(f"ΔSlope = {tele_res['delta_slope']:+.4f}/yr", f"ΔSlope = {tele_res['delta_slope']:+.4f}/yıl"),
            *w_lines,
        ),
        height=540,
        xaxis_title=L("Date", "Tarih"),
        yaxis_title=L("Shared weighted composite index (z-score)", "Ortak Ağırlıklı Bütünleşik İndeks (Z-Skoru)"),
        margin=dict(l=60, r=30, t=40, b=30),
        **_LAYOUT_BASE,
    )
    return fig


def plot_multi_region_comparison(
    region_composites: Dict[str, pd.DataFrame],
    region_forecasts: Dict[str, pd.DataFrame],
    tele_res: Dict[str, Any],
) -> Tuple[go.Figure, go.Figure]:
    """Çoklu modda bütünleşik N çizgili grafik + bölgeler arası korelasyon ısı haritası."""
    fig_ts = go.Figure()
    r_names = list(region_composites.keys())
    first_fc = None

    for idx, r_name in enumerate(r_names):
        color = REGION_PALETTE[idx % len(REGION_PALETTE)]
        df_c = region_composites[r_name]
        roll_y = pd.Series(df_c["value"]).rolling(window=12, min_periods=1).mean()
        slope_r = tele_res["region_stats"].get(r_name, {}).get("sen_slope_annual", 0.0)
        fig_ts.add_trace(go.Scatter(
            x=df_c["date"], y=roll_y, mode="lines",
            name=L(f"{r_name} (slope: {slope_r:+.4f}/yr)", f"{r_name} (Eğim: {slope_r:+.4f}/yıl)"), line=dict(color=color, width=2.6),
        ))
        fc_df = region_forecasts.get(r_name)
        if fc_df is not None and not fc_df.empty:
            first_fc = first_fc if first_fc is not None else fc_df
            fig_ts.add_trace(go.Scatter(
                x=fc_df["date"], y=fc_df["forecast"], mode="lines",
                name=L(f"{r_name} (forecast)", f"{r_name} (Tahmin)"), line=dict(color=color, width=2.0, dash="dot"), showlegend=False,
            ))

    any_df = region_composites[r_names[0]]
    fig_ts.add_hline(y=0.0, line_dash="dot", line_color="rgba(148,163,184,0.4)")
    fig_ts.update_layout(
        title=_title(
            L(f"🌐 <b>Multi-Region Composite Comparison ({len(r_names)} locations, {_hist_period_label(any_df)})</b>",
              f"🌐 <b>Bütünleşik Çoklu Bölge ({len(r_names)} Konum) Zıtlık / İlişki Grafiği ({_hist_period_label(any_df)})</b>"),
            L(f"+ statistical extrapolation ({forecast_period_label(first_fc)})",
              f"+ İstatistiksel Uzatma ({forecast_period_label(first_fc)})"),
        ),
        height=520,
        xaxis_title=L("Date", "Tarih"),
        yaxis_title=L("Shared weighted composite index (z-score)", "Ortak Ağırlıklı Bütünleşik İndeks (Z-Skoru)"),
        margin=dict(l=60, r=30, t=40, b=30),
        **_LAYOUT_BASE,
    )

    corr_df = tele_res["corr_df"]
    short = [n if len(n) <= 28 else n[:26] + "…" for n in corr_df.columns]
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_df.values, x=short, y=short, colorscale="RdBu", zmin=-1.0, zmax=1.0,
        text=np.round(corr_df.values, 2), texttemplate="%{text}", colorbar=dict(title="r"),
    ))
    fig_heat.update_layout(
        title=_title(L("🧭 <b>Composite Index — Correlation Matrix Between Regions</b>",
                       "🧭 <b>Bütünleşik İndeks — Bölgeler Arası Korelasyon Matrisi</b>")),
        template="plotly_dark", height=450, margin=dict(l=55, r=30, t=40, b=45),
        paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
    )
    return fig_ts, fig_heat


# ---------------------------------------------------------------------------
# ÇİFTLİ / ÇOKLU BÖLGE — PARAMETRE BAŞINA AYRI KIYAS GRAFİĞİ
# ---------------------------------------------------------------------------
def plot_parameter_regions_comparison(
    var_code: str,
    region_stl: Dict[str, Dict[str, Any]],
    region_mk: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    weight_pct: float,
    rel_stats: Dict[str, Any],
    hide_seasonality: bool = True,
) -> go.Figure:
    """
    TEK BİR parametre için tüm seçili bölgeleri (2 veya en fazla 10) kendi fiziksel biriminde
    aynı grafikte kıyaslar: her bölge için seri + Sen's Slope doğrusu.
    """
    fig = go.Figure()
    for idx, (name, stl_res) in enumerate(region_stl.items()):
        if not stl_res.get("valid", False):
            continue
        color = REGION_PALETTE[idx % len(REGION_PALETTE)]
        df = stl_res["df"]
        y = df["trend"] if hide_seasonality else df["observed"]
        mk = region_mk.get(name, {})
        fig.add_trace(go.Scatter(
            x=df["date"], y=y, mode="lines",
            name=L(f"{name} ({mk.get('sen_slope_annual', 0.0):+.4f} {meta['unit']}/yr)", f"{name} ({mk.get('sen_slope_annual', 0.0):+.4f} {meta['unit']}/yıl)"),
            line=dict(color=color, width=2.4),
            hovertemplate=f"{name}<br>%{{x|%Y-%m}}: %{{y:.3f}} {meta['unit']}<extra></extra>",
        ))
        if len(mk.get("trend_line", [])) == len(df):
            fig.add_trace(go.Scatter(
                x=df["date"], y=mk["trend_line"], mode="lines", showlegend=False,
                name=f"{name} Sen's Slope", line=dict(color=color, width=1.6, dash="dash"),
            ))

    if rel_stats.get("valid", False):
        rel_txt = (
            f"{rel_stats['verdict_badge']} | ρ = {rel_stats['spearman_rho']:+.3f} | "
            + L(f"ΔSlope (1st–2nd) = {rel_stats['delta_slope']:+.4f} {meta['unit']}/yr",
                f"ΔSlope (1.–2.) = {rel_stats['delta_slope']:+.4f} {meta['unit']}/yıl")
        )
    elif rel_stats.get("relationship") == "identical":
        rel_txt = rel_stats["verdict_badge"]
    else:
        rel_txt = L("Relationship could not be computed (not enough overlapping data)", "İlişki hesaplanamadı (yetersiz ortak veri)")

    y_kind = (L("13-month mean, deseasonalized", "13 aylık ort., arındırılmış") if hide_seasonality
              else L("raw observation", "Ham Gözlem"))
    fig.update_layout(
        title=_title(
            L(f"<b>{meta['label']}</b> — comparison across regions ({_weight_txt(weight_pct)})",
              f"<b>{meta['label']}</b> — Bölgeler Arası Ayrık Kıyas ({_weight_txt(weight_pct)})"),
            rel_txt,
        ),
        height=450,
        xaxis_title=L("Date", "Tarih"),
        yaxis_title=f"{meta['unit']} ({y_kind})",
        margin=dict(l=60, r=30, t=40, b=30),
        **_LAYOUT_BASE,
    )
    return fig


# ---------------------------------------------------------------------------
# DÜNYA HARİTASI: NASA POWER 5° NOKTA ÖRNEKLEMİ (ERKEN DÖNEM / GEÇ DÖNEM / DEĞİŞİM)
# Kaynak dosya scripts/build_global_grid.py ile üretilir; harita yalnızca bu dosyadaki değerleri ve
# kullanıcının seçtiği noktaların kendi serisinden aynı tam yıllarla hesaplanan ortalamaları çizer.
# ---------------------------------------------------------------------------
GRID_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "global_grid_power_5deg.csv")
GRID_JSON = GRID_CSV[:-4] + ".json"
_WATER_VARS = ("PRECTOTCORR", "GWETROOT", "EVLAND", "TQV")


@lru_cache(maxsize=1)
def load_global_grid() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """5° ızgara dosyasını ve köken kaydını okur (dosya yoksa FileNotFoundError)."""
    with open(GRID_JSON, encoding="utf-8") as fh:
        prov = json.load(fh)
    return pd.read_csv(GRID_CSV), prov


def point_period_mean(df: pd.DataFrame, y0: int, y1: int, col: str = "value") -> Optional[float]:
    """Nokta serisinin y0..y1 tam yıllarındaki 60 ayın ortalaması; tek bir ay eksikse None (ızgarayla aynı kural)."""
    if df is None or df.empty or col not in df.columns:
        return None
    sub = df[(df["year"] >= y0) & (df["year"] <= y1)].dropna(subset=[col])
    if len(sub) != (y1 - y0 + 1) * 12 or sub[["year", "month"]].duplicated().any():
        return None
    return round(float(sub[col].mean()), 4)


def _period_status(df: Optional[pd.DataFrame], y0: int, y1: int) -> str:
    """Seçilen noktanın bir dönem için neden değeri olmadığını söyler (yıldız üzerindeki açıklama)."""
    if df is None or df.empty or "value" not in df.columns:
        return L("no data for this variable at this point (undefined for this surface or not retrieved)",
                 "bu noktada bu değişken için veri yok (yüzeyde tanımsız veya alınamadı)")
    sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
    n_ok = int(sub["value"].notna().sum())
    if sub.empty or n_ok == 0:
        return L("no real months in this period (variable not available or undefined in these years)",
                 "bu dönemde hiç gerçek ay yok (değişken bu yıllarda yok veya tanımsız)")
    return L(f"only {n_ok} of the period's 60 months are valid → no value", f"dönemin 60 ayından yalnızca {n_ok}'ı geçerli → değer yok")


def _change_limits(change: np.ndarray) -> Tuple[float, float, float, bool]:
    """Değişim panelinin sıfır merkezli ölçeği: |değişim|'in 98. yüzdeliği. Döner: (sınır, gerçek min, gerçek max, kesildi mi)."""
    fin = change[np.isfinite(change)]
    if not fin.size:
        return 1.0, 0.0, 0.0, False
    c_abs = float(np.percentile(np.abs(fin), 98)) or 1.0
    c_min, c_max = float(fin.min()), float(fin.max())
    return c_abs, c_min, c_max, bool(c_min < -c_abs or c_max > c_abs)


def world_map_caption(var_code: str, analysis_range: Optional[Tuple[int, int]] = None) -> str:
    """Haritanın ne olduğunu tam olarak söyleyen açıklama (dönemler sabittir)."""
    grid, prov = load_global_grid()
    (e0, e1), (l0, l1) = prov["periods"][var_code]["early"], prov["periods"][var_code]["late"]
    txt = L(
        f"Map: POINT samples taken from the NASA POWER monthly point API at the cell centres of a regular 5° grid "
        f"(not area averages; built on {prov['built_utc'][:10]}). For this variable the early period is the "
        f"{e0}–{e1} mean and the late period the {l0}–{l1} mean; a cell is coloured only if all 60 months of the "
        f"period are valid, others stay blank (e.g. land variables over the ocean, cells with ice-covered or polar-night "
        f"months). The change panel is late − early. Each 5° cell represents a single point: single-cell changes "
        f"(especially precipitation) are locally noisy and can change sign at neighbouring points; read the broad "
        f"patterns, not individual cells. Stars are means of the selected points' own series over the same complete "
        f"years; they are not blended into the field.",
        f"Harita: NASA POWER aylık nokta API'sinden 5° düzenli ızgaranın hücre merkezlerinde alınmış NOKTA örnekleri "
        f"(alan ortalaması değildir; {prov['built_utc'][:10]} tarihinde üretildi). Bu değişken için erken dönem "
        f"{e0}–{e1}, geç dönem {l0}–{l1} ortalamasıdır; bir hücre yalnızca dönemin 60 ayının tamamı geçerliyse "
        f"boyanır, diğerleri (ör. okyanusta kara değişkenleri, buzlu/kutup gecesi ayları olan hücreler) boş kalır. "
        f"Değişim paneli geç − erken farkıdır. Her 5° hücre tek bir noktayı temsil eder: tek hücredeki değişimler "
        f"(özellikle yağış) yerel olarak gürültülüdür ve komşu noktalarda işaret değiştirebilir; tek tek hücreleri değil "
        f"geniş örüntüleri okuyun. Yıldızlar seçilen noktaların kendi serisinden aynı tam yıllarla hesaplanan "
        f"ortalamalardır; alana karıştırılmaz."
    )
    c_abs, c_min, c_max, capped = _change_limits(
        grid[f"{var_code}_late"].to_numpy(dtype=float) - grid[f"{var_code}_early"].to_numpy(dtype=float)
    )
    if capped:
        txt += L(
            f" The change colour scale is capped at ±{c_abs:.2f} (98th percentile of |change|); the real range is "
            f"{c_min:+.2f} … {c_max:+.2f} — the most extreme 2% of cells are drawn in the end colour of the scale, "
            f"their values appear on hover.",
            f" Değişim renk ölçeği ±{c_abs:.2f}'de kesilmiştir (|değişim|'in 98. yüzdeliği); gerçek aralık "
            f"{c_min:+.2f} … {c_max:+.2f} — en uç %2 hücre ölçeğin uç rengiyle çizilir, değerleri üzerine gelince görünür."
        )
    if analysis_range is not None and (int(analysis_range[0]) > e0 or int(analysis_range[1]) < l1):
        txt += L(
            f" Note: the map periods are fixed ({e0}–{e1} and {l0}–{l1}) and extend beyond your analysis range "
            f"({analysis_range[0]}–{analysis_range[1]}); the map does not change with that range.",
            f" Not: harita dönemleri sabittir ({e0}–{e1} ve {l0}–{l1}) ve seçtiğiniz analiz aralığının "
            f"({analysis_range[0]}–{analysis_range[1]}) dışına taşar; harita bu aralığa göre değişmez."
        )
    return txt


# Haritalar alt alta 3 panel: her panel tam genişlikte, yüksekliği sabit. Konteyner genişliği panel yüksekliğinin
# en az 2 katı olduğunda (eşdikdörtgen 360°x180°) ölçek yüksekliğe bağlıdır, bu yüzden 5° hücre karesinin piksel
# boyutu sabittir ve hücreler boşluksuz/örtüşmesiz bitişir.
_MAP_HEIGHT = 1020
_MAP_MARGIN_T, _MAP_MARGIN_B = 115, 10  # +20 px headroom under the subtitle; panel heights unchanged
_MAP_VSPACE = 0.05


def _cell_marker_px() -> float:
    row_px = (_MAP_HEIGHT - _MAP_MARGIN_T - _MAP_MARGIN_B) * (1.0 - 2 * _MAP_VSPACE) / 3.0
    return round(5.0 * row_px / 180.0, 2)


def plot_world_period_change_map(
    var_code: str,
    title_label: str,
    unit_label: str,
    points: List[Dict[str, Any]],
    analysis_range: Optional[Tuple[int, int]] = None,
) -> Optional[go.Figure]:
    """
    3 panel (alt alta): erken dönem, geç dönem (aynı sıralı Viridis ölçeği) ve geç − erken değişim (sıfır merkezli
    ıraksak ölçek, 98. yüzdelikte kesilirse renk çubuğunda gerçek min/max yazar). points: [{"name", "lat", "lon",
    "df"(year, month, value)}] — kendi seri ortalamaları işaretçi olarak çizilir. Izgarada bu değişken yoksa
    (RUNOFF, bütünleşik indeks) None döner.
    """
    grid, prov = load_global_grid()
    if var_code not in prov["periods"]:
        return None
    (e0, e1), (l0, l1) = prov["periods"][var_code]["early"], prov["periods"][var_code]["late"]
    early = grid[f"{var_code}_early"].to_numpy(dtype=float)
    late = grid[f"{var_code}_late"].to_numpy(dtype=float)
    change = late - early

    pts = []
    for p in points:
        pe = point_period_mean(p.get("df"), e0, e1)
        pl = point_period_mean(p.get("df"), l0, l1)
        pts.append({
            **p, "early": pe, "late": pl, "change": (pl - pe) if pe is not None and pl is not None else None,
            "why_early": _period_status(p.get("df"), e0, e1), "why_late": _period_status(p.get("df"), l0, l1),
        })

    lv = np.concatenate([early[np.isfinite(early)], late[np.isfinite(late)]])
    z_min, z_max = (float(np.min(lv)), float(np.max(lv))) if lv.size else (0.0, 1.0)
    c_abs, c_min, c_max, capped = _change_limits(change)
    div_scale = "BrBG" if var_code in _WATER_VARS else "RdBu_r"
    c_title = f"Δ {unit_label}" + (
        L(f"<br>scale capped at ±{c_abs:.2f}<br>real: {c_min:+.2f} … {c_max:+.2f}",
          f"<br>ölçek ±{c_abs:.2f}'de kesildi<br>gerçek: {c_min:+.2f} … {c_max:+.2f}") if capped else "")
    msize = _cell_marker_px()

    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"type": "geo"}], [{"type": "geo"}], [{"type": "geo"}]],
        subplot_titles=(
            L(f"Early-period mean {e0}–{e1}", f"Erken dönem ortalaması {e0}–{e1}"),
            L(f"Late-period mean {l0}–{l1}", f"Geç dönem ortalaması {l0}–{l1}"),
            L(f"Change (late − early): {l0}–{l1} minus {e0}–{e1}", f"Değişim (geç − erken): {l0}–{l1} eksi {e0}–{e1}"),
        ),
        vertical_spacing=_MAP_VSPACE,
    )
    panels = [
        (1, early, "coloraxis", f"{e0}–{e1}", "early", "why_early"),
        (2, late, "coloraxis", f"{l0}–{l1}", "late", "why_late"),
        (3, change, "coloraxis2", L(f"Change {l0}–{l1} minus {e0}–{e1}", f"Değişim {l0}–{l1} eksi {e0}–{e1}"), "change", None),
    ]
    for row, vals, cax, tag, key, why in panels:
        ok = np.isfinite(vals)
        fig.add_trace(
            go.Scattergeo(
                lat=grid["lat"].to_numpy()[ok],
                lon=grid["lon"].to_numpy()[ok],
                mode="markers",
                marker=dict(symbol="square", size=msize, color=vals[ok], coloraxis=cax, line=dict(width=0)),
                name=L(f"POWER 5° grid ({tag})", f"POWER 5° ızgara ({tag})"),
                hovertemplate=L(
                    f"<b>NASA POWER point sample ({tag})</b><br>Cell centre: %{{lat:.1f}}°, %{{lon:.1f}}°<br>"
                    f"Value: %{{marker.color:.3f}} {unit_label}<extra></extra>",
                    f"<b>NASA POWER nokta örneği ({tag})</b><br>Hücre merkezi: %{{lat:.1f}}°, %{{lon:.1f}}°<br>"
                    f"Değer: %{{marker.color:.3f}} {unit_label}<extra></extra>",
                ),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        shown = [p for p in pts if p[key] is not None]
        missing = [p for p in pts if p[key] is None]
        if shown:
            fig.add_trace(
                go.Scattergeo(
                    lat=[p["lat"] for p in shown],
                    lon=[p["lon"] for p in shown],
                    mode="markers+text",
                    marker=dict(symbol="star", size=13, color="#FFFFFF", line=dict(color="#0F172A", width=1.5)),
                    text=[f"{p[key]:+.2f}" if key == "change" else f"{p[key]:.2f}" for p in shown],
                    textposition="top center",
                    textfont=dict(color="#FFFFFF", size=11, shadow="0px 0px 3px #000000"),
                    customdata=[p["name"] for p in shown],
                    name=L(f"Selected point ({tag})", f"Seçilen nokta ({tag})"),
                    hovertemplate=L(
                        f"<b>%{{customdata}}</b><br>Own NASA POWER series, {tag}: %{{text}} {unit_label}"
                        f"<br>Location: %{{lat:.2f}}°, %{{lon:.2f}}°<extra></extra>",
                        f"<b>%{{customdata}}</b><br>Kendi NASA POWER serisi, {tag}: %{{text}} {unit_label}"
                        f"<br>Konum: %{{lat:.2f}}°, %{{lon:.2f}}°<extra></extra>",
                    ),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        if missing:
            reasons = [
                p[why] if why else "; ".join(dict.fromkeys([p["why_early"], p["why_late"]])) for p in missing
            ]
            fig.add_trace(
                go.Scattergeo(
                    lat=[p["lat"] for p in missing],
                    lon=[p["lon"] for p in missing],
                    mode="markers",
                    marker=dict(symbol="star-open", size=13, color="#FFFFFF"),
                    customdata=np.array([[p["name"], r] for p, r in zip(missing, reasons)], dtype=object),
                    name=L(f"Selected point, no complete-period data ({tag})", f"Seçilen nokta, tam dönem verisi yok ({tag})"),
                    hovertemplate=f"<b>%{{customdata[0]}}</b><br>{tag}: %{{customdata[1]}}<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

    fig.update_geos(
        projection_type="equirectangular",
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.7)",
        coastlinewidth=0.6,
        showland=True,
        landcolor="#2B3440",
        showocean=True,
        oceancolor="#111827",
        showframe=False,
        lataxis_range=[-90, 90],
        lonaxis_range=[-180, 180],
        bgcolor="#0E1117",
    )
    fig.update_layout(
        title=dict(
            text=L(
                f"🗺️ <b>{title_label} — NASA POWER 5° point samples: {e0}–{e1} and {l0}–{l1} means, change</b><br>"
                f"<span style='font-size:12px;color:#CBD5E1;'>Point samples (not area averages); blank cell = "
                f"not all 60 months valid in that period, or variable undefined on that surface. Periods are fixed.</span>",
                f"🗺️ <b>{title_label} — NASA POWER 5° nokta örnekleri: {e0}–{e1} ve {l0}–{l1} ortalamaları, değişim</b><br>"
                f"<span style='font-size:12px;color:#CBD5E1;'>Nokta örnekleri (alan ortalaması değil); boş hücre = "
                f"o dönemde 60 ayın tamamı geçerli değil veya değişken bu yüzeyde tanımsız. Dönemler sabittir.</span>",
            ),
            x=0.01,
        ),
        coloraxis=dict(colorscale="Viridis", cmin=z_min, cmax=z_max,
                       colorbar=dict(title=unit_label, thickness=12, len=0.6, y=0.69)),
        coloraxis2=dict(colorscale=div_scale, cmin=-c_abs, cmax=c_abs, cmid=0.0,
                        colorbar=dict(title=c_title, thickness=12, len=0.3, y=0.15)),
        template="plotly_dark",
        height=_MAP_HEIGHT,
        margin=dict(l=10, r=110, t=_MAP_MARGIN_T, b=_MAP_MARGIN_B),
        paper_bgcolor="#0E1117",
    )
    return fig
