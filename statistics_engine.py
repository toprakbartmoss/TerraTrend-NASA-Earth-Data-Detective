"""
NASA Earth Data Detective (EDD) - İstatistiksel Analiz, Mevsimsel Ayrıştırma, CRITIC Ağırlıklandırma
ve 60 Aylık İstatistiksel Projeksiyon Motoru (statistics_engine.py)

Yöntemler (yanlış-pozitif oranı ve kapsama scripts/mc_validate_stats.py ile Monte Carlo doğrulandı):
1. Mevsimsellikten arındırma: seri eksiksiz aylık takvim eksenine yerleştirilir, her takvim ayının uzun
   dönem ortalaması (aylık klimatoloji) çıkarılır. Eksik aylar eksik kalır; hiçbir değer doldurulmaz.
2. Trend testi: Mann-Kendall S istatistiği + Theil-Sen medyan eğimi (Sen, 1968). Var(S), artıklardan
   hesaplanan üç standart bağımlılık tahmininin EN BÜYÜĞÜYLE büyütülür (her biri doğrusal trend eğiminin
   bağımlılık altındaki varyans oranı): (a) takvim gecikme-1 AR(1) (Yue & Wang, 2004), (b) uzun hafıza:
   yerel Whittle d -> ARFIMA(0,d,0) (Robinson, 1995), (c) eşit ağırlıklı kosinüs (EWC) düşük frekans uzun dönem
   varyansı (Lazarus vd., 2018). p ve Sen eğiminin %95 güven aralığı aynı varyansla, B = 0.4 T^(2/3) serbestlik
   dereceli t dağılımından alınır. "Anlamlı" kararı yalnızca p < alpha VE güven aralığı sıfırı dışlıyorsa verilir
   (bağlı değerlerde ikisi ayrışabildiği için karar ve aralık yapı gereği tutarlıdır).
3. Bileşik indeks: her değişkenin sabit işareti meta veri "polarity" alanından gelir; ağırlıklar
   CRITIC (Diakoulaki vd., 1995): C_j = sigma_j * sum_k (1 - |r_jk|), normalize. VIF yalnızca tanı amaçlıdır.
4. Bölgeler arası ilişki: arındırılmış ve doğrusal trendi çıkarılmış seriler arasında Spearman/Pearson
   korelasyonu; p-değeri Bretherton vd. (1999) etkin örneklem büyüklüğüyle.
5. Projeksiyon: doğrusal trend + durağan AR(1) (statsmodels SARIMAX). Model yakınsamaz veya geçmiş
   veri zarfının dışına çıkarsa son 5 yılın aylık ortalaması kullanılır; ikisi de geçersizse boş sonuç
   ve gerekçe döner. Son 60 ay üzerinde geriye dönük sınama (backtest) ile beceri raporlanır.
"""

from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from statsmodels.tsa.statespace.sarimax import SARIMAX

from i18n import L, LocalizedMapping

# Bölge ilişkisi için en az ortak gözlem ayı (5 yıl).
MIN_OVERLAP_MONTHS = 60
# Bundan kısa aylık serilerde (10 yıl) trend kararı "güvenilir değil" olarak işaretlenir. Yanlış-pozitif
# oranları scripts/mc_validate_stats.py ile ölçülür (AR(1), AR(2), ARFIMA, bağlı değerler).
MIN_MONTHS_RELIABLE = 120
MIN_YEARS_RELIABLE = 10
# Bileşik indeks yalnızca ortak aylar, en az bir değişkenin gözlendiği ayların bu oranına ulaşırsa kurulur
# (data_loader'daki %70 doluluk kuralıyla aynı eşik).
MIN_COMPOSITE_OVERLAP = 0.70


def decimal_years(df: pd.DataFrame) -> np.ndarray:
    """Aylık DataFrame için gerçek tarih ekseni: yıl + (ay - 1) / 12."""
    return df["year"].to_numpy(dtype=float) + (df["month"].to_numpy(dtype=float) - 1.0) / 12.0


def _remove_calendar_climatology(y: np.ndarray, month_number: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    y (sonlu değerler) ve tamsayı ay numaraları (yıl*12 + ay-1) için her takvim ayının ortalamasını çıkarır.
    Döner: (deseasonalized, seasonal); seasonal'ın mevcut takvim ayları üzerinden ortalaması 0'dır,
    böylece arındırılmış seri gözlem ile aynı seviyede kalır.
    """
    moy = np.asarray(month_number) % 12
    clim = np.full(12, np.nan)
    for m in np.unique(moy):
        clim[m] = float(np.mean(y[moy == m]))
    seasonal = (clim - np.nanmean(clim))[moy]
    return y - seasonal, seasonal


def _lag1_calendar(resid: np.ndarray, steps: np.ndarray) -> float:
    """
    Gecikme-1 otokorelasyonu: yalnızca takvimde ardışık (1 ay / 1 yıl arayla) gözlem çiftleri kullanılır.
    Küçük örneklem yanlılığı r + (1 + 4r)/n ile düzeltilir; AR katsayısı [0, 0.99] aralığında tutulur
    (negatif tahmin 0 alınır: düzeltme hiçbir zaman testi liberalleştirmez).
    """
    n = len(resid)
    resid = resid - np.mean(resid)
    ok = np.diff(steps) == 1
    denom = float(np.sum(resid ** 2))
    if ok.sum() < 3 or denom <= 0:
        return 0.0
    r = (float(np.sum(resid[:-1][ok] * resid[1:][ok])) / ok.sum()) / (denom / n)
    r = r + (1.0 + 4.0 * r) / n
    return float(min(max(r, 0.0), 0.99))


def _trend_variance_factor(r: float, steps: np.ndarray) -> float:
    """AR(1) katsayısı r altında, gerçek takvim zamanlarında doğrusal trend eğiminin varyansı / bağımsız varyans."""
    if r <= 0:
        return 1.0
    t = steps.astype(float) - float(np.mean(steps))
    w = t / float(np.sum(t ** 2))
    sigma = r ** np.abs(steps[:, None] - steps[None, :]).astype(float)
    return float(w @ sigma @ w / np.sum(w ** 2))


def _rho_trend_factor(rho: np.ndarray, steps: np.ndarray) -> float:
    """rho[k] = gecikme-k otokorelasyonu iken doğrusal trend eğiminin varyansı / bağımsız varyans (takvim zamanlarında)."""
    t = steps.astype(float) - float(np.mean(steps))
    w = t / float(np.sum(t ** 2))
    sigma = rho[np.abs(steps[:, None] - steps[None, :])]
    return float(w @ sigma @ w / np.sum(w ** 2))


def _long_memory_factor(resid: np.ndarray, steps: np.ndarray) -> Tuple[float, float]:
    """
    Uzun hafıza: artıkların yerel Whittle d tahmini (Robinson 1995; m = T^0.65 en düşük frekans, eksik aylar 0
    alınır) ve ARFIMA(0,d,0) otokorelasyonuyla trend varyans oranı. d [0, 0.45] aralığında tutulur. Döner: (çarpan, d).
    """
    span = int(steps[-1] - steps[0]) + 1
    x = np.zeros(span)
    x[steps - steps[0]] = resid - np.mean(resid)
    m = int(span ** 0.65)
    if m < 4:
        return 1.0, 0.0
    lam = 2.0 * np.pi * np.arange(1, m + 1) / span
    per = np.abs(np.fft.fft(x)[1 : m + 1]) ** 2
    if not np.any(per > 0):
        return 1.0, 0.0

    def _whittle(d: float) -> float:
        return float(np.log(np.mean(lam ** (2 * d) * per)) - 2 * d * np.mean(np.log(lam)))

    d = float(np.clip(minimize_scalar(_whittle, bounds=(-0.49, 0.49), method="bounded").x, 0.0, 0.45))
    if d <= 0:
        return 1.0, 0.0
    k = np.arange(span)
    rho = np.exp(gammaln(k + d) + gammaln(1 - d) - gammaln(k - d + 1) - gammaln(d))
    rho[0] = 1.0
    return max(1.0, _rho_trend_factor(rho, steps)), d


def _ewc_factor(resid: np.ndarray, steps: np.ndarray) -> Tuple[float, int]:
    """
    Eşit ağırlıklı kosinüs (EWC) uzun dönem varyansı (Lazarus, Lewis, Stock & Watson 2018): eğim skorunun
    (t - t̄) * artık ilk B = 0.4 T^(2/3) kosinüs izdüşümü. Döner: (trend varyans oranı, B); B, testin t dağılımı
    serbestlik derecesidir.
    """
    span = int(steps[-1] - steps[0]) + 1
    x = steps.astype(float) - float(np.mean(steps))
    u = resid - np.mean(resid)
    b = max(4, int(0.4 * span ** (2.0 / 3.0)))
    denom = float(np.mean(u ** 2) * np.sum(x ** 2))
    if denom <= 0:
        return 1.0, b
    pos = (steps - steps[0]).astype(float) + 0.5
    proj = np.sqrt(2.0 / span) * (np.cos(np.pi * np.arange(1, b + 1)[:, None] * pos[None, :] / span) @ (x * u))
    return max(1.0, span * float(np.mean(proj ** 2)) / denom), b


def _format_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _p_phrase(p: float) -> str:
    """Metin içinde: 'p < 0.001' veya 'p = 0.012' (asla 'p = < 0.001')."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def compute_shared_weights(
    region_stl_maps: List[Dict[str, Dict[str, Any]]],
    selected_vars: List[str],
    official_variables_meta: Dict[str, Dict[str, Any]],
    hide_seasonality: bool = True,
) -> Dict[str, Any]:
    """
    Çiftli/Çoklu bölge kıyasında TÜM bölgeler için ORTAK ağırlık üretir.
    Her bölgede CRITIC ağırlıkları hesaplanır, bölgeler arası ortalaması alınır
    ve yeniden 1.00'e normalize edilir. Böylece bütünleşik çizgiler aynı ağırlık setiyle
    (elma-elma) karşılaştırılır. Yalnızca tüm bölgelerde geçerli verisi olan parametreler kullanılır.
    """
    common_vars = [
        v for v in selected_vars
        if v in official_variables_meta and all(v in m and m[v].get("valid", False) for m in region_stl_maps)
    ]
    dropped = [v for v in selected_vars if v not in common_vars]
    if not common_vars:
        return {"valid": False, "weights_df": pd.DataFrame(), "vif_warnings": [], "common_vars": [], "dropped_vars": dropped,
                "reason": L("None of the selected parameters has valid data in all regions.", "Seçilen parametrelerin hiçbiri tüm bölgelerde geçerli veri içermiyor.")}

    per_region = [
        compute_objective_weights_and_composite(m, common_vars, official_variables_meta, hide_seasonality)
        for m in region_stl_maps
    ]
    failed = [r["reason"] for r in per_region if not r["valid"]]
    if failed:
        # Bir bölgede bile bileşik indeks kurulamıyorsa ortak indeks yok (sansürlü alt küme kullanılmaz).
        return {"valid": False, "weights_df": pd.DataFrame(), "vif_warnings": [], "common_vars": common_vars,
                "dropped_vars": dropped, "reason": failed[0]}
    stacked = pd.concat([r["weights_df"] for r in per_region], ignore_index=True)

    agg = (
        stacked.groupby("var_code")
        .agg(
            rank=("rank", "first"),
            label=("label", "first"),
            ipcc_base_weight=("ipcc_base_weight", "first"),
            team_importance_score=("team_importance_score", "first"),
            critic_score=("critic_score", "mean"),
            vif_value=("vif_value", "max"),
            normalized_weight=("normalized_weight", "mean"),
            polarity_sign=("polarity_sign", "first"),
        )
        .reset_index()
        .sort_values("rank")
        .reset_index(drop=True)
    )
    total = float(agg["normalized_weight"].sum())
    agg["normalized_weight"] = (agg["normalized_weight"] / total).round(4) if total > 0 else 1.0 / len(agg)
    agg["weight_pct"] = (agg["normalized_weight"] * 100.0).round(2)
    agg["critic_score"] = agg["critic_score"].round(3)
    agg["vif_status"] = agg["vif_value"].apply(_vif_status_text)

    vif_warnings: List[str] = []
    for r in per_region:
        vif_warnings.extend(r["vif_warnings"])

    return {
        "valid": True,
        "reason": "",
        "weights_df": agg,
        "vif_warnings": list(dict.fromkeys(vif_warnings)),
        "common_vars": common_vars,
        "dropped_vars": dropped,
        "index_description": per_region[0]["index_description"],
        "weight_method": per_region[0]["weight_method"],
    }


def apply_weights_to_composite(
    param_stl_map: Dict[str, Dict[str, Any]],
    weights_df: pd.DataFrame,
    hide_seasonality: bool = True,
) -> pd.DataFrame:
    """
    Verilen (ortak) ağırlık setiyle bir bölgenin ağırlıklı bütünleşik Z-skoru serisini üretir.
    Yalnızca tüm değişkenlerin gerçekten gözlendiği ortak aylar kullanılır.
    Her değişken, arındırılmış anomalisinin ortalaması ve standart sapmasıyla ölçeklenir; "value_deseasonalized"
    bu Z-indeksidir. Ham görünümde ("value", hide_seasonality=False) aynı ölçeğe mevsimsel döngü eklenir, yani
    iki görünüm aynı birimdedir ve aradaki fark yalnızca takvim ayına bağlıdır.
    """
    merged: Optional[pd.DataFrame] = None
    for _, row in weights_df.iterrows():
        v = row["var_code"]
        sub = param_stl_map[v]["df"][["date", "year", "month", "anomaly", "observed"]].rename(
            columns={"anomaly": f"anom_{v}", "observed": f"obs_{v}"}
        )
        merged = sub if merged is None else pd.merge(merged, sub, on=["date", "year", "month"], how="inner")

    if merged is not None:
        merged = merged.dropna().sort_values("date").reset_index(drop=True)
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["date", "year", "month", "value", "composite_z", "value_deseasonalized", "source_tag"])

    comp_anom = np.zeros(len(merged), dtype=float)
    comp_obs = np.zeros(len(merged), dtype=float)
    for _, row in weights_df.iterrows():
        v = row["var_code"]
        anom = merged[f"anom_{v}"].to_numpy(dtype=float)
        obs = merged[f"obs_{v}"].to_numpy(dtype=float)
        sd = float(np.std(anom))
        if sd > 1e-9:
            w_s = float(row["normalized_weight"]) * float(row["polarity_sign"]) / sd
            comp_anom += w_s * (anom - anom.mean())
            comp_obs += w_s * (obs - obs.mean())
    comp = comp_anom if hide_seasonality else comp_obs

    return pd.DataFrame(
        {
            "date": merged["date"],
            "year": merged["year"],
            "month": merged["month"],
            "value": np.round(comp, 4),
            "composite_z": np.round(comp, 4),
            "value_deseasonalized": np.round(comp_anom, 4),
            "source_tag": L("Shared CRITIC-Weighted Composite Index", "Ortak CRITIC Ağırlıklı Bütünleşik İndeks"),
        }
    )


def compare_parameter_across_regions(
    region_stl: Dict[str, Dict[str, Any]],
    use_col: str = "deseasonalized",
) -> Dict[str, Any]:
    """
    Tek bir parametrenin bölgeler arasındaki ilişkisini (korelasyon, ΔSlope, zıtlık) hesaplar.
    Mevsimsellikten arındırılmış seri kullanılır; böylece ortak yaz-kış döngüsü sahte
    pozitif korelasyon üretmez.
    """
    series = {}
    for name, stl_res in region_stl.items():
        if stl_res.get("valid", False):
            d = stl_res["df"][["date", "year", "month", use_col]].rename(columns={use_col: "value"})
            series[name] = d
    if len(series) < 2:
        return {"valid": False}
    return compare_regions_teleconnection(series, value_col="value")


def stl_decompose_series(df: pd.DataFrame, period: int = 12) -> Dict[str, Any]:
    """
    Aylık zaman serisini takvim ayı klimatolojisiyle `Trend + Seasonal + Residual` bileşenlerine ayırır.
    (Fonksiyon adı geriye dönük uyumluluk için korunmuştur; STL/LOESS artık kullanılmaz, çünkü eksik
    aylarda 12 satırlık periyodu kaydırıyordu.)

    Seri önce eksiksiz aylık takvim eksenine yerleştirilir; seasonal = takvim ayı ortalaması - genel
    ortalama. Eksik aylar doldurulmaz: dönen tabloda yalnızca gerçekten gözlenen aylar vardır.
    En az 2 kez gözlenmemiş takvim ayları arındırılamayacağı için çıkarılır.

    Döndürülen DataFrame sütunları:
    - date, year, month, observed
    - trend (arındırılmış serinin merkezli 13 aylık hareketli ortalaması, yalnızca görselleştirme için)
    - seasonal, resid (= deseasonalized - trend)
    - deseasonalized (observed - seasonal)
    - anomaly (deseasonalized - mean(deseasonalized))
    """
    if df is None or df.empty or "value" not in df.columns:
        return {"valid": False, "df": pd.DataFrame(), "reason": L("No data.", "Veri yok.")}

    clean = df.dropna(subset=["value"]).sort_values("date").copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean = clean.drop_duplicates(subset="date", keep="last")
    month_counts = clean["date"].dt.month.map(clean["date"].dt.month.value_counts())
    clean = clean[month_counts >= 2]
    if len(clean) < max(24, period * 2):
        return {"valid": False, "df": pd.DataFrame(), "reason": L("At least 24 observed months are required.", "En az 24 gözlenmiş ay gerekir.")}

    full_idx = pd.date_range(clean["date"].iloc[0], clean["date"].iloc[-1], freq="MS")
    y_full = clean.set_index("date")["value"].astype(float).reindex(full_idx).to_numpy()
    obs = np.isfinite(y_full)
    month_number = (full_idx.year * 12 + full_idx.month - 1).to_numpy()

    deseas_full = np.full(len(full_idx), np.nan)
    seasonal_full = np.full(len(full_idx), np.nan)
    deseas_full[obs], seasonal_full[obs] = _remove_calendar_climatology(y_full[obs], month_number[obs])
    trend_full = pd.Series(deseas_full).rolling(window=13, center=True, min_periods=7).mean().to_numpy()

    y = y_full[obs]
    seasonal = seasonal_full[obs]
    deseasonalized = deseas_full[obs]
    trend = trend_full[obs]
    resid = deseasonalized - trend
    clim_mean = float(np.mean(deseasonalized))
    anomaly = deseasonalized - clim_mean

    clean = clean.set_index("date").reindex(full_idx[obs])
    out_df = pd.DataFrame(
        {
            "date": full_idx[obs],
            "year": full_idx[obs].year.astype(int),
            "month": full_idx[obs].month.astype(int),
            "observed": np.round(y, 4),
            "trend": np.round(trend, 4),
            "seasonal": np.round(seasonal, 4),
            "resid": np.round(resid, 4),
            "deseasonalized": np.round(deseasonalized, 4),
            "anomaly": np.round(anomaly, 4),
            "source_tag": clean["source_tag"].to_numpy() if "source_tag" in clean.columns else L("Official Archive", "Resmi Arşiv"),
        }
    ).reset_index(drop=True)

    return {
        "valid": True,
        "climatology_mean": round(clim_mean, 4),
        "seasonal_amplitude": round(float(np.nanpercentile(seasonal, 95) - np.nanpercentile(seasonal, 5)), 4),
        "df": out_df,
        "method": L("Deseasonalization using calendar-month climatology (monthly mean)", "Takvim ayı klimatolojisi (aylık ortalama) ile mevsimsellikten arındırma"),
        "n_months_observed": int(obs.sum()),
        "n_months_missing": int((~obs).sum()),
        "n_months_span": int(len(full_idx)),
    }


def hamed_rao_mann_kendall_test(
    dates_or_years: np.ndarray,
    values: np.ndarray,
    alpha: float = 0.05,
    is_monthly: bool = True,
) -> Dict[str, Any]:
    """
    Mann-Kendall trend testi + Theil-Sen medyan eğimi, otokorelasyon ve uzun hafıza düzeltmesiyle.
    (Fonksiyon adı geriye dönük uyumluluk için korunmuştur; uygulanan yöntem "method" anahtarındadır.)

    - dates_or_years: ondalık yıl (yıl + (ay-1)/12). Zaman farkları ve gecikmeler takvim üzerinden
      (ay/yıl) hesaplanır, NaN atıldıktan sonraki satır sırası üzerinden değil.
    - is_monthly=True iken girdide hâlâ mevsimsel döngü varsa (aylara göre ANOVA p < 0.01) test öncesinde
      takvim ayı ortalamaları çıkarılır (deseasonalized_in_test=True). Böylece ham/arındırılmış girdi aynı
      kararı verir; zaten arındırılmış seriye dokunulmaz.
    - Var(S) = Var0(S) (bağ düzeltmeli) x autocorr_factor; autocorr_factor = max(AR(1), uzun hafıza, EWC)
      trend varyans oranları (variance_factors, variance_binding). p ve GA, t dağılımı (t_dof = EWC'nin B'si).
    - Sen eğiminin %95 güven aralığı aynı düzeltilmiş Var(S) ile sıra istatistiklerinden alınır.
    - is_significant = (p < alpha) VE ci_excludes_zero; ayrıştıklarında consistency_note nedenini söyler.
    - kendall_tau: tau-b (bağ düzeltmeli; scipy.stats.kendalltau ile aynı).

    Çıktılar (önceki anahtarlar korunmuştur):
    - S, var_s_raw, var_s_corrected, autocorr_factor, z_score, p_value (float), p_value_display, kendall_tau
    - sen_slope_annual, sen_slope_decade, sen_ci_low_annual, sen_ci_high_annual (+ _decade), sen_intercept
    - trend_line (girdi ile aynı uzunlukta), is_significant, ci_excludes_zero, consistency_note, direction,
      direction_tr, badge_text
    - method, method_short, lag1_autocorr, long_memory_d, variance_factors, variance_binding, t_dof, n_obs,
      n_effective, reliable_verdict, reliability_note
    """
    y_all = np.asarray(values, dtype=float)
    n_all = len(y_all)
    mask = np.isfinite(y_all)
    unit = 12.0 if is_monthly else 1.0

    t_raw = np.asarray(dates_or_years, dtype=float) if dates_or_years is not None else np.array([])
    if t_raw.shape[0] == n_all and np.all(np.isfinite(t_raw[mask])) and len(np.unique(t_raw[mask])) == mask.sum():
        steps_all = np.round(t_raw * unit)  # takvim adımı: ay (aylık) veya yıl
        has_calendar = True
    else:
        steps_all = np.arange(n_all, dtype=float)  # tarih yoksa satır sırası (belgelenmiş geri dönüş)
        has_calendar = False

    y = y_all[mask]
    steps = steps_all[mask].astype(int)
    n = len(y)
    method = L(
        "Mann-Kendall + Theil-Sen slope; Var(S) is inflated by the largest of three dependence estimates from the residuals: "
        "AR(1) effective sample size (Yue & Wang 2004), long memory (local Whittle d -> ARFIMA(0,d,0)) and low-frequency "
        "EWC (Lazarus et al. 2018); p and 95% CI from the same variance, using the t distribution; on the deseasonalized series",
        "Mann-Kendall + Theil-Sen eğimi; Var(S), artıkların üç bağımlılık tahmininin en büyüğüyle büyütülür: "
        "AR(1) etkin örneklem (Yue & Wang 2004), uzun hafıza (yerel Whittle d -> ARFIMA(0,d,0)) ve düşük frekans "
        "EWC (Lazarus vd. 2018); p ve %95 GA aynı varyansla, t dağılımından; mevsimsellikten arındırılmış seri üzerinde"
    )
    method_short = L(
        "MK + Sen, corrected for autocorrelation and long memory (AR(1) / local Whittle / EWC, t distribution)",
        "MK + Sen, otokorelasyon ve uzun hafıza düzeltmeli (AR(1) / yerel Whittle / EWC, t dağılımı)",
    )

    if n < 10:
        note = L(f"Series too short ({n} observations); no reliable trend verdict is possible.",
                 f"Seri çok kısa ({n} gözlem); güvenilir trend kararı verilemez.")
        return {
            "S": 0, "var_s_raw": 0.0, "var_s_corrected": 0.0, "autocorr_factor": 1.0,
            "z_score": 0.0, "p_value": 1.0, "p_value_display": "—", "kendall_tau": 0.0,
            "sen_slope_annual": 0.0, "sen_slope_decade": 0.0,
            "sen_ci_low_annual": 0.0, "sen_ci_high_annual": 0.0,
            "sen_ci_low_decade": 0.0, "sen_ci_high_decade": 0.0, "sen_intercept": 0.0,
            "trend_line": np.zeros(n_all), "is_significant": False, "ci_excludes_zero": False, "consistency_note": "",
            "direction": "stable", "direction_tr": L("No Clear Trend", "Belirgin Eğilim Yok"),
            "badge_text": L(f"⚪ No Verdict — {note}", f"⚪ Karar Verilemedi — {note}"),
            "method": method, "method_short": method_short, "lag1_autocorr": 0.0, "long_memory_d": 0.0,
            "variance_factors": {}, "variance_binding": "", "t_dof": 0,
            "n_obs": int(n), "n_effective": float(n), "reliable_verdict": False, "reliability_note": note,
            "alpha": alpha,
        }

    # Girdide hâlâ mevsimsel döngü varsa (aylara göre tek yönlü ANOVA, p < 0.01) takvim ayı ortalamalarını çıkar.
    # Zaten arındırılmış seriye dokunulmaz (değerler ve bağlar aynen korunur).
    deseasonalized_in_test = False
    if is_monthly and has_calendar and n >= 24:
        groups = [y[steps % 12 == m] for m in np.unique(steps % 12)]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            p_season = stats.f_oneway(*groups).pvalue if len(groups) > 1 else np.nan
        if p_season < 0.01:
            y, _ = _remove_calendar_climatology(y, steps)
            deseasonalized_in_test = True

    t_years = (steps - steps[0]) / unit

    # 1. Mann-Kendall S, Kendall tau ve Theil-Sen eğimi (tüm i < j çiftleri)
    iu = np.triu_indices(n, k=1)
    dy = y[iu[1]] - y[iu[0]]
    dt = t_years[iu[1]] - t_years[iu[0]]
    s_stat = int(np.sum(np.sign(dy)))
    n_pairs = n * (n - 1) / 2.0
    slopes = np.sort(dy / dt)
    sen_slope_annual = float(np.median(slopes)) + 0.0
    sen_intercept = float(np.median(y) - sen_slope_annual * np.median(t_years)) + 0.0

    # 2. Bağ (tie) düzeltmeli Var0(S) ve tau-b (zamanda bağ yoktur)
    _, tie_counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
    var_s_raw = float((n * (n - 1) * (2 * n + 5) - tie_term) / 18.0)
    tied_pairs = float(np.sum(tie_counts * (tie_counts - 1) / 2.0))
    kendall_tau = float(s_stat / np.sqrt(n_pairs * (n_pairs - tied_pairs))) if n_pairs > tied_pairs else 0.0

    # 3. Bağımlılık düzeltmesi: AR(1), uzun hafıza ve EWC trend varyans oranlarının en büyüğü (takvim zamanında)
    resid = y - sen_slope_annual * t_years
    lag1 = _lag1_calendar(resid, steps)
    factor_lm, long_memory_d = _long_memory_factor(resid, steps)
    factor_ewc, t_dof = _ewc_factor(resid, steps)
    variance_factors = {"AR(1)": _trend_variance_factor(lag1, steps), "uzun hafıza": factor_lm, "EWC": factor_ewc}
    binding_key = max(variance_factors, key=variance_factors.get)
    autocorr_factor = variance_factors[binding_key]
    # variance_factors anahtarları sabit kalır (testler); yalnızca gösterilen variance_binding çevrilir.
    variance_binding = L("long memory", "uzun hafıza") if binding_key == "uzun hafıza" else binding_key
    var_s_corrected = var_s_raw * autocorr_factor
    n_effective = n / autocorr_factor

    # 4. Z ve çift yönlü p (süreklilik düzeltmeli), t dağılımı (t_dof serbestlik derecesi)
    if var_s_corrected > 0 and s_stat != 0:
        z_score = (s_stat - np.sign(s_stat)) / np.sqrt(var_s_corrected)
    else:
        z_score = 0.0
    p_value = float(2.0 * stats.t.sf(abs(z_score), t_dof))

    # 5. Sen eğimi %95 GA: aynı düzeltilmiş varyans ve aynı t dağılımıyla sıra istatistikleri
    c_alpha = stats.t.ppf(1.0 - alpha / 2.0, t_dof) * np.sqrt(max(var_s_corrected, 0.0))
    n_sl = len(slopes)
    upper_idx = min(int(np.round((n_sl + c_alpha) / 2.0)), n_sl - 1)
    lower_idx = max(int(np.round((n_sl - c_alpha) / 2.0)) - 1, 0)
    sen_ci_low_annual = float(slopes[lower_idx]) + 0.0
    sen_ci_high_annual = float(slopes[upper_idx]) + 0.0
    ci_excludes_zero = bool(sen_ci_low_annual > 0 or sen_ci_high_annual < 0)

    # Karar ve aralık yapı gereği tutarlı: ikisi de anlamlı diyorsa anlamlı (bağlı değerlerde ayrışabilirler).
    is_significant = bool(p_value < alpha and ci_excludes_zero)
    consistency_note = ""
    if p_value < alpha and not ci_excludes_zero:
        consistency_note = L(
            f"p < {alpha:g} but the 95% confidence interval touches zero (many equal/tied values, e.g. zero precipitation or "
            "zero sea ice); the cautious verdict is 'not significant'.",
            f"p < {alpha:g} ancak %95 güven aralığı sıfıra değiyor (çok sayıda eşit/bağlı değer, ör. sıfır yağış veya "
            "sıfır deniz buzu); karar temkinli olarak 'anlamlı değil'."
        )
    elif ci_excludes_zero and p_value >= alpha:
        consistency_note = L(f"The 95% confidence interval excludes zero but p ≥ {alpha:g} (borderline); verdict: 'not significant'.",
                             f"%95 güven aralığı sıfırı dışlıyor ancak p ≥ {alpha:g} (sınırda); karar 'anlamlı değil'.")

    trend_line = sen_intercept + sen_slope_annual * (steps_all - steps[0]) / unit

    reliable = bool(n >= (MIN_MONTHS_RELIABLE if is_monthly else MIN_YEARS_RELIABLE))
    _min_rel = MIN_MONTHS_RELIABLE if is_monthly else MIN_YEARS_RELIABLE
    reliability_note = (
        "" if reliable else L(
            f"Series is short ({n} observations < {_min_rel}); in short autocorrelated "
            "series the false-alarm rate rises above 5%, so the trend verdict is not reliable; "
            "choose a period of at least 10 years.",
            f"Seri kısa ({n} gözlem < {_min_rel}); otokorelasyonlu "
            "kısa serilerde yanlış alarm oranı %5'in üstüne çıktığından trend kararı güvenilir değildir, "
            "en az 10 yıllık bir dönem seçin."
        )
    )

    if sen_slope_annual > 0:
        direction, direction_tr = "increasing", L("Increasing Trend (Positive Slope)", "Artış Eğilimi (Pozitif Eğim)")
    elif sen_slope_annual < 0:
        direction, direction_tr = "decreasing", L("Decreasing Trend (Negative Slope)", "Azalış Eğilimi (Negatif Eğim)")
    else:
        direction, direction_tr = "stable", L("Stable (Zero Slope)", "Durağan (Sıfır Eğim)")
    if direction != "stable" and not is_significant:
        direction_tr += L(" — not yet statistically significant", " — henüz istatistiksel olarak anlamlı değil")

    if is_significant:
        badge_text = L(f"🟢 Statistically Significant Trend (p < {alpha:g})", f"🟢 İstatistiksel Olarak Anlamlı Trend (p < {alpha:g})")
    else:
        badge_text = L("🟡 Not Yet Significant (Noise Dominates / Longer Period Needed)",
                       "🟡 Henüz Anlamlı Değil (Gürültü Baskın / Daha Uzun Süre Gerekli)")
    if not reliable:
        badge_text += L(" · ⚠️ Series too short for a reliable verdict (< 10 years)", " · ⚠️ Seri güvenilir karar için çok kısa (< 10 yıl)")

    return {
        "S": s_stat,
        "var_s_raw": round(var_s_raw, 2),
        "var_s_corrected": round(var_s_corrected, 2),
        "autocorr_factor": round(autocorr_factor, 3),
        "z_score": round(float(z_score), 4),
        "p_value": p_value,
        "p_value_display": _format_p(p_value),
        "kendall_tau": round(kendall_tau, 4),
        "sen_slope_annual": sen_slope_annual,
        "sen_slope_decade": sen_slope_annual * 10.0,
        "sen_ci_low_annual": sen_ci_low_annual,
        "sen_ci_high_annual": sen_ci_high_annual,
        "sen_ci_low_decade": sen_ci_low_annual * 10.0,
        "sen_ci_high_decade": sen_ci_high_annual * 10.0,
        "sen_intercept": round(sen_intercept, 4),
        "trend_line": np.round(trend_line, 4),
        "is_significant": is_significant,
        "ci_excludes_zero": ci_excludes_zero,
        "consistency_note": consistency_note,
        "direction": direction,
        "direction_tr": direction_tr,
        "badge_text": badge_text,
        "method": method,
        "method_short": method_short,
        "lag1_autocorr": round(lag1, 3),
        "long_memory_d": round(long_memory_d, 3),
        "variance_factors": {k: round(v, 3) for k, v in variance_factors.items()},
        "variance_binding": variance_binding,
        "t_dof": int(t_dof),
        "n_obs": int(n),
        "n_effective": round(float(n_effective), 1),
        "reliable_verdict": reliable,
        "reliability_note": reliability_note,
        "deseasonalized_in_test": deseasonalized_in_test,
        "alpha": alpha,
    }


def _vif_status_text(vif_val: float) -> str:
    if vif_val > 5.0:
        return L(f"⚠️ VIF = {vif_val:.2f} (> 5: high multicollinearity — diagnostic only, not applied to the weight)",
                 f"⚠️ VIF = {vif_val:.2f} (> 5: yüksek çoklu doğrusallık — yalnızca tanı, ağırlığa uygulanmaz)")
    return L(f"🟢 VIF = {vif_val:.2f} (<= 5.0 OK)", f"🟢 VIF = {vif_val:.2f} (<= 5.0 Uygun)")


def _composite_sign(meta: Dict[str, Any]) -> float:
    """Değişkenin bileşik indeksteki sabit işareti = meta veri "polarity" alanının işareti (data_loader yön kuralı)."""
    return -1.0 if float(meta.get("polarity", 1.0)) < 0 else 1.0


def compute_objective_weights_and_composite(
    param_stl_map: Dict[str, Dict[str, Any]],
    selected_vars: List[str],
    official_variables_meta: Dict[str, Dict[str, Any]],
    hide_seasonality: bool = True,
) -> Dict[str, Any]:
    """
    Seçilen 1 ila 4 parametre için ağırlıklı bütünleşik Z-skoru indeksi.
    1. İşaretler sabittir ve meta "polarity" alanından gelir (seçim sırasına bağlı değildir); indeksin
       yükselmesinin ne anlama geldiği "index_description" içinde düz metinle döner.
    2. Ağırlıklar CRITIC: C_j = sigma_j * sum_k (1 - |r_jk|) (min-max normalize edilmiş, mevsimsellikten
       arındırılmış seriler üzerinde), w_j = C_j / sum C. Ekip önem puanı ağırlığa katılmaz.
    3. VIF_j = diag(R^-1)_j yalnızca tanı amaçlı raporlanır (VIF > 5 uyarısı), ağırlığı değiştirmez.
    4. Yalnızca tüm değişkenlerin gerçekten gözlendiği ortak aylar kullanılır (n_months_used).
    composite_df: "value" görüntüleme seçimine göre (arındırılmış / mevsimsel döngü eklenmiş, aynı Z ölçeği),
    "value_deseasonalized" her zaman arındırılmış Z-indeksidir; trend testi bu sütun üzerinde yapılmalıdır
    (hamed_rao_mann_kendall_test ham "value" verildiğinde de mevsimsel döngüyü kendisi çıkarır).
    """
    valid_vars = [
        v for v in selected_vars
        if v in param_stl_map and v in official_variables_meta and param_stl_map[v].get("valid", False) and not param_stl_map[v]["df"].empty
    ]
    empty_result = {
        "valid": False,
        "reason": L("No valid variables for the composite index.", "Bütünleşik indeks için geçerli değişken yok."),
        "weights_df": pd.DataFrame(),
        "composite_df": pd.DataFrame(),
        "vif_warnings": [],
        "n_months_used": 0,
        "index_description": "",
        "signs": {},
        "weight_method": "",
    }
    if not valid_vars:
        return empty_result

    # Ortak ay kuralı: tüm değişkenlerin gözlendiği aylar, en az birinin gözlendiği ayların >= %70'i olmalı.
    # Aksi halde bileşik indeks sansürlü bir alt kümeden (ör. yalnızca buzsuz aylar) kurulur ve gösterilmez.
    months = {v: set(pd.to_datetime(param_stl_map[v]["df"]["date"])) for v in valid_vars}
    union = set().union(*months.values())
    common = set.intersection(*months.values())
    if len(valid_vars) > 1 and len(common) < MIN_COMPOSITE_OVERLAP * len(union):
        share = {v: len(months[v]) / len(union) for v in valid_vars}
        limiting = sorted((v for v in valid_vars if share[v] < 1.0), key=share.get)
        limiting = [v for v in limiting if share[v] < MIN_COMPOSITE_OVERLAP] or limiting[:2]
        names = ", ".join(L(f"{official_variables_meta[v]['short_label']} (share of months with data {share[v] * 100:.0f}%)",
                            f"{official_variables_meta[v]['short_label']} (veri olan ay oranı %{share[v] * 100:.0f})") for v in limiting)
        late_start = max((int(official_variables_meta[v].get("satellite_start_year", 0)) for v in limiting), default=0)
        if late_start > min(union).year:
            names += L(f"; this variable (these variables) starts in {late_start}; starting the analysis period in {late_start} raises the shared-month share",
                       f"; bu değişken(ler) {late_start}'de başlıyor, analiz aralığını {late_start}'den başlatmak ortak ay oranını yükseltir")
        return {
            **empty_result,
            "reason": L(
                f"Composite index not built: the months in which all selected variables are observed make up only "
                f"{len(common) / len(union) * 100:.0f}% of the months in which at least one is observed ({len(common)}/{len(union)} months; "
                f"threshold {MIN_COMPOSITE_OVERLAP * 100:.0f}%). Limiting the shared months: {names}. Such an index would be built only "
                f"from selected (censored) months. Per-variable results are shown separately below.",
                f"Bütünleşik indeks oluşturulmadı: seçili değişkenlerin hepsinin gözlendiği ortak ayların, en az birinin "
                f"gözlendiği aylara oranı yalnızca %{len(common) / len(union) * 100:.0f} ({len(common)}/{len(union)} ay; "
                f"eşik %{MIN_COMPOSITE_OVERLAP * 100:.0f}). Ortak ayları kısıtlayan: {names}. Böyle bir indeks yalnızca "
                f"seçilmiş (sansürlü) aylardan kurulurdu. Değişken bazlı sonuçlar aşağıda ayrı ayrı gösterilir."
            ),
        }

    signs = {v: _composite_sign(official_variables_meta[v]) for v in valid_vars}
    # CRITIC/VIF için ortak gerçek aylar üzerinde arındırılmış anomalileri birleştir.
    merged: Optional[pd.DataFrame] = None
    for v in valid_vars:
        sub = param_stl_map[v]["df"][["date", "year", "month", "anomaly"]].rename(columns={"anomaly": v})
        merged = sub if merged is None else pd.merge(merged, sub, on=["date", "year", "month"], how="inner")
    merged = merged.dropna().sort_values("date").reset_index(drop=True)
    k = len(valid_vars)
    if len(merged) < 24:
        return {**empty_result, "reason": L(f"Composite index not built: number of shared observed months {len(merged)} < 24.",
                                             f"Bütünleşik indeks oluşturulmadı: ortak gözlenen ay sayısı {len(merged)} < 24.")}

    anom_matrix = merged[valid_vars].to_numpy(dtype=float)
    if k == 1:
        critic_raw = np.ones(1)
        vif_values = np.ones(1)
    else:
        rng_ = anom_matrix.max(axis=0) - anom_matrix.min(axis=0)
        norm_matrix = np.where(rng_ > 1e-12, (anom_matrix - anom_matrix.min(axis=0)) / np.where(rng_ > 1e-12, rng_, 1.0), 0.5)
        corr_matrix = np.nan_to_num(np.corrcoef(norm_matrix, rowvar=False), nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)
        try:
            vif_values = np.diag(np.linalg.inv(corr_matrix + np.eye(k) * 1e-6))
        except np.linalg.LinAlgError:
            vif_values = np.full(k, np.inf)
        critic_raw = np.std(norm_matrix, axis=0, ddof=1) * np.sum(1.0 - np.abs(corr_matrix), axis=1)

    total_c = float(np.sum(critic_raw))
    norm_weights = critic_raw / total_c if total_c > 1e-12 else np.ones(k) / k

    vif_warnings: List[str] = []
    weights_records = []
    for j, v in enumerate(valid_vars):
        meta = official_variables_meta[v]
        vif_val = float(vif_values[j])
        if vif_val > 5.0:
            vif_warnings.append(L(
                f"**{meta['short_label']}** shows high multicollinearity (**VIF = {vif_val:.2f} > 5.0**): "
                f"it carries largely the same signal as the other selected variables. This is only a diagnostic; "
                f"the CRITIC weight already accounts for correlation.",
                f"**{meta['short_label']}** değişkeninde yüksek çoklu doğrusallık (**VIF = {vif_val:.2f} > 5.0**): "
                f"diğer seçili değişkenlerle büyük ölçüde aynı sinyali taşıyor. Bu yalnızca bir tanıdır; "
                f"CRITIC ağırlığı korelasyonu zaten hesaba katar."
            ))
        weights_records.append(
            {
                "var_code": v,
                "rank": meta["rank"],
                "label": meta["short_label"],
                # Geriye dönük uyumluluk: ekibin kendi önem puanı (/10 ölçeği). Ağırlıkta KULLANILMAZ.
                "ipcc_base_weight": round(float(meta.get("base_weight", np.nan)), 3),
                "team_importance_score": round(float(meta.get("base_weight", np.nan)), 3),
                "critic_score": round(float(critic_raw[j]), 3),
                "vif_value": round(vif_val, 2),
                "vif_status": _vif_status_text(vif_val),
                "raw_weight": float(critic_raw[j]),
                "normalized_weight": round(float(norm_weights[j]), 4),
                "weight_pct": round(float(norm_weights[j]) * 100.0, 2),
                "polarity_sign": signs[v],
            }
        )
    weights_df = pd.DataFrame(weights_records)

    comp_df = apply_weights_to_composite(param_stl_map, weights_df, hide_seasonality)
    comp_df["source_tag"] = (L("CRITIC-Weighted Composite Index", "CRITIC Ağırlıklı Bütünleşik İndeks") if k > 1
                             else L("Single-Variable Z-Score", "Tek Değişken Z-Skoru"))

    up = [official_variables_meta[v]["short_label"] for v in valid_vars if signs[v] > 0]
    down = [official_variables_meta[v]["short_label"] for v in valid_vars if signs[v] < 0]
    parts = []
    if up:
        parts.append(L("higher ", "yüksek ") + ", ".join(up))
    if down:
        parts.append(L("lower ", "düşük ") + ", ".join(down))
    index_description = L(
        "A rising index means: " + " and ".join(parts) + ". "
        "(Unit: weighted Z-score; each variable is scaled by the standard deviation of its own monthly anomaly.)",
        "İndeksin yükselmesi şu anlama gelir: " + " ve ".join(parts) + ". "
        "(Birim: ağırlıklı Z-skoru, her değişken kendi aylık anomalisinin standart sapmasıyla ölçeklenir.)"
    )

    return {
        "valid": True,
        "reason": "",
        "weights_df": weights_df,
        "composite_df": comp_df,
        "vif_warnings": vif_warnings,
        "n_months_used": int(len(comp_df)),
        "index_description": index_description,
        "signs": signs,
        "weight_method": (L("CRITIC (sigma x sum(1 - |r|), normalized); VIF diagnostic only", "CRITIC (sigma x toplam(1 - |r|), normalize); VIF yalnızca tanı")
                          if k > 1 else L("Single variable (weight = 1)", "Tek değişken (ağırlık = 1)")),
    }


_FC_COLS = ["date", "forecast", "lower_95", "upper_95", "model_name"]


def _empty_forecast(reason: str, backtest: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    out = pd.DataFrame(columns=_FC_COLS)
    out.attrs.update({"valid": False, "reason": reason, "model_name": "", "fallback_used": False,
                      "backtest": backtest or {}, "has_skill": False, "clipped": False})
    return out


def _fourier(month_number: np.ndarray) -> np.ndarray:
    a = 2.0 * np.pi * (np.asarray(month_number) % 12) / 12.0
    return np.column_stack([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)])


def _fit_trend_ar1(y: np.ndarray, month_number: np.ndarray, seasonal: bool, steps: int):
    """
    Doğrusal trend + durağan/tersinir AR(1) (+ 2 harmonik). Eksik aylar NaN olarak Kalman filtresinde atlanır.
    Optimizasyonun ölçek sorunu yaşamaması için seri standartlaştırılır ve zaman on yıl biriminde verilir.
    """
    fut = month_number[-1] + 1 + np.arange(steps)
    mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
    if not sd > 0:
        return None

    def _exog(mn: np.ndarray) -> np.ndarray:
        t = ((mn - month_number[0]) / 120.0)[:, None]
        return np.hstack([t, _fourier(mn)]) if seasonal else t

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(
                (y - mu) / sd, exog=_exog(month_number), order=(1, 0, 0), trend="c",
                enforce_stationarity=True, enforce_invertibility=True,
            ).fit(disp=False, maxiter=200)
            if not res.mle_retvals.get("converged", False):
                return None
            pred = res.get_forecast(steps=steps, exog=_exog(fut))
            ci = np.asarray(pred.conf_int(alpha=0.05), dtype=float) * sd + mu
            return np.asarray(pred.predicted_mean, dtype=float) * sd + mu, ci[:, 0], ci[:, 1]
    except Exception:
        return None


def _fit_recent_climatology(y: np.ndarray, month_number: np.ndarray, seasonal: bool, steps: int, years: int = 5):
    """
    Son `years` yılın ortalaması (seasonal=True ise takvim ayı başına) ve bağımsız normal hata varsayımıyla
    %95 tahmin aralığı: mean ± t * s * sqrt(1 + 1/k). Hem yedek model hem de beceri karşılaştırma tabanıdır.
    """
    win = slice(max(0, len(y) - 12 * years), len(y))
    yr, mr = y[win], np.asarray(month_number)[win] % 12
    ok = np.isfinite(yr)
    yr, mr = yr[ok], mr[ok]
    fut_m = (month_number[-1] + 1 + np.arange(steps)) % 12
    groups = mr if seasonal else np.zeros_like(mr)
    fut_g = fut_m if seasonal else np.zeros_like(fut_m)
    means, counts = np.full(12, np.nan), np.zeros(12)
    for g in np.unique(groups):
        means[g] = yr[groups == g].mean()
        counts[g] = (groups == g).sum()
    dof = len(yr) - len(np.unique(groups))
    if dof < 3:
        return None
    s = float(np.sqrt(np.sum((yr - means[groups]) ** 2) / dof))
    mean_f = means[fut_g]
    with np.errstate(divide="ignore", invalid="ignore"):
        half = stats.t.ppf(0.975, dof) * s * np.sqrt(1.0 + 1.0 / counts[fut_g])
    return mean_f, mean_f - half, mean_f + half


def _within_envelope(hist: np.ndarray, mean: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Tahmin geçmiş aralığın ±%50'si, bant ±%150'si içinde ve sonlu olmalı (ıraksama denetimi)."""
    h = hist[np.isfinite(hist)]
    a, b = float(h.min()), float(h.max())
    rng_ = max(b - a, 1e-9)
    fin = np.isfinite(mean)
    if not fin.any() or not (np.all(np.isfinite(lo[fin])) and np.all(np.isfinite(hi[fin]))):
        return False
    return bool(
        np.all(mean[fin] >= a - 0.5 * rng_) and np.all(mean[fin] <= b + 0.5 * rng_)
        and np.all(lo[fin] >= a - 1.5 * rng_) and np.all(hi[fin] <= b + 1.5 * rng_)
    )


def generate_5year_sarima_forecast(
    df: pd.DataFrame,
    value_col: str = "value",
    hide_seasonality: bool = True,
    steps: int = 60,
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> pd.DataFrame:
    """
    Serinin kendi son gözlem tarihinden sonraki `steps` ay (varsayılan 60) için tek değişkenli istatistiksel
    dışa değerleme (fiziksel iklim projeksiyonu değildir).

    bounds = (alt, üst) fiziksel sınırlar (gözlem biriminde; None = sınırsız). Tahmin ve %95 bandı bu sınırlarda
    kırpılır; value_col "deseasonalized" ise sınırlar o takvim ayının mevsimsel bileşeni kadar kaydırılır
    (arındırılmış = gözlem - mevsimsel). Kırpma olduysa attrs["clipped"] = True.

    Ana model: doğrusal trend + durağan AR(1) (SARIMAX; hide_seasonality=False ise + 2 harmonik mevsimsellik),
    eksik aylar takvim ekseninde NaN olarak modele verilir. Model yakınsamazsa veya tahmin/bant geçmiş veri
    zarfının dışına çıkarsa "son 5 yılın aylık ortalaması" modeline düşülür; o da geçersizse boş tablo döner.
    Geçmişte hiç gözlenmemiş takvim ayları için tahmin NaN bırakılır.

    Dönen DataFrame sütunları: date, forecast, lower_95, upper_95, model_name (gerçekte kullanılan model).
    DataFrame.attrs: valid, reason, model_name, fallback_used, backtest (son 60 ay geriye dönük sınama:
    mae_model, mae_baseline, skill_score, coverage_95, train_end, test_period), has_skill, skill_note.
    """
    if df is None or df.empty or value_col not in df.columns:
        return _empty_forecast(L("No data.", "Veri yok."))

    clean = df[["date", value_col]].dropna().copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean = clean.sort_values("date").drop_duplicates(subset="date", keep="last")
    if len(clean) < 36:
        return _empty_forecast(L("At least 36 observed months are required for a projection.", "Projeksiyon için en az 36 gözlenmiş ay gerekir."))

    idx = pd.date_range(clean["date"].iloc[0], clean["date"].iloc[-1], freq="MS")
    y = clean.set_index("date")[value_col].astype(float).reindex(idx).to_numpy()
    month_number = (idx.year * 12 + idx.month - 1).to_numpy()
    seasonal = not hide_seasonality
    main_name = L("Linear Trend + AR(1)", "Doğrusal Trend + AR(1)") + (L(" + Seasonal Harmonics", " + Mevsimsel Harmonikler") if seasonal else "") + " (SARIMAX)"
    fb_name = (L("Mean of the Last 5 Years", "Son 5 Yılın Ortalaması") + (L(" (per calendar month)", " (takvim ayı başına)") if seasonal else "")
               + L(" — fallback model", " — yedek model"))

    def _predict(y_: np.ndarray, mn_: np.ndarray):
        out = _fit_trend_ar1(y_, mn_, seasonal, steps)
        if out is not None and _within_envelope(y_, *out):
            return out, main_name, False
        out = _fit_recent_climatology(y_, mn_, seasonal, steps)
        if out is not None and _within_envelope(y_, *out):
            return out, fb_name, True
        return None, "", True

    # Geriye dönük sınama: son 60 ay dışarıda bırakılır (veri 2025'te bitiyorsa: eğitim ≤2020, test 2021–2025).
    backtest: Dict[str, Any] = {}
    if len(y) >= 60 + 120:
        tr, te = y[:-60], y[-60:]
        ok = np.isfinite(te)
        pred_m, name_m, _ = _predict(tr, month_number[:-60])
        base = _fit_recent_climatology(tr, month_number[:-60], seasonal, 60)
        if pred_m is not None and base is not None and ok.any():
            mae_m = float(np.nanmean(np.abs(pred_m[0][ok] - te[ok])))
            mae_b = float(np.nanmean(np.abs(base[0][ok] - te[ok])))
            backtest = {
                "train_end": str(idx[-61].date()),
                "test_period": f"{idx[-60].year}-{idx[-60].month:02d}–{idx[-1].year}-{idx[-1].month:02d}",
                "model_name": name_m,
                "mae_model": round(mae_m, 4),
                "mae_baseline": round(mae_b, 4),
                "skill_score": round(1.0 - mae_m / mae_b, 3) if mae_b > 0 else 0.0,
                "coverage_95": round(float(np.mean((te[ok] >= pred_m[1][ok]) & (te[ok] <= pred_m[2][ok]))), 3),
                "baseline_name": L("Mean of the last 5 years", "Son 5 yılın ortalaması"),
            }

    pred, model_name, fallback_used = _predict(y, month_number)
    if pred is None:
        return _empty_forecast(
            L("Neither the model nor the fallback model produced a projection consistent with the historical data range; no projection is shown.",
              "Model ve yedek model geçmiş veri aralığıyla tutarlı bir projeksiyon üretemedi; projeksiyon gösterilmiyor."),
            backtest,
        )
    mean_fc, lower_95, upper_95 = (np.array(a, dtype=float) for a in pred)

    future_dates = pd.date_range(idx[-1] + pd.DateOffset(months=1), periods=steps, freq="MS")
    seen_months = set(idx.month[np.isfinite(y)])
    unseen = ~np.isin(future_dates.month, list(seen_months))
    mean_fc[unseen] = lower_95[unseen] = upper_95[unseen] = np.nan

    clipped = False
    if bounds is not None and any(b is not None for b in bounds):
        shift = np.zeros(steps)
        if value_col == "deseasonalized" and "seasonal" in df.columns:
            seas = pd.Series(df["seasonal"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]).dt.month).groupby(level=0).mean()
            shift = seas.reindex(future_dates.month).fillna(0.0).to_numpy()
        lo_b = -np.inf if bounds[0] is None else bounds[0] - shift
        hi_b = np.inf if bounds[1] is None else bounds[1] - shift
        before = np.concatenate([mean_fc, lower_95, upper_95])
        mean_fc, lower_95, upper_95 = (np.clip(a, lo_b, hi_b) for a in (mean_fc, lower_95, upper_95))
        after = np.concatenate([mean_fc, lower_95, upper_95])
        fin = np.isfinite(before)
        clipped = bool(np.any(np.abs(before[fin] - after[fin]) > 1e-9))

    # Tek bir 60 aylık pencere gürültülü olduğundan, MAE'de en az %5 iyileşme "beceri" sayılır.
    has_skill = bool(backtest and backtest["skill_score"] >= 0.05)
    if not backtest:
        skill_note = L("The series is not long enough for a backtest (at least 15 years required); skill is unknown.",
                       "Geriye dönük sınama için seri yeterince uzun değil (en az 15 yıl gerekir); beceri bilinmiyor.")
    elif has_skill:
        skill_note = L(
            f"In the backtest ({backtest['test_period']}) the model beats the mean of the last 5 years "
            f"(skill score {backtest['skill_score']:+.2f}).",
            f"Geriye dönük sınamada ({backtest['test_period']}) model, son 5 yıl ortalamasından daha iyi "
            f"(beceri skoru {backtest['skill_score']:+.2f})."
        )
    else:
        skill_note = L(
            f"In the backtest ({backtest['test_period']}) the model is not clearly better than the mean of the last 5 years "
            f"(skill score {backtest['skill_score']:+.2f}); the projection is only an illustrative extrapolation.",
            f"Geriye dönük sınamada ({backtest['test_period']}) model, son 5 yıl ortalamasından belirgin biçimde iyi değil "
            f"(beceri skoru {backtest['skill_score']:+.2f}); projeksiyon yalnızca açıklayıcı bir dışa değerlemedir."
        )

    out = pd.DataFrame(
        {
            "date": future_dates,
            "forecast": np.round(mean_fc, 4),
            "lower_95": np.round(lower_95, 4),
            "upper_95": np.round(upper_95, 4),
            "model_name": model_name,
        }
    )
    out.attrs.update({"valid": True, "reason": "", "model_name": model_name, "fallback_used": fallback_used,
                      "backtest": backtest, "has_skill": has_skill, "skill_note": skill_note, "clipped": clipped})
    return out


def _detrended_anomalies(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Eksiksiz aylık takvim ekseninde: takvim ayı ortalamaları ve doğrusal (OLS) trend çıkarılmış seri (eksikler NaN)."""
    d = df[["date", value_col]].dropna().copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates(subset="date", keep="last")
    if len(d) < 24:
        return pd.Series(dtype=float)
    idx = pd.date_range(d["date"].iloc[0], d["date"].iloc[-1], freq="MS")
    y = d.set_index("date")[value_col].astype(float).reindex(idx).to_numpy()
    mn = (idx.year * 12 + idx.month - 1).to_numpy()
    ok = np.isfinite(y)
    ya, _ = _remove_calendar_climatology(y[ok], mn[ok])
    coef = np.polyfit(mn[ok].astype(float), ya, 1)
    out = np.full(len(idx), np.nan)
    out[ok] = ya - np.polyval(coef, mn[ok].astype(float))
    return pd.Series(out, index=idx)


def _pair_relationship(sa: pd.Series, sb: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    İki arındırılmış + trendi çıkarılmış seri arasındaki eşzamanlı (gecikme 0) ilişki.
    p-değeri: Bretherton vd. (1999) n_eff = n (1 - r_a r_b) / (1 + r_a r_b), t-dağılımı (n_eff - 2 sd).
    Gecikme kuralı: lag k > 0 -> corr(A(t), B(t + k)), yani "A, B'den k ay önde". En iyi gecikme yalnızca
    betimseldir; çoklu gecikme taraması yanlış-pozitif üreteceği için karara katılmaz.
    """
    j = pd.concat([sa.rename("a"), sb.rename("b")], axis=1).asfreq("MS")
    both = j["a"].notna() & j["b"].notna()
    n = int(both.sum())
    out: Dict[str, Any] = {"n_common": n, "rho": 0.0, "p": 1.0, "pearson": 0.0, "n_eff": float(n),
                           "best_lag": 0, "best_xcorr": 0.0, "relationship": "none", "valid": n >= MIN_OVERLAP_MONTHS}
    if not out["valid"]:
        return out
    a, b = j["a"][both].to_numpy(), j["b"][both].to_numpy()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return out
    if np.allclose(a, b, rtol=0.0, atol=1e-9):
        # Özdeş seriler: iki nokta aynı NASA POWER hücresinde. rho = 1 bir ilişki değil, aynı veridir.
        out.update({"rho": 1.0, "pearson": 1.0, "p": float("nan"), "relationship": "identical", "valid": False})
        return out

    def _r1(s: pd.Series) -> float:
        s = s.dropna()
        return _lag1_calendar(s.to_numpy(), (s.index.year * 12 + s.index.month).to_numpy())

    ra, rb = _r1(sa), _r1(sb)
    n_eff = float(np.clip(n * (1.0 - ra * rb) / (1.0 + ra * rb), 3.0, n))
    rho = float(stats.spearmanr(a, b).statistic)
    pearson = float(np.corrcoef(a, b)[0, 1])
    t_stat = rho * np.sqrt((n_eff - 2.0) / max(1e-12, 1.0 - rho ** 2))
    p = float(2.0 * stats.t.sf(abs(t_stat), n_eff - 2.0))

    best_lag, best_xc = 0, pearson
    for lag in range(-6, 7):
        sh = j["b"].shift(-lag)  # B(t + lag) A(t) ile hizalanır
        m = j["a"].notna() & sh.notna()
        if lag != 0 and m.sum() >= MIN_OVERLAP_MONTHS:
            xc = float(np.corrcoef(j["a"][m], sh[m])[0, 1])
            if abs(xc) > abs(best_xc):
                best_lag, best_xc = lag, xc

    relationship = "none"
    if p < alpha:
        relationship = "opposite" if rho < 0 else "same"
    out.update({"rho": rho, "p": p, "pearson": pearson, "n_eff": n_eff, "best_lag": best_lag,
                "best_xcorr": best_xc, "relationship": relationship})
    return out


_REL_LABEL = LocalizedMapping({
    "opposite": {"en": "↔️ Opposite Relationship (significant negative correlation)",
                 "tr": "↔️ Ters Yönlü İlişki (anlamlı negatif korelasyon)"},
    "same": {"en": "🔗 Same-Direction Relationship (significant positive correlation)",
             "tr": "🔗 Eş Yönlü İlişki (anlamlı pozitif korelasyon)"},
    "none": {"en": "⚖️ No Significant Relationship Detected", "tr": "⚖️ Anlamlı İlişki Saptanmadı"},
    "identical": {"en": "⚪ Same NASA POWER Cell — Series Identical, No Relationship Verdict",
                  "tr": "⚪ Aynı NASA POWER Hücresi — Seriler Özdeş, İlişki Kararı Verilmedi"},
})


def _lag_sentence(name_a: str, name_b: str, lag: int) -> str:
    if lag > 0:
        return L(f"{name_a} leads {name_b} by {lag} months (A leads B by {lag} months)",
                 f"{name_a}, {name_b}'den {lag} ay önde (A, B'ye {lag} ay öncülük ediyor)")
    if lag < 0:
        return L(f"{name_b} leads {name_a} by {-lag} months (B leads A by {-lag} months)",
                 f"{name_b}, {name_a}'den {-lag} ay önde (B, A'ya {-lag} ay öncülük ediyor)")
    return L("the strongest link is simultaneous (0-month lag)", "en güçlü bağ eşzamanlı (0 ay gecikme)")


def compare_regions_teleconnection(
    region_series: Dict[str, pd.DataFrame],
    value_col: str = "value",
) -> Dict[str, Any]:
    """
    Çiftli (Dual) ve Çoklu (Multi - 10 bölgeye kadar) modlarda bölgeler arası ilişki.
    Korelasyonlar, takvim ayı ortalamaları ve doğrusal trendi çıkarılmış seriler üzerinde, yalnızca ortak
    gözlenen aylarda (en az MIN_OVERLAP_MONTHS) hesaplanır; p-değerleri otokorelasyona göre etkin örneklem
    büyüklüğüyle düzeltilir. Karar yalnızca anlamlı korelasyona dayanır (eğim işaretleri karara katılmaz).
    Gecikme kuralı: best_lag_months = k > 0 ise A, B'den k ay öndedir (corr(A(t), B(t+k))).
    Ana özet (spearman_*, best_*, verdict) ilk iki bölge (A = 1., B = 2.) içindir; pairwise_df tüm çiftleri verir.
    """
    region_names = list(region_series.keys())
    if len(region_names) < 2:
        return {"valid": False, "reason": L("At least two regions are required.", "En az iki bölge gerekir.")}

    region_stats: Dict[str, Dict[str, Any]] = {}
    resid: Dict[str, pd.Series] = {}
    for r_name in region_names:
        df_r = region_series[r_name].dropna(subset=[value_col]).sort_values("date")
        region_stats[r_name] = hamed_rao_mann_kendall_test(
            decimal_years(df_r), df_r[value_col].to_numpy(dtype=float), is_monthly=True
        )
        resid[r_name] = _detrended_anomalies(df_r, value_col)

    n_reg = len(region_names)
    corr = np.eye(n_reg)
    pairwise_records = []
    pair_res: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i in range(n_reg):
        for j in range(i + 1, n_reg):
            ra, rb = region_names[i], region_names[j]
            pr = _pair_relationship(resid[ra], resid[rb])
            pair_res[(i, j)] = pr
            corr[i, j] = corr[j, i] = pr["pearson"]
            sa = float(region_stats[ra]["sen_slope_annual"])
            sb = float(region_stats[rb]["sen_slope_annual"])
            pairwise_records.append(
                {
                    L("Region 1", "Bölge 1"): ra,
                    L("Region 2", "Bölge 2"): rb,
                    L("Slope 1 (per yr)", "Eğim 1 (Yıllık)"): round(sa, 4),
                    L("Slope 2 (per yr)", "Eğim 2 (Yıllık)"): round(sb, 4),
                    L("ΔSlope (Slope Difference)", "ΔSlope (Eğim Farkı)"): round(sa - sb, 4),
                    L("Cross-Correlation (r)", "Çapraz Korelasyon (r)"): round(pr["pearson"], 3),
                    "Spearman ρ": round(pr["rho"], 3),
                    L("p (effective n)", "p (etkin n)"): _format_p(pr["p"]) if pr["valid"] else "—",
                    L("Shared Months", "Ortak Ay"): pr["n_common"],
                    L("Response Class", "Tepki Sınıfı"): (
                        _REL_LABEL[pr["relationship"]] if pr["valid"] or pr["relationship"] == "identical"
                        else L("⚪ Insufficient shared data", "⚪ Yetersiz ortak veri")
                    ),
                }
            )
    corr_df = pd.DataFrame(np.round(corr, 3), index=region_names, columns=region_names)

    name_a, name_b = region_names[0], region_names[1]
    pr = pair_res[(0, 1)]
    slope_a = float(region_stats[name_a]["sen_slope_annual"])
    slope_b = float(region_stats[name_b]["sen_slope_annual"])
    delta_slope = slope_a - slope_b
    lag_convention = L("best_lag_months = k > 0: A (region 1) leads B (region 2) by k months; k < 0: B leads A by |k| months.",
                       "best_lag_months = k > 0: A (1. bölge), B'den (2. bölge) k ay önde; k < 0: B, A'dan |k| ay önde.")

    if pr["relationship"] == "identical":
        verdict_badge = _REL_LABEL["identical"]
        explanation = L(
            f"For **{name_a}** and **{name_b}** the series are exactly the same in all {pr['n_common']} shared months: both points fall in the same "
            f"NASA POWER cell (POWER gives cell values, not point values). Here ρ = 1 is not a relationship but "
            f"the same data; no relationship verdict was given. For a comparison, pick points in different cells (MERRA-2 0.5°×0.625°; "
            f"CERES 1° for radiation variables).",
            f"**{name_a}** ve **{name_b}** için seriler {pr['n_common']} ortak ayın hepsinde birebir aynı: iki nokta aynı "
            f"NASA POWER hücresine düşüyor (POWER nokta değeri değil, hücre değeri verir). ρ = 1 burada bir ilişki değil, "
            f"aynı veridir; ilişki kararı verilmedi. Karşılaştırma için farklı hücrelerde (MERRA-2 0.5°×0.625°, ışınım "
            f"değişkenlerinde CERES 1°) noktalar seçin."
        )
    elif not pr["valid"]:
        verdict_badge = L("⚪ Insufficient Shared Data — Relationship Could Not Be Assessed",
                          "⚪ Yetersiz Ortak Veri — İlişki Değerlendirilemedi")
        explanation = L(
            f"**{name_a}** and **{name_b}** were observed in only {pr['n_common']} shared months; a relationship needs at least "
            f"{MIN_OVERLAP_MONTHS} shared months.",
            f"**{name_a}** ve **{name_b}** yalnızca {pr['n_common']} ortak ayda gözlenmiş; ilişki için en az "
            f"{MIN_OVERLAP_MONTHS} ortak ay gerekir."
        )
    else:
        verdict_badge = _REL_LABEL[pr["relationship"]]
        strength = (L("weak", "zayıf") if abs(pr["rho"]) < 0.3
                    else (L("moderate", "orta") if abs(pr["rho"]) < 0.6 else L("strong", "güçlü")))
        explanation = L(
            f"After removing seasonality and the linear trend, **{name_a}** and **{name_b}** have "
            f"Spearman $\\rho = {pr['rho']:+.3f}$ ({strength}), {_p_phrase(pr['p'])} "
            f"({pr['n_common']} shared months, autocorrelation-corrected effective n ≈ {pr['n_eff']:.0f}). "
            f"Highest lagged correlation r = {pr['best_xcorr']:+.3f}: {_lag_sentence(name_a, name_b, pr['best_lag'])} "
            f"(descriptive; not used in the verdict). Slopes: {name_a} `{slope_a:+.4f}/yr`, {name_b} `{slope_b:+.4f}/yr` "
            f"($\\Delta\\text{{Slope}} = {delta_slope:+.4f}/\\text{{yr}}$). Correlation shows an association, "
            f"not a causal link (teleconnection mechanism).",
            f"Mevsimsellik ve doğrusal trend çıkarıldıktan sonra **{name_a}** ile **{name_b}** arasında "
            f"Spearman $\\rho = {pr['rho']:+.3f}$ ({strength}), {_p_phrase(pr['p'])} "
            f"({pr['n_common']} ortak ay, otokorelasyon düzeltmeli etkin n ≈ {pr['n_eff']:.0f}). "
            f"En yüksek gecikmeli korelasyon r = {pr['best_xcorr']:+.3f}: {_lag_sentence(name_a, name_b, pr['best_lag'])} "
            f"(betimsel; karara katılmaz). Eğimler: {name_a} `{slope_a:+.4f}/yıl`, {name_b} `{slope_b:+.4f}/yıl` "
            f"($\\Delta\\text{{Slope}} = {delta_slope:+.4f}/\\text{{yıl}}$). Korelasyon bir ilişkiyi gösterir, "
            f"nedensel bir bağlantı (teleconnection mekanizması) anlamına gelmez."
        )

    return {
        "valid": bool(pr["valid"]),
        "reason": "" if pr["valid"] else (
            L("The two points are in the same NASA POWER cell (identical series).", "İki nokta aynı NASA POWER hücresinde (özdeş seriler).")
            if pr["relationship"] == "identical"
            else L(f"Shared observations {pr['n_common']} months < {MIN_OVERLAP_MONTHS}.", f"Ortak gözlem {pr['n_common']} ay < {MIN_OVERLAP_MONTHS}.")
        ),
        "region_stats": region_stats,
        "corr_df": corr_df,
        "pairwise_df": pd.DataFrame(pairwise_records),
        "spearman_rho": round(pr["rho"], 4),
        "spearman_p": pr["p"],
        "p_value_display": _format_p(pr["p"]) if pr["valid"] else "—",
        "pearson_r": round(pr["pearson"], 4),
        "best_lag_months": int(pr["best_lag"]),
        "best_xcorr": round(pr["best_xcorr"], 4),
        "lag_convention": lag_convention,
        "n_common_months": pr["n_common"],
        "n_effective": round(pr["n_eff"], 1),
        "slope_a": round(slope_a, 5),
        "slope_b": round(slope_b, 5),
        "delta_slope": round(delta_slope, 5),
        "opposite_slopes": bool(slope_a * slope_b < 0),
        "relationship": pr["relationship"],
        "is_teleconnection": pr["relationship"] == "opposite",
        "verdict_badge": verdict_badge,
        "explanation": explanation,
    }
