"""
NASA Earth Data Detective (EDD) - İstatistiksel Analiz, STL Ayrıştırma, Nesnel Ağırlıklandırma (CRITIC + VIF)
ve 5 Yıllık (60 Ay) SARIMA İklim Projeksiyon Motoru (statistics_engine.py)

Bilimsel Metodoloji Referansları:
1. STL Decomposition (Cleveland et al., 1990): Trend + Seasonal + Residual ayrıştırması (statsmodels.tsa.seasonal.STL)
2. Modified Mann-Kendall (Hamed & Rao, 1998): Otokorelasyon düzeltmeli parametrik olmayan trend anlamlılık testi
3. Theil-Sen Robust Estimator (Sen, 1968): Medyan eğim (Sen's Slope) ve %95 Güven Aralığı
4. CRITIC + IPCC AR6 + VIF (> 5) Çoklu Doğrusallık Koruması: Kullanıcı müdahalesine kapalı nesnel ağırlıklandırma
5. SARIMA (Box-Jenkins / statsmodels SARIMAX): 60 aylık (2026–2030) gelecek projeksiyonu ve %95 güven bandı
"""

from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX


def decimal_years(df: pd.DataFrame) -> np.ndarray:
    """Aylık DataFrame için gerçek tarih ekseni: yıl + (ay - 1) / 12."""
    return df["year"].to_numpy(dtype=float) + (df["month"].to_numpy(dtype=float) - 1.0) / 12.0


def compute_shared_weights(
    region_stl_maps: List[Dict[str, Dict[str, Any]]],
    selected_vars: List[str],
    official_variables_meta: Dict[str, Dict[str, Any]],
    hide_seasonality: bool = True,
) -> Dict[str, Any]:
    """
    Çiftli/Çoklu bölge kıyasında TÜM bölgeler için ORTAK nesnel ağırlık üretir.
    Her bölgede IPCC AR6 + CRITIC + VIF ağırlıkları hesaplanır, bölgeler arası ortalaması alınır
    ve yeniden 1.00'e normalize edilir. Böylece bütünleşik çizgiler aynı ağırlık setiyle
    (elma-elma) karşılaştırılır. Yalnızca tüm bölgelerde geçerli verisi olan parametreler kullanılır.
    """
    common_vars = [
        v for v in selected_vars
        if v in official_variables_meta and all(v in m and m[v].get("valid", False) for m in region_stl_maps)
    ]
    dropped = [v for v in selected_vars if v not in common_vars]
    if not common_vars:
        return {"valid": False, "weights_df": pd.DataFrame(), "vif_warnings": [], "common_vars": [], "dropped_vars": dropped}

    per_region = [
        compute_objective_weights_and_composite(m, common_vars, official_variables_meta, hide_seasonality)
        for m in region_stl_maps
    ]
    per_region = [r for r in per_region if r["valid"]]
    stacked = pd.concat([r["weights_df"] for r in per_region], ignore_index=True)

    agg = (
        stacked.groupby("var_code")
        .agg(
            rank=("rank", "first"),
            label=("label", "first"),
            ipcc_base_weight=("ipcc_base_weight", "first"),
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
    agg["vif_status"] = agg["vif_value"].apply(
        lambda x: f"⚠️ VIF = {x:.2f} (> 5.0 Sönümlendi)" if x > 5.0 else f"🟢 VIF = {x:.2f} (<= 5.0 Uygun)"
    )

    vif_warnings: List[str] = []
    for r in per_region:
        vif_warnings.extend(r["vif_warnings"])

    return {
        "valid": True,
        "weights_df": agg,
        "vif_warnings": list(dict.fromkeys(vif_warnings)),
        "common_vars": common_vars,
        "dropped_vars": dropped,
    }


def apply_weights_to_composite(
    param_stl_map: Dict[str, Dict[str, Any]],
    weights_df: pd.DataFrame,
    hide_seasonality: bool = True,
) -> pd.DataFrame:
    """Verilen (ortak) ağırlık setiyle bir bölgenin ağırlıklı bütünleşik Z-skoru serisini üretir."""
    merged: Optional[pd.DataFrame] = None
    col = "anomaly" if hide_seasonality else "observed"
    for _, row in weights_df.iterrows():
        v = row["var_code"]
        sub = param_stl_map[v]["df"][["date", "year", "month", col]].rename(columns={col: v})
        merged = sub if merged is None else pd.merge(merged, sub, on=["date", "year", "month"], how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["date", "year", "month", "value", "composite_z", "source_tag"])

    merged = merged.sort_values("date").reset_index(drop=True)
    comp = np.zeros(len(merged), dtype=float)
    for _, row in weights_df.iterrows():
        arr = merged[row["var_code"]].to_numpy(dtype=float)
        sd = float(np.nanstd(arr))
        z = (arr - float(np.nanmean(arr))) / sd if sd > 1e-9 else np.zeros_like(arr)
        comp += float(row["normalized_weight"]) * float(row["polarity_sign"]) * z

    return pd.DataFrame(
        {
            "date": merged["date"],
            "year": merged["year"],
            "month": merged["month"],
            "value": np.round(comp, 4),
            "composite_z": np.round(comp, 4),
            "source_tag": "Ortak Ağırlıklı Bütünleşik İndeks",
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
    Aylık zaman serisini STL (Seasonal-Trend decomposition using LOESS) ile
    `Trend + Seasonal + Residual` bileşenlerine ayırır.

    Döndürülen DataFrame sütunları:
    - date, year, month, observed
    - trend, seasonal, resid
    - deseasonalized (Mevsimsellikten arındırılmış mutlak seri: observed - seasonal)
    - anomaly (Mevsimsellikten arındırılmış anomali serisi: deseasonalized - mean(deseasonalized))
    """
    if df is None or df.empty or "value" not in df.columns:
        return {"valid": False, "df": pd.DataFrame()}

    clean = df.dropna(subset=["value"]).sort_values("date").copy()
    if len(clean) < max(24, period * 2):
        return {"valid": False, "df": pd.DataFrame()}

    clean = clean.reset_index(drop=True)
    y = clean["value"].to_numpy(dtype=float)

    try:
        stl = STL(y, period=period, robust=True)
        res = stl.fit()
        trend = np.asarray(res.trend, dtype=float)
        seasonal = np.asarray(res.seasonal, dtype=float)
        resid = np.asarray(res.resid, dtype=float)
    except Exception:
        # Fallback: aylık klimatolojik ortalama ile mevsimsel ayrıştırma
        month_means = clean.groupby("month")["value"].transform("mean").to_numpy(dtype=float)
        seasonal = month_means - float(np.nanmean(y))
        deseas_tmp = y - seasonal
        trend = pd.Series(deseas_tmp).rolling(window=period, min_periods=1, center=True).mean().to_numpy(dtype=float)
        resid = deseas_tmp - trend

    deseasonalized = y - seasonal
    clim_mean = float(np.nanmean(deseasonalized))
    anomaly = deseasonalized - clim_mean

    out_df = pd.DataFrame(
        {
            "date": clean["date"],
            "year": clean["year"].astype(int),
            "month": clean["month"].astype(int),
            "observed": np.round(y, 4),
            "trend": np.round(trend, 4),
            "seasonal": np.round(seasonal, 4),
            "resid": np.round(resid, 4),
            "deseasonalized": np.round(deseasonalized, 4),
            "anomaly": np.round(anomaly, 4),
            "source_tag": clean["source_tag"] if "source_tag" in clean.columns else "Resmi Arşiv",
        }
    )

    return {
        "valid": True,
        "climatology_mean": round(clim_mean, 4),
        "seasonal_amplitude": round(float(np.nanpercentile(seasonal, 95) - np.nanpercentile(seasonal, 5)), 4),
        "df": out_df,
    }


def hamed_rao_mann_kendall_test(
    dates_or_years: np.ndarray,
    values: np.ndarray,
    alpha: float = 0.05,
    is_monthly: bool = True,
) -> Dict[str, Any]:
    """
    Mevsimsellikten arındırılmış seri üzerinde Otokorelasyon Düzeltmeli
    Modified Mann-Kendall (Hamed & Rao, 1998) testi ve Theil-Sen Medyan Eğim (Sen's Slope)
    hesaplamasını yürütür.

    Çıktılar:
    - S, var_s_raw, var_s_corrected, z_score, p_value, kendall_tau
    - sen_slope_annual (Yıllık değişim hızı), sen_ci_low_annual, sen_ci_high_annual (%95 Güven Aralığı)
    - sen_intercept, trend_line
    - is_significant (p < 0.05), badge_text, direction_tr
    """
    y = np.asarray(values, dtype=float)
    mask = ~np.isnan(y)
    y = y[mask]
    n = len(y)

    if n < 6:
        return {
            "S": 0,
            "var_s_raw": 0.0,
            "var_s_corrected": 0.0,
            "autocorr_factor": 1.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "kendall_tau": 0.0,
            "sen_slope_annual": 0.0,
            "sen_slope_decade": 0.0,
            "sen_ci_low_annual": 0.0,
            "sen_ci_high_annual": 0.0,
            "sen_intercept": 0.0,
            "trend_line": np.zeros_like(y),
            "is_significant": False,
            "direction": "stable",
            "direction_tr": "Belirgin Eğilim Yok",
            "badge_text": "🟡 Henüz Anlamlı Değil (Gürültü Baskın / Daha Uzun Süre Gerekli)",
        }

    # Zaman eksenini yıl cinsinden oluştur (Sen's Slope yıllık değişim hızı versin).
    # Ondalık yıl dizisi (yıl + (ay-1)/12) verilirse gerçek tarih ekseni kullanılır; böylece
    # eksik aylar/boşluklar zaman eksenini sıkıştırıp eğimi bozmaz.
    t_raw = np.asarray(dates_or_years, dtype=float) if dates_or_years is not None else np.array([])
    if t_raw.shape[0] == mask.shape[0] and len(np.unique(t_raw[mask])) == n:
        t_years = t_raw[mask] - t_raw[mask][0]
    elif is_monthly:
        t_years = np.arange(n, dtype=float) / 12.0
    else:
        t_years = np.arange(n, dtype=float)

    # 1. Theil-Sen Medyan Eğim ve %95 Güven Aralığı
    res_theil = stats.theilslopes(y, t_years, alpha=1.0 - alpha)
    sen_slope_annual = float(res_theil.slope)
    sen_intercept = float(res_theil.intercept)
    sen_ci_low_annual = float(res_theil.low_slope)
    sen_ci_high_annual = float(res_theil.high_slope)
    trend_line = sen_intercept + sen_slope_annual * t_years

    # 2. Mann-Kendall S İstatistiği (Vektörel hesaplama)
    # i < j için sign(y[j] - y[i])
    diff_matrix = y[np.newaxis, :] - y[:, np.newaxis]
    upper_tri = np.triu_indices(n, k=1)
    signs = np.sign(diff_matrix[upper_tri])
    s_stat = int(np.sum(signs))

    n_pairs = n * (n - 1) / 2.0
    kendall_tau = float(s_stat / n_pairs) if n_pairs > 0 else 0.0

    # 3. Bağlı Değer (Tie) Düzeltmeli Ham Varyans Var0(S)
    _, tie_counts = np.unique(np.round(y, 6), return_counts=True)
    tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    var_s_raw = float((n * (n - 1) * (2 * n + 5) - tie_term) / 18.0)

    # 4. Hamed & Rao (1998) Otokorelasyon Varyans Düzeltmesi (n / n_S*)
    detrended = y - trend_line
    ranks = stats.rankdata(detrended)
    ranks_centered = ranks - np.mean(ranks)
    denom = float(np.sum(ranks_centered**2))

    max_lag = min(n - 3, 24 if is_monthly else 10)
    crit_r = 1.96 / np.sqrt(n)  # %95 anlamlılık sınırındaki otokorelasyon gecikmeleri
    correction_sum = 0.0

    if denom > 0 and max_lag >= 1:
        for k in range(1, max_lag + 1):
            r_k = float(np.sum(ranks_centered[:-k] * ranks_centered[k:]) / denom)
            if abs(r_k) > crit_r:
                correction_sum += (n - k) * (n - k - 1) * (n - k - 2) * r_k

    n_ns_ratio = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * correction_sum
    autocorr_factor = float(np.clip(n_ns_ratio, 0.4, 12.0))
    var_s_corrected = max(1e-9, var_s_raw * autocorr_factor)

    # 5. Z-Skoru ve Çift Yönlü p-değeri
    if s_stat > 0:
        z_score = (s_stat - 1.0) / np.sqrt(var_s_corrected)
    elif s_stat < 0:
        z_score = (s_stat + 1.0) / np.sqrt(var_s_corrected)
    else:
        z_score = 0.0

    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    is_significant = bool(p_value < alpha)

    if s_stat > 0:
        direction = "increasing"
        direction_tr = "Artış Trendi (Pozitif Eğim)"
    elif s_stat < 0:
        direction = "decreasing"
        direction_tr = "Azalış Trendi (Negatif Eğim)"
    else:
        direction = "stable"
        direction_tr = "Durağan (Sıfır Eğim)"

    if is_significant:
        badge_text = "🟢 İstatistiksel Olarak Anlamlı Trend (p < 0.05)"
    else:
        badge_text = "🟡 Henüz Anlamlı Değil (Gürültü Baskın / Daha Uzun Süre Gerekli)"

    return {
        "S": s_stat,
        "var_s_raw": round(var_s_raw, 2),
        "var_s_corrected": round(var_s_corrected, 2),
        "autocorr_factor": round(autocorr_factor, 3),
        "z_score": round(float(z_score), 4),
        "p_value": round(p_value, 6),
        "kendall_tau": round(kendall_tau, 4),
        "sen_slope_annual": round(sen_slope_annual, 5),
        "sen_slope_decade": round(sen_slope_annual * 10.0, 4),
        "sen_ci_low_annual": round(sen_ci_low_annual, 5),
        "sen_ci_high_annual": round(sen_ci_high_annual, 5),
        "sen_intercept": round(sen_intercept, 4),
        "trend_line": np.round(trend_line, 4),
        "is_significant": is_significant,
        "direction": direction,
        "direction_tr": direction_tr,
        "badge_text": badge_text,
    }


def compute_objective_weights_and_composite(
    param_stl_map: Dict[str, Dict[str, Any]],
    selected_vars: List[str],
    official_variables_meta: Dict[str, Dict[str, Any]],
    hide_seasonality: bool = True,
) -> Dict[str, Any]:
    """
    Seçilen 1 ila 4 parametre için kullanıcı müdahalesine kapalı, nesnel ağırlıklandırma:
    1. IPCC AR6 Radyatif Zorlama / İklim Etkisi Baz Ağırlığı (base_weight)
    2. CRITIC (Criteria Importance Through Intercriteria Correlation) Algoritması:
       C_j = sigma_j * sum_{k}(1 - |r_{jk}|)
    3. Çoklu Doğrusallık (Multicollinearity) Koruması (VIF > 5 Denetimi):
       Birbirinden türetilen veya yüksek korelasyonlu değişkenler (örn. Yağış + Toprak Nemi)
       birlikte seçildiğinde VIF_j = diag(R^-1)_j > 5.0 ise sinyal şişmesini önlemek için
       1 / sqrt(VIF_j / 5.0) sönümleme çarpanı uygulanır.
    4. Seçilen alt kümenin ağırlıkları toplamı 1.00 (%100) olacak şekilde normalize edilir.
    """
    valid_vars = [
        v for v in selected_vars
        if v in param_stl_map and v in official_variables_meta and param_stl_map[v].get("valid", False) and not param_stl_map[v]["df"].empty
    ]

    if not valid_vars:
        return {
            "valid": False,
            "weights_df": pd.DataFrame(),
            "composite_df": pd.DataFrame(),
            "vif_warnings": [],
        }

    # Ortak tarih ekseninde birleştir
    merged: Optional[pd.DataFrame] = None
    col_choice = "anomaly" if hide_seasonality else "observed"

    for v in valid_vars:
        sub = param_stl_map[v]["df"][["date", "year", "month", "observed", "deseasonalized", "anomaly"]].copy()
        sub = sub.rename(
            columns={
                "observed": f"obs_{v}",
                "deseasonalized": f"deseas_{v}",
                "anomaly": f"anom_{v}",
            }
        )
        if merged is None:
            merged = sub
        else:
            merged = pd.merge(merged, sub, on=["date", "year", "month"], how="inner")

    if merged is None or merged.empty:
        return {
            "valid": False,
            "weights_df": pd.DataFrame(),
            "composite_df": pd.DataFrame(),
            "vif_warnings": [],
        }

    merged = merged.sort_values("date").reset_index(drop=True)
    k = len(valid_vars)

    # CRITIC ve VIF hesabı için mevsimsellikten arındırılmış (deseasonalized) matris kullanılır
    deseas_matrix = np.column_stack([merged[f"deseas_{v}"].to_numpy(dtype=float) for v in valid_vars])

    # Min-Max normalizasyonu (CRITIC standart sapması için)
    norm_matrix = np.zeros_like(deseas_matrix)
    for j in range(k):
        col = deseas_matrix[:, j]
        c_min, c_max = float(np.nanmin(col)), float(np.nanmax(col))
        if c_max - c_min > 1e-9:
            norm_matrix[:, j] = (col - c_min) / (c_max - c_min)
        else:
            norm_matrix[:, j] = 0.5

    if k == 1:
        v0 = valid_vars[0]
        meta0 = official_variables_meta[v0]
        weights_records = [
            {
                "var_code": v0,
                "rank": meta0["rank"],
                "label": meta0["short_label"],
                "ipcc_base_weight": meta0["base_weight"],
                "critic_score": 1.0,
                "vif_value": 1.0,
                "vif_status": "🟢 Bağımsız Sinyal (VIF = 1.00)",
                "normalized_weight": 1.0,
                "weight_pct": 100.0,
                "polarity_sign": 1.0,
            }
        ]
        # Tek parametrede bileşik seri doğrudan kendisidir (veya Z-skoru)
        active_vals = merged[f"anom_{v0}" if hide_seasonality else f"obs_{v0}"].to_numpy(dtype=float)
        std_val = float(np.nanstd(active_vals))
        z_vals = (active_vals - float(np.nanmean(active_vals))) / std_val if std_val > 1e-9 else np.zeros_like(active_vals)

        comp_df = pd.DataFrame(
            {
                "date": merged["date"],
                "year": merged["year"],
                "month": merged["month"],
                "value": np.round(active_vals if not hide_seasonality else z_vals, 4),
                "composite_z": np.round(z_vals, 4),
                "source_tag": "Hibrit Çoklu Kaynak",
            }
        )
        return {
            "valid": True,
            "weights_df": pd.DataFrame(weights_records),
            "composite_df": comp_df,
            "vif_warnings": [],
        }

    # k >= 2 için Korelasyon Matrisi, CRITIC ve VIF (> 5) Hesabı
    corr_matrix = np.corrcoef(norm_matrix, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # VIF = diag((R + reg*I)^-1)
    reg_corr = corr_matrix + np.eye(k) * 1e-4
    try:
        inv_corr = np.linalg.inv(reg_corr)
        vif_values = np.maximum(1.0, np.diag(inv_corr))
    except np.linalg.LinAlgError:
        vif_values = np.ones(k, dtype=float)

    std_devs = np.std(norm_matrix, axis=0, ddof=1)
    conflict_sums = np.sum(1.0 - np.abs(corr_matrix), axis=1)
    critic_raw = std_devs * conflict_sums
    max_critic = float(np.max(critic_raw)) if np.max(critic_raw) > 1e-9 else 1.0
    critic_norm = critic_raw / max_critic

    # Polarite hizalaması:
    # Eğer seçilen tüm değişkenler negatif polariteli (Sıcaklık, Sera Gazı, AOD, Rüzgar) ise kendi yönlerini koru (+1).
    # Karışık seçim varsa (örn. Yağış + Toprak Nemi + Yüzey Sıcaklığı), zıt yönlü değişkenlerin birbirini
    # sıfırlamaması için birincil değişkenin fiziksel polaritesiyle hizala.
    polarities = [float(official_variables_meta[v].get("polarity", 1.0)) for v in valid_vars]
    if all(p < 0 for p in polarities) or all(p > 0 for p in polarities):
        aligned_signs = [1.0 for _ in valid_vars]
    else:
        # Hidrolojik/Kriyosferik (+1) değişkenlerle Termal/Zorlama (-1) değişkenleri aynı fiziksel eksende hizala
        ref_pol = polarities[0]
        aligned_signs = [1.0 if p == ref_pol else -1.0 for p in polarities]

    raw_combined_weights = np.zeros(k, dtype=float)
    vif_warnings: List[str] = []
    weights_records = []

    for j, v in enumerate(valid_vars):
        meta = official_variables_meta[v]
        base_w = float(meta["base_weight"])
        c_score = float(critic_norm[j])
        vif_val = float(vif_values[j])

        if vif_val > 5.0:
            vif_penalty = 1.0 / np.sqrt(vif_val / 5.0)
            vif_status = f"⚠️ VIF = {vif_val:.2f} (> 5.0 Çoklu Doğrusallık Sönümlendi: x{vif_penalty:.2f})"
            vif_warnings.append(
                f"**{meta['short_label']}** değişkeninde yüksek çoklu doğrusallık (**VIF = {vif_val:.2f} > 5.0**) "
                f"tespit edildi. Ortak sinyal şişmesini önlemek için ağırlığı otomatik olarak dengelendi."
            )
        else:
            vif_penalty = 1.0
            vif_status = f"🟢 VIF = {vif_val:.2f} (<= 5.0 Uygun)"

        raw_w = base_w * (0.65 + 0.35 * c_score) * vif_penalty
        raw_combined_weights[j] = raw_w
        weights_records.append(
            {
                "var_code": v,
                "rank": meta["rank"],
                "label": meta["short_label"],
                "ipcc_base_weight": round(base_w, 3),
                "critic_score": round(c_score, 3),
                "vif_value": round(vif_val, 2),
                "vif_status": vif_status,
                "raw_weight": raw_w,
                "polarity_sign": aligned_signs[j],
            }
        )

    total_w = float(np.sum(raw_combined_weights))
    norm_weights = raw_combined_weights / total_w if total_w > 0 else np.ones(k) / k

    for j in range(k):
        weights_records[j]["normalized_weight"] = round(float(norm_weights[j]), 4)
        weights_records[j]["weight_pct"] = round(float(norm_weights[j]) * 100.0, 2)

    # Ağırlıklı Bütünleşik Z-Skoru Zaman Serisi Hesabı
    composite_z = np.zeros(len(merged), dtype=float)
    for j, v in enumerate(valid_vars):
        col_name = f"anom_{v}" if hide_seasonality else f"obs_{v}"
        arr = merged[col_name].to_numpy(dtype=float)
        arr_std = float(np.nanstd(arr))
        z_j = (arr - float(np.nanmean(arr))) / arr_std if arr_std > 1e-9 else np.zeros_like(arr)
        composite_z += norm_weights[j] * aligned_signs[j] * z_j

    comp_df = pd.DataFrame(
        {
            "date": merged["date"],
            "year": merged["year"],
            "month": merged["month"],
            "value": np.round(composite_z, 4),
            "composite_z": np.round(composite_z, 4),
            "source_tag": "IPCC AR6 + CRITIC + VIF Bütünleşik İndeks",
        }
    )

    return {
        "valid": True,
        "weights_df": pd.DataFrame(weights_records),
        "composite_df": comp_df,
        "vif_warnings": vif_warnings,
    }


def generate_5year_sarima_forecast(
    df: pd.DataFrame,
    value_col: str = "value",
    hide_seasonality: bool = True,
    steps: int = 60,
) -> pd.DataFrame:
    """
    1980–2025 tarihsel verisi üzerinde SARIMAX (Seasonal Autoregressive Integrated Moving Average)
    modelini eğiterek gelecek 60 ay (5 Yıl: 2026–2030) için noktasal tahmin ve %95 Güven Aralığı üretir.
    """
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["date", "forecast", "lower_95", "upper_95", "model_name"])

    clean = df.dropna(subset=[value_col]).sort_values("date").reset_index(drop=True)
    if len(clean) < 24:
        return pd.DataFrame(columns=["date", "forecast", "lower_95", "upper_95", "model_name"])

    y = clean[value_col].to_numpy(dtype=float)
    last_date = pd.Timestamp(clean["date"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq="MS",
    )

    months_hist = clean["date"].dt.month.to_numpy(dtype=float)
    months_fut = future_dates.month.to_numpy(dtype=float)

    def _build_fourier_exog(m_arr: np.ndarray) -> np.ndarray:
        angle1 = 2.0 * np.pi * m_arr / 12.0
        angle2 = 4.0 * np.pi * m_arr / 12.0
        return np.column_stack([np.sin(angle1), np.cos(angle1), np.sin(angle2), np.cos(angle2)])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hide_seasonality:
                # Mevsimsellikten arındırılmış seride SARIMAX(1, 1, 1) + drift
                model = SARIMAX(
                    y,
                    order=(1, 1, 1),
                    trend="t",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_res = model.fit(disp=False, maxiter=60)
                pred_res = fit_res.get_forecast(steps=steps)
                model_label = "SARIMAX(1,1,1) + Trend"
            else:
                # Ham seride 12 aylık harmonik mevsimsel bileşenli SARIMAX
                exog_hist = _build_fourier_exog(months_hist)
                exog_fut = _build_fourier_exog(months_fut)
                model = SARIMAX(
                    y,
                    exog=exog_hist,
                    order=(1, 1, 1),
                    trend="t",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_res = model.fit(disp=False, maxiter=60)
                pred_res = fit_res.get_forecast(steps=steps, exog=exog_fut)
                model_label = "SARIMA(1,1,1)x(Harmonic-12)"

            mean_fc = np.asarray(pred_res.predicted_mean, dtype=float)
            ci_arr = np.asarray(pred_res.conf_int(alpha=0.05), dtype=float)
            lower_95 = ci_arr[:, 0]
            upper_95 = ci_arr[:, 1]
    except Exception:
        # Analitik Theil-Sen + Mevsimsel Harmonik + AR(1) yedek projeksiyonu
        t_idx = np.arange(len(y), dtype=float)
        fut_idx = np.arange(len(y), len(y) + steps, dtype=float)
        res_ts = stats.theilslopes(y, t_idx)
        trend_hist = res_ts.intercept + res_ts.slope * t_idx
        trend_fut = res_ts.intercept + res_ts.slope * fut_idx

        if not hide_seasonality:
            ex_h = _build_fourier_exog(months_hist)
            ex_f = _build_fourier_exog(months_fut)
            beta, _, _, _ = np.linalg.lstsq(ex_h, y - trend_hist, rcond=None)
            seas_fut = ex_f @ beta
            resid_std = float(np.nanstd(y - trend_hist - (ex_h @ beta)))
        else:
            seas_fut = np.zeros(steps, dtype=float)
            resid_std = float(np.nanstd(y - trend_hist))

        mean_fc = trend_fut + seas_fut
        horizon_scale = np.sqrt(1.0 + np.arange(1, steps + 1, dtype=float) * 0.08)
        half_width = 1.96 * resid_std * horizon_scale
        lower_95 = mean_fc - half_width
        upper_95 = mean_fc + half_width
        model_label = "SARIMA-Fallback (Robust Trend + Harmonik)"

    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast": np.round(mean_fc, 4),
            "lower_95": np.round(lower_95, 4),
            "upper_95": np.round(upper_95, 4),
            "model_name": model_label,
        }
    )


def compare_regions_teleconnection(
    region_series: Dict[str, pd.DataFrame],
    value_col: str = "value",
) -> Dict[str, Any]:
    """
    Çiftli (Dual) ve Çoklu (Multi - 10 bölgeye kadar) modlarda bölgeler arası
    Çapraz Korelasyon (Cross-Correlation), Spearman Sıra Korelasyonu ve Eğim Farkı (Delta Slope)
    hesaplayarak 'Ters Yönlü İklim Tepkisi / Teleconnection' tespiti yapar.
    """
    region_names = list(region_series.keys())
    if len(region_names) < 2:
        return {"valid": False}

    # Her bölgenin Hamed-Rao Mann-Kendall ve Sen's Slope özetini çıkar
    region_stats: Dict[str, Dict[str, Any]] = {}
    merged: Optional[pd.DataFrame] = None

    for idx, r_name in enumerate(region_names):
        df_r = region_series[r_name].dropna(subset=[value_col]).sort_values("date")
        mk_r = hamed_rao_mann_kendall_test(
            decimal_years(df_r),
            df_r[value_col].to_numpy(dtype=float),
            is_monthly=True,
        )
        region_stats[r_name] = mk_r

        sub = df_r[["date", value_col]].rename(columns={value_col: f"col_{idx}"})
        if merged is None:
            merged = sub
        else:
            merged = pd.merge(merged, sub, on="date", how="inner")

    if merged is None or len(merged) < 12:
        return {"valid": False}

    merged = merged.sort_values("date").reset_index(drop=True)
    n_reg = len(region_names)

    # Korelasyon matrisi (N x N)
    val_matrix = np.column_stack([merged[f"col_{i}"].to_numpy(dtype=float) for i in range(n_reg)])
    corr_matrix = np.corrcoef(val_matrix, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
    corr_df = pd.DataFrame(np.round(corr_matrix, 3), index=region_names, columns=region_names)

    # İlk iki ana bölge (Dual mod için A ve B, Multi mod için tüm çiftler)
    name_a, name_b = region_names[0], region_names[1]
    vec_a = val_matrix[:, 0]
    vec_b = val_matrix[:, 1]

    spearman_res = stats.spearmanr(vec_a, vec_b)
    rho = float(spearman_res.statistic) if not np.isnan(spearman_res.statistic) else 0.0
    p_rho = float(spearman_res.pvalue) if not np.isnan(spearman_res.pvalue) else 1.0
    pearson_r = float(corr_matrix[0, 1])

    # Gecikmeli Çapraz Korelasyon (-6 ay ... +6 ay)
    a_norm = (vec_a - np.mean(vec_a)) / max(1e-9, np.std(vec_a))
    b_norm = (vec_b - np.mean(vec_b)) / max(1e-9, np.std(vec_b))
    best_lag = 0
    best_xcorr = pearson_r
    for lag in range(-6, 7):
        if lag < 0:
            xc = float(np.mean(a_norm[:lag] * b_norm[-lag:]))
        elif lag > 0:
            xc = float(np.mean(a_norm[lag:] * b_norm[:-lag]))
        else:
            xc = pearson_r
        if abs(xc) > abs(best_xcorr):
            best_xcorr = xc
            best_lag = lag

    slope_a = float(region_stats[name_a]["sen_slope_annual"])
    slope_b = float(region_stats[name_b]["sen_slope_annual"])
    delta_slope = slope_a - slope_b
    opposite_slopes = (slope_a * slope_b) < 0

    # Tüm bölge çiftleri için özet liste
    pairwise_records = []
    for i in range(n_reg):
        for j in range(i + 1, n_reg):
            ra, rb = region_names[i], region_names[j]
            sa = float(region_stats[ra]["sen_slope_annual"])
            sb = float(region_stats[rb]["sen_slope_annual"])
            r_ij = float(corr_matrix[i, j])
            opp = (sa * sb) < 0
            sig_ij = bool(region_stats[ra]["is_significant"] or region_stats[rb]["is_significant"])
            if (opp and sig_ij) or r_ij <= -0.35:
                status = "⚡ Ters Yönlü İklim Tepkisi / Teleconnection"
            elif r_ij >= 0.40 and not opp:
                status = "🔗 Eş Fazlı (Senkron) İklim Tepkisi"
            else:
                status = "⚖️ Zayıf / Kısmi Bağlantı"
            pairwise_records.append(
                {
                    "Bölge 1": ra,
                    "Bölge 2": rb,
                    "Eğim 1 (Yıllık)": round(sa, 4),
                    "Eğim 2 (Yıllık)": round(sb, 4),
                    "ΔSlope (Eğim Farkı)": round(sa - sb, 4),
                    "Çapraz Korelasyon (r)": round(r_ij, 3),
                    "Tepki Sınıfı": status,
                }
            )

    # Ters yönlü tepki: (a) güçlü negatif korelasyon VEYA (b) zıt işaretli eğimlerden en az biri
    # istatistiksel olarak anlamlı. Sadece işaret farkı (gürültü düzeyinde eğimler) yeterli sayılmaz.
    sig_any = bool(region_stats[name_a]["is_significant"] or region_stats[name_b]["is_significant"])
    is_teleconnection = bool((opposite_slopes and sig_any) or rho <= -0.35 or best_xcorr <= -0.35)
    if is_teleconnection:
        verdict_badge = "⚡ Ters Yönlü İklim Tepkisi / Teleconnection Doğrulandı"
        explanation = (
            f"**{name_a}** (Sen's Eğim: `{slope_a:+.4f}/yıl`) ile **{name_b}** "
            f"(Sen's Eğim: `{slope_b:+.4f}/yıl`) arasında zıt yönlü iklim eğilimi "
            f"($\\Delta\\text{{Slope}} = {delta_slope:+.4f}/\\text{{yıl}}$, Spearman $\\rho = {rho:.3f}$, "
            f"Maks. Çapraz Korelasyon $r = {best_xcorr:.3f}$ @ {best_lag} ay gecikme) tespit edilmiştir."
        )
    else:
        verdict_badge = "🔗 Eş Fazlı / Benzer Yönlü İklim Tepkisi"
        explanation = (
            f"**{name_a}** (Sen's Eğim: `{slope_a:+.4f}/yıl`) ve **{name_b}** "
            f"(Sen's Eğim: `{slope_b:+.4f}/yıl`) benzer yönde değişim göstermektedir "
            f"($\\Delta\\text{{Slope}} = {delta_slope:+.4f}/\\text{{yıl}}$, Spearman $\\rho = {rho:.3f}$)."
        )

    return {
        "valid": True,
        "region_stats": region_stats,
        "corr_df": corr_df,
        "pairwise_df": pd.DataFrame(pairwise_records),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p_rho, 6),
        "pearson_r": round(pearson_r, 4),
        "best_lag_months": best_lag,
        "best_xcorr": round(best_xcorr, 4),
        "slope_a": round(slope_a, 5),
        "slope_b": round(slope_b, 5),
        "delta_slope": round(delta_slope, 5),
        "opposite_slopes": opposite_slopes,
        "is_teleconnection": is_teleconnection,
        "verdict_badge": verdict_badge,
        "explanation": explanation,
    }
