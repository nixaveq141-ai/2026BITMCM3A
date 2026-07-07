import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datas"
Q2_DIR = ROOT / "q2"
OUTPUT_DIR = ROOT / "q3" / "outputs"
BASE_SCRIPT = Q2_DIR / "run_q2_analysis.py"

CHINA_MARKET = "China"
Q2_LAYERED_OUTPUT_DIR = Q2_DIR / "outputs_layered_market"

MATURE_SCORE_MAP = {"Mature": 0.85, "Growing": 0.60, "Emerging": 0.35}
COMPETITION_SCORE_MAP = {"High": 0.85, "Medium": 0.55, "Low": 0.25}
LAYER_LABELS = {5: "S", 4: "A", 3: "B", 2: "C", 1: "D"}
OBSERVED_LAYER_MAP = {
    "United_States": 5,
    "Japan": 4,
    "United_Kingdom": 4,
    "Germany": 4,
    "South_Korea": 4,
    "Brazil": 3,
    "Hong_Kong": 2,
    "Australia": 2,
    "Vietnam": 1,
}


@dataclass
class ChinaAssumptions:
    fifa_initial_ask: float = 275.0
    cctv_counter_offer: float = 55.0
    final_deal_price: float = 60.0
    hhi: float = 0.82
    maturity_score: float = 0.60
    effective_bidders: int = 1
    broadcaster_count: int = 1
    rho_alpha: float = 0.55
    rho_eta: float = 0.30
    rho_omega: float = 0.50
    tau: float = 2.0
    lambda1: float = 0.20
    lambda2: float = 0.10
    lambda3: float = 0.10
    lambda_m: float = 0.28
    fifa_anchor_rate: float = 0.22
    fifa_beta_extra: float = 0.20
    interval_delta1: float = 0.22
    interval_delta2: float = 0.27


@dataclass
class TunedParams:
    k: int
    alpha: float
    gamma1: float
    gamma2: float
    eta: float
    beta: float
    qualified_bonus: float
    unqualified_discount: float
    competition_c: float
    rights_split_step: float


def load_base_module():
    spec = importlib.util.spec_from_file_location("q2_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()
base.OUTPUT_DIR = OUTPUT_DIR


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_best_q2_params() -> TunedParams:
    default = TunedParams(
        k=2,
        alpha=0.4,
        gamma1=0.05,
        gamma2=0.05,
        eta=0.3,
        beta=0.35,
        qualified_bonus=0.05,
        unqualified_discount=0.05,
        competition_c=0.16,
        rights_split_step=0.0,
    )
    try:
        csv_path = Q2_LAYERED_OUTPUT_DIR / "第二问参数寻优结果_分层版.csv"
        tuning_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        best = tuning_df.iloc[0]
        return TunedParams(
            k=int(best["参考市场数量"]),
            alpha=float(best["结构价值弹性系数alpha"]),
            gamma1=float(best["GDP对数敏感系数gamma1"]),
            gamma2=float(best["人口对数敏感系数gamma2"]),
            eta=float(best["层级溢价强度eta"]),
            beta=float(best["beta"]),
            qualified_bonus=float(best["参赛上浮系数"]),
            unqualified_discount=float(best["未参赛折扣系数"]),
            competition_c=float(best["竞购系数"]),
            rights_split_step=float(best["结构系数"]),
        )
    except Exception:
        return default


def build_china_target_row() -> pd.DataFrame:
    hist = pd.read_csv(DATA_DIR / "02_historical_broadcasting_rights.csv")
    view = pd.read_csv(DATA_DIR / "05_worldcup_viewership.csv")
    econ = pd.read_csv(DATA_DIR / "03_economic_indicators.csv")
    football = pd.read_csv(DATA_DIR / "04_football_market_indicators.csv")
    timezone = pd.read_csv(DATA_DIR / "06_timezone_viewer_convenience.csv")
    market = pd.read_csv(DATA_DIR / "07_market_structure.csv")

    hist = hist[hist["country_region"] == CHINA_MARKET].copy()
    hist["price_usd_million"] = pd.to_numeric(hist["price_usd_million"], errors="coerce")
    hist["world_cup_year"] = pd.to_numeric(hist["world_cup_year"], errors="coerce")
    latest_hist = (
        hist.sort_values("world_cup_year")
        .groupby("country_region", as_index=False)
        .tail(1)[["country_region", "price_usd_million", "world_cup_year"]]
        .rename(columns={"price_usd_million": "历史转播权价格", "world_cup_year": "历史价格年份"})
    )

    view = view[view["country_region"] == CHINA_MARKET].copy()
    view["value_2022"] = pd.to_numeric(view["value_2022"], errors="coerce")
    view["value_2018"] = pd.to_numeric(view["value_2018"], errors="coerce")
    unit_factor = {
        "million_viewers": 1.0,
        "million_people": 1.0,
        "million_streams": 0.35,
        "billion_views": 250.0,
        "pct_of_global": 3.0,
    }
    view["单位换算系数"] = view["unit"].map(unit_factor).fillna(1.0)
    view["指标平均值"] = view[["value_2022", "value_2018"]].mean(axis=1, skipna=True)
    view["历史收视热度"] = view["指标平均值"] * view["单位换算系数"]
    china_view = view.groupby("country_region", as_index=False)["历史收视热度"].sum()

    timezone = timezone[timezone["country_region"] == CHINA_MARKET].copy()
    timezone["赛程时段因子"] = timezone.apply(base.schedule_based_time_factor, axis=1)
    timezone_summary = timezone.groupby("country_region", as_index=False).agg(
        观赛友好度评分=("viewer_friendliness_score_1_to_5", "mean"),
        黄金时段占比=("prime_time_match_pct", "mean"),
        赛程时段因子=("赛程时段因子", "mean"),
    )

    econ = econ[econ["country_region"] == CHINA_MARKET].copy()
    football = football[football["country_region"] == CHINA_MARKET].copy()
    market = market[market["country_region"] == CHINA_MARKET].copy()

    china_price_shell = pd.DataFrame(
        [
            {
                "country_region": CHINA_MARKET,
                "2026成交价": np.nan,
                "广播商数量": 1,
                "广播商列表": "央视",
                "合同结构": "2026_single",
            }
        ]
    )

    merged = china_price_shell.merge(econ, on="country_region", how="left")
    merged = merged.merge(football, on="country_region", how="left")
    merged = merged.merge(timezone_summary, on="country_region", how="left")
    merged = merged.merge(china_view, on="country_region", how="left")
    merged = merged.merge(latest_hist, on="country_region", how="left")
    merged = merged.merge(market, on="country_region", how="left")

    merged["市场"] = "中国内地"
    merged["人口规模"] = pd.to_numeric(merged["population_million_2024"], errors="coerce")
    merged["总GDP"] = pd.to_numeric(merged["total_gdp_billion_usd_2024"], errors="coerce")
    merged["人均GDP"] = pd.to_numeric(merged["gdp_per_capita_usd_2024"], errors="coerce")
    merged["体育媒体市场规模"] = pd.to_numeric(merged["sports_media_market_billion_usd_est"], errors="coerce")
    merged["球迷数量"] = pd.to_numeric(merged["football_fans_million"], errors="coerce")
    merged["球迷比例"] = pd.to_numeric(merged["football_fan_pct_of_population"], errors="coerce")
    merged["FIFA排名"] = pd.to_numeric(merged["fifa_ranking_june_2026"], errors="coerce")
    merged["赛程时段因子"] = pd.to_numeric(merged["赛程时段因子"], errors="coerce")
    merged["主要竞购方数量"] = pd.to_numeric(merged["num_major_bidders"], errors="coerce")
    merged["是否参赛"] = merged["qualified_2026_wc"].fillna("No").astype(str).str.startswith("Yes").astype(int)

    missing_gdp = merged["总GDP"].isna() & merged["人口规模"].notna() & merged["人均GDP"].notna()
    merged.loc[missing_gdp, "总GDP"] = merged.loc[missing_gdp, "人口规模"] * merged.loc[missing_gdp, "人均GDP"] / 1000.0

    merged["联赛强度得分"] = merged["domestic_league_strength_tier"].map(base.LEAGUE_STRENGTH_MAP).fillna(1.0)
    merged["FIFA排名"] = merged["FIFA排名"].fillna(80.0)
    rank_score = (120.0 + 1.0 - merged["FIFA排名"]) / 120.0
    merged["足球热度指数"] = (
        0.25 * base.minmax(rank_score.fillna(rank_score.median()))
        + 0.30 * base.minmax(merged["球迷比例"].fillna(merged["球迷比例"].median()))
        + 0.25 * base.minmax(base.safe_log1p(merged["球迷数量"]))
        + 0.20 * base.minmax(merged["联赛强度得分"])
    )

    if merged["历史收视热度"].isna().any():
        merged["历史收视热度"] = (
            merged["球迷数量"].fillna(merged["球迷数量"].median())
            * (0.45 + 0.55 * merged["赛程时段因子"].fillna(0.35))
        )

    merged["历史价格是否插补"] = 0
    keep_columns = [
        "country_region",
        "市场",
        "2026成交价",
        "广播商数量",
        "广播商列表",
        "合同结构",
        "人口规模",
        "总GDP",
        "人均GDP",
        "球迷数量",
        "历史收视热度",
        "足球热度指数",
        "体育媒体市场规模",
        "历史转播权价格",
        "历史价格年份",
        "历史价格是否插补",
        "是否参赛",
        "赛程时段因子",
        "主要竞购方数量",
    ]
    return merged[keep_columns].copy()


def build_extended_master_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = base.build_master_table()
    china = build_china_target_row()
    extended = pd.concat([observed, china], ignore_index=True)
    return observed, extended


def structural_indicator_matrix(df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.set_index("country_region")
    matrix = pd.DataFrame(index=indexed.index)
    matrix["历史收视热度"] = base.safe_log1p(indexed["历史收视热度"])
    matrix["人均GDP"] = base.safe_log1p(indexed["人均GDP"])
    matrix["球迷数量"] = base.safe_log1p(indexed["球迷数量"])
    matrix["足球热度指数"] = indexed["足球热度指数"]
    matrix["体育媒体市场规模"] = base.safe_log1p(indexed["体育媒体市场规模"])
    matrix["历史转播权价格"] = base.safe_log1p(indexed["历史转播权价格"])
    return matrix


def minmax_with_reference(series: pd.Series, ref_min: float, ref_max: float) -> pd.Series:
    if math.isclose(ref_min, ref_max):
        return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    return ((values - ref_min) / (ref_max - ref_min)).clip(lower=0.0, upper=1.0)


def normalize_with_observed_reference(observed_matrix: pd.DataFrame, full_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_norm = observed_matrix.apply(base.minmax, axis=0)
    full_norm = pd.DataFrame(index=full_matrix.index)
    for col in full_matrix.columns:
        ref_series = pd.to_numeric(observed_matrix[col], errors="coerce")
        full_norm[col] = minmax_with_reference(full_matrix[col], float(ref_series.min()), float(ref_series.max()))
    return observed_norm, full_norm


def rights_structure_complexity(df: pd.DataFrame) -> pd.Series:
    structure_count = (
        df["合同结构"]
        .fillna("")
        .astype(str)
        .map(lambda text: len([part for part in text.split("、") if part.strip()]))
    )
    package_bonus = df["合同结构"].fillna("").astype(str).str.contains("package", case=False).astype(int)
    return structure_count + 0.5 * package_bonus + 0.5 * pd.to_numeric(df["广播商数量"], errors="coerce").fillna(1.0)


def zscore_from_reference(values: pd.Series, ref_values: pd.Series) -> pd.Series:
    ref = pd.to_numeric(ref_values, errors="coerce")
    val = pd.to_numeric(values, errors="coerce")
    std = float(ref.std(ddof=0))
    if math.isclose(std, 0.0):
        return pd.Series(np.zeros(len(val)), index=val.index, dtype=float)
    return (val - float(ref.mean())) / std


def build_layer_profile(observed_df: pd.DataFrame, full_df: pd.DataFrame, assumptions: ChinaAssumptions) -> pd.DataFrame:
    layer_df = full_df[
        ["country_region", "市场", "总GDP", "人口规模", "体育媒体市场规模", "历史转播权价格", "主要竞购方数量", "广播商数量", "合同结构", "赛程时段因子"]
    ].copy()
    layer_df["结构复杂度"] = rights_structure_complexity(full_df)

    observed_layer = layer_df[layer_df["country_region"].isin(observed_df["country_region"])].copy()
    layer_df["Z_lnGDP"] = zscore_from_reference(np.log(layer_df["总GDP"].clip(lower=1e-6)), np.log(observed_layer["总GDP"].clip(lower=1e-6)))
    layer_df["Z_lnPOP"] = zscore_from_reference(np.log(layer_df["人口规模"].clip(lower=1e-6)), np.log(observed_layer["人口规模"].clip(lower=1e-6)))
    layer_df["Z_lnMedia"] = zscore_from_reference(
        np.log(layer_df["体育媒体市场规模"].clip(lower=1e-6)),
        np.log(observed_layer["体育媒体市场规模"].clip(lower=1e-6)),
    )
    layer_df["Z_lnHistPrice"] = zscore_from_reference(
        np.log(layer_df["历史转播权价格"].clip(lower=1e-6)),
        np.log(observed_layer["历史转播权价格"].clip(lower=1e-6)),
    )
    layer_df["Z_Bidder"] = zscore_from_reference(layer_df["主要竞购方数量"], observed_layer["主要竞购方数量"])
    layer_df["Z_RightStructure"] = zscore_from_reference(layer_df["结构复杂度"], observed_layer["结构复杂度"])

    score_cols = ["Z_lnGDP", "Z_lnPOP", "Z_lnMedia", "Z_lnHistPrice", "Z_Bidder", "Z_RightStructure"]
    layer_df["原始层级评分S"] = layer_df[score_cols].mean(axis=1)

    observed_scores = layer_df[layer_df["country_region"].isin(observed_df["country_region"])]["原始层级评分S"]
    q20, q40, q60, q80 = observed_scores.quantile([0.2, 0.4, 0.6, 0.8]).tolist()

    def score_to_layer(score: float) -> int:
        if score >= q80:
            return 5
        if score >= q60:
            return 4
        if score >= q40:
            return 3
        if score >= q20:
            return 2
        return 1

    layer_df["定量层级"] = layer_df["原始层级评分S"].map(score_to_layer)
    layer_df["规则修正层级"] = layer_df["country_region"].map(OBSERVED_LAYER_MAP)

    china_mask = layer_df["country_region"] == CHINA_MARKET
    maturity_map = observed_market_maturity_map()
    layer_df["商业化成熟度"] = layer_df["country_region"].map(maturity_map).fillna(0.50)
    layer_df.loc[china_mask, "商业化成熟度"] = assumptions.maturity_score

    layer_df.loc[china_mask, "商业化成熟度"] = assumptions.maturity_score
    china_s = float(layer_df.loc[china_mask, "赛程时段因子"].iloc[0])
    china_s_adj = float(
        layer_df.loc[china_mask, "原始层级评分S"].iloc[0]
        - assumptions.lambda1 * assumptions.hhi
        - assumptions.lambda2 * (1.0 - china_s)
        - assumptions.lambda3 * (1.0 - assumptions.maturity_score)
    )
    layer_df["修正层级评分S_adj"] = np.nan
    layer_df.loc[china_mask, "修正层级评分S_adj"] = china_s_adj

    china_initial_layer = score_to_layer(china_s_adj)
    china_layer = 4 if (china_initial_layer >= 4 or assumptions.hhi >= 0.7) else max(3, china_initial_layer)

    layer_df.loc[~china_mask, "规则修正层级"] = layer_df.loc[~china_mask, "规则修正层级"].fillna(layer_df.loc[~china_mask, "定量层级"]).astype(int)
    layer_df.loc[china_mask, "规则修正层级"] = china_layer
    layer_df["层级标签"] = layer_df["规则修正层级"].map(LAYER_LABELS)
    return layer_df


def observed_market_maturity_map() -> dict[str, float]:
    market = pd.read_csv(DATA_DIR / "07_market_structure.csv")
    market = market[market["country_region"].isin(OBSERVED_LAYER_MAP)].copy()
    maturity = market["media_broadcast_market_maturity"].map(MATURE_SCORE_MAP).fillna(0.50)
    return dict(zip(market["country_region"], maturity))


def observed_market_mechanism_map() -> dict[str, float]:
    market = pd.read_csv(DATA_DIR / "07_market_structure.csv")
    market = market[market["country_region"].isin(OBSERVED_LAYER_MAP)].copy()
    maturity = market["media_broadcast_market_maturity"].map(MATURE_SCORE_MAP).fillna(0.50)
    competition = market["competition_level"].map(COMPETITION_SCORE_MAP).fillna(0.50)
    mechanism = 0.6 * maturity + 0.4 * competition
    return dict(zip(market["country_region"], mechanism))


def compute_layer_factor(layer: int, eta: float) -> float:
    return float(np.exp(eta * (layer - 3.0)))


def china_adjusted_params(q2_params: TunedParams, assumptions: ChinaAssumptions) -> dict[str, float]:
    alpha_c = q2_params.alpha * (1.0 - assumptions.rho_alpha * assumptions.hhi)
    eta_c = q2_params.eta * (1.0 - assumptions.rho_eta * assumptions.hhi)
    omega_c = q2_params.rights_split_step * (1.0 - assumptions.rho_omega * assumptions.hhi)
    c_c = q2_params.competition_c * (1.0 - assumptions.hhi)
    return {
        "alpha_c": max(alpha_c, 0.05),
        "gamma1_c": q2_params.gamma1,
        "gamma2_c": q2_params.gamma2,
        "eta_c": max(eta_c, 0.05),
        "beta_c": q2_params.beta,
        "q": q2_params.qualified_bonus,
        "q0": q2_params.unqualified_discount,
        "c_c": max(c_c, 0.0),
        "omega_c": max(omega_c, 0.0),
    }


def pick_reference_markets(
    full_normalized: pd.DataFrame,
    weights: pd.Series,
    layer_df: pd.DataFrame,
    q2_params: TunedParams,
    assumptions: ChinaAssumptions,
) -> pd.DataFrame:
    layer_map = layer_df.set_index("country_region")["规则修正层级"].to_dict()
    china_layer = layer_map[CHINA_MARKET]
    candidate_markets = [
        market
        for market, layer in layer_map.items()
        if market != CHINA_MARKET and abs(layer - china_layer) <= 1
    ]
    raw_grey = base.grey_relation_scores(full_normalized, weights, CHINA_MARKET, candidate_markets)
    mechanism_map = observed_market_mechanism_map()

    detail = pd.DataFrame({"原始灰色关联度": raw_grey})
    detail["参考市场"] = detail.index.map(base.market_to_cn)
    detail["参考层级"] = detail.index.map(lambda name: LAYER_LABELS[layer_map[name]])
    detail["商业机制成熟度"] = detail.index.map(lambda name: mechanism_map.get(name, 0.50))
    detail["机制惩罚因子"] = np.exp(-assumptions.tau * np.abs(detail["商业机制成熟度"] - assumptions.maturity_score))
    detail["修正灰色关联度"] = detail["原始灰色关联度"] * detail["机制惩罚因子"]
    detail = detail.sort_values("修正灰色关联度", ascending=False).head(q2_params.k).copy()
    detail["修正权重"] = detail["修正灰色关联度"] / detail["修正灰色关联度"].sum()
    return detail.reset_index().rename(columns={"index": "country_region"})


def compute_china_dynamic_factor(
    china_row: pd.Series,
    adjusted_params: dict[str, float],
    assumptions: ChinaAssumptions,
) -> pd.DataFrame:
    s_c = float(china_row["赛程时段因子"])
    qual = int(china_row["是否参赛"])
    m_c = (104 / 64) ** adjusted_params["beta_c"]
    q_c = 1.0 + adjusted_params["q"] if qual == 1 else 1.0 - adjusted_params["q0"]
    t_c = 0.65 + 0.35 * s_c
    c_adj = 1.0 + adjusted_params["c_c"] * math.log(max(assumptions.effective_bidders, 1))
    b_adj = 1.0 + adjusted_params["omega_c"] * (max(assumptions.broadcaster_count, 1) - 1)
    total = m_c * q_c * t_c * c_adj * b_adj
    return pd.DataFrame(
        [
            {"因子": "内容增量因子", "数值": m_c},
            {"因子": "参赛因子", "数值": q_c},
            {"因子": "时差因子", "数值": t_c},
            {"因子": "修正竞购因子", "数值": c_adj},
            {"因子": "修正权利结构因子", "数值": b_adj},
            {"因子": "修正动态总因子", "数值": total},
        ]
    )


def compute_china_pricing(
    observed_df: pd.DataFrame,
    full_df: pd.DataFrame,
    full_normalized: pd.DataFrame,
    market_values: pd.Series,
    weights: pd.Series,
    layer_df: pd.DataFrame,
    q2_params: TunedParams,
    assumptions: ChinaAssumptions,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adjusted = china_adjusted_params(q2_params, assumptions)
    reference_df = pick_reference_markets(full_normalized, weights, layer_df, q2_params, assumptions)

    dynamic_params = base.DynamicParams(
        q2_params.beta,
        q2_params.qualified_bonus,
        q2_params.unqualified_discount,
        q2_params.competition_c,
        q2_params.rights_split_step,
    )
    observed_dynamic = base.compute_dynamic_factors(observed_df, dynamic_params).set_index("country_region")
    observed_map = observed_df.set_index("country_region")
    observed_layers = layer_df[layer_df["country_region"].isin(observed_df["country_region"])].set_index("country_region")
    china_row = full_df.set_index("country_region").loc[CHINA_MARKET]
    china_layer = int(layer_df.set_index("country_region").loc[CHINA_MARKET, "规则修正层级"])
    china_layer_factor = compute_layer_factor(china_layer, adjusted["eta_c"])
    china_dynamic_df = compute_china_dynamic_factor(china_row, adjusted, assumptions)
    china_dynamic_total = float(china_dynamic_df.loc[china_dynamic_df["因子"] == "修正动态总因子", "数值"].iloc[0])

    rows = []
    china_value = float(market_values.loc[CHINA_MARKET])
    china_gdp = float(china_row["总GDP"])
    china_pop = float(china_row["人口规模"])

    base_price = 0.0
    for ref in reference_df.itertuples(index=False):
        ref_market = ref.country_region
        ref_price = float(observed_map.loc[ref_market, "2026成交价"])
        ref_dynamic = float(observed_dynamic.loc[ref_market, "动态修正总因子"])
        ref_layer_factor = compute_layer_factor(int(observed_layers.loc[ref_market, "规则修正层级"]), q2_params.eta)
        ref_base = ref_price / (ref_dynamic * ref_layer_factor)
        ref_gdp = float(observed_map.loc[ref_market, "总GDP"])
        ref_pop = float(observed_map.loc[ref_market, "人口规模"])
        ref_value = float(market_values.loc[ref_market])

        gdp_log = math.log(max(china_gdp, 1e-6) / max(ref_gdp, 1e-6))
        pop_log = math.log(max(china_pop, 1e-6) / max(ref_pop, 1e-6))
        scale_factor = max(0.1, 1.0 + adjusted["gamma1_c"] * gdp_log + adjusted["gamma2_c"] * pop_log)
        quality_ratio = max(china_value, 1e-6) / max(ref_value, 1e-6)
        quality_factor = quality_ratio ** adjusted["alpha_c"]
        china_base_from_ref = ref_base * scale_factor * quality_factor
        weighted_contribution = float(ref.修正权重) * china_base_from_ref
        base_price += weighted_contribution

        rows.append(
            {
                "参考市场": ref.参考市场,
                "参考层级": ref.参考层级,
                "原始灰色关联度": float(ref.原始灰色关联度),
                "商业机制成熟度": float(ref.商业机制成熟度),
                "机制惩罚因子": float(ref.机制惩罚因子),
                "修正灰色关联度": float(ref.修正灰色关联度),
                "修正权重": float(ref.修正权重),
                "参考市场基础价格": ref_base,
                "GDP对数差异": gdp_log,
                "人口对数差异": pop_log,
                "体量平滑乘数": scale_factor,
                "结构价值比": quality_ratio,
                "结构弹性修正项": quality_factor,
                "推得中国基础价格": china_base_from_ref,
                "加权贡献价格": weighted_contribution,
            }
        )

    mid_price = base_price * china_dynamic_total * china_layer_factor
    low_price = mid_price * (1.0 - assumptions.interval_delta1)
    high_price = mid_price * (1.0 + assumptions.interval_delta2)

    summary = pd.DataFrame(
        [
            {"项目": "中国内地基础价格", "数值": base_price},
            {"项目": "中国内地修正动态总因子", "数值": china_dynamic_total},
            {"项目": "中国内地层级因子", "数值": china_layer_factor},
            {"项目": "中国内地保守报价", "数值": low_price},
            {"项目": "中国内地基准合理报价", "数值": mid_price},
            {"项目": "中国内地进取报价", "数值": high_price},
        ]
    )
    return pd.DataFrame(rows), china_dynamic_df, summary


def build_comparison_and_decomposition(
    pricing_summary: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    adjusted_params: dict[str, float],
    assumptions: ChinaAssumptions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_price = float(pricing_summary.loc[pricing_summary["项目"] == "中国内地基础价格", "数值"].iloc[0])
    mid_price = float(pricing_summary.loc[pricing_summary["项目"] == "中国内地基准合理报价", "数值"].iloc[0])
    low_price = float(pricing_summary.loc[pricing_summary["项目"] == "中国内地保守报价", "数值"].iloc[0])
    high_price = float(pricing_summary.loc[pricing_summary["项目"] == "中国内地进取报价", "数值"].iloc[0])

    beta_f = min(adjusted_params["beta_c"] + assumptions.fifa_beta_extra, 0.85)
    premium_expand = base_price * (((104 / 64) ** beta_f) - ((104 / 64) ** adjusted_params["beta_c"]))
    premium_anchor = mid_price * assumptions.fifa_anchor_rate
    premium_total = premium_expand + premium_anchor

    monopoly_discount = mid_price * assumptions.lambda_m * assumptions.hhi
    timezone_factor = float(dynamic_df.loc[dynamic_df["因子"] == "时差因子", "数值"].iloc[0])
    timezone_discount = mid_price * ((1.0 / timezone_factor) - 1.0)
    other_residual = assumptions.fifa_initial_ask - assumptions.final_deal_price - premium_total - monopoly_discount - timezone_discount

    denominator = assumptions.fifa_initial_ask - assumptions.cctv_counter_offer
    theta = (assumptions.final_deal_price - assumptions.cctv_counter_offer) / denominator if denominator else np.nan

    compare_df = pd.DataFrame(
        [
            {"价格类型": "模型保守报价", "价格(百万美元)": low_price},
            {"价格类型": "模型基准合理报价", "价格(百万美元)": mid_price},
            {"价格类型": "模型进取报价", "价格(百万美元)": high_price},
            {"价格类型": "FIFA最初报价", "价格(百万美元)": assumptions.fifa_initial_ask},
            {"价格类型": "央视还价", "价格(百万美元)": assumptions.cctv_counter_offer},
            {"价格类型": "最终成交价", "价格(百万美元)": assumptions.final_deal_price},
        ]
    )
    compare_df["相对模型基准偏离率"] = (compare_df["价格(百万美元)"] - mid_price) / mid_price

    decomposition_df = pd.DataFrame(
        [
            {"分解项": "FIFA扩军溢价", "金额(百万美元)": premium_expand},
            {"分解项": "FIFA心理锚定溢价", "金额(百万美元)": premium_anchor},
            {"分解项": "FIFA总溢价", "金额(百万美元)": premium_total},
            {"分解项": "央视买方垄断压价", "金额(百万美元)": monopoly_discount},
            {"分解项": "北美时差折价", "金额(百万美元)": timezone_discount},
            {"分解项": "其他残差", "金额(百万美元)": other_residual},
            {"分解项": "谈判权重theta", "金额(百万美元)": theta},
        ]
    )
    return compare_df, decomposition_df


def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


def save_markdown(content: str, filename: str) -> None:
    (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")


def plot_reference_weights(reference_df: pd.DataFrame) -> None:
    data = reference_df.sort_values("修正权重", ascending=True).reset_index(drop=True)
    width, height = 1200, 110 + 65 * len(data)
    image, draw = base.make_canvas(width, height)
    title_font = base.get_font(28)
    label_font = base.get_font(20)
    value_font = base.get_font(16)
    left, right = 260, 1040
    top = 75
    max_value = float(data["修正权重"].max())
    draw.text((width // 2 - 170, 20), "第三问参考市场修正权重图", fill="black", font=title_font)
    for idx, row in data.iterrows():
        y = top + idx * 56
        bar_width = int((row["修正权重"] / max_value) * (right - left)) if max_value > 0 else 0
        draw.text((30, y), str(row["参考市场"]), fill="black", font=label_font)
        draw.rectangle([left, y + 6, left + bar_width, y + 32], fill="#2563eb")
        draw.text((left + bar_width + 12, y + 5), f"{row['修正权重']:.4f}", fill="black", font=value_font)
    base.save_image(image, "第三问参考市场修正权重图.png")


def plot_pricing_comparison(compare_df: pd.DataFrame) -> None:
    data = compare_df.copy().reset_index(drop=True)
    width, height = 1350, 110 + 65 * len(data)
    image, draw = base.make_canvas(width, height)
    title_font = base.get_font(28)
    label_font = base.get_font(20)
    value_font = base.get_font(16)
    left, right = 290, 1130
    top = 80
    max_value = float(data["价格(百万美元)"].max())
    colors = ["#0f766e", "#0f766e", "#0f766e", "#dc2626", "#f59e0b", "#2563eb"]
    draw.text((width // 2 - 170, 20), "第三问中国内地价格对比图", fill="black", font=title_font)
    for idx, row in data.iterrows():
        y = top + idx * 56
        bar_width = int((row["价格(百万美元)"] / max_value) * (right - left)) if max_value > 0 else 0
        draw.text((30, y), str(row["价格类型"]), fill="black", font=label_font)
        draw.rectangle([left, y + 6, left + bar_width, y + 32], fill=colors[idx % len(colors)])
        draw.text((left + bar_width + 12, y + 5), f"{row['价格(百万美元)']:.2f}", fill="black", font=value_font)
    base.save_image(image, "第三问中国内地价格对比图.png")


def plot_decomposition(decomposition_df: pd.DataFrame) -> None:
    data = decomposition_df[decomposition_df["分解项"] != "谈判权重theta"].copy()
    width, height = 1300, 110 + 65 * len(data)
    image, draw = base.make_canvas(width, height)
    title_font = base.get_font(28)
    label_font = base.get_font(20)
    value_font = base.get_font(16)
    left, center = 610, 620
    top = 80
    max_abs = float(np.abs(data["金额(百万美元)"]).max())
    draw.text((width // 2 - 200, 20), "第三问价格差异分解图", fill="black", font=title_font)
    draw.line([center, top - 10, center, height - 20], fill="black", width=2)
    for idx, row in data.reset_index(drop=True).iterrows():
        y = top + idx * 56
        value = float(row["金额(百万美元)"])
        length = int((abs(value) / max_abs) * 420) if max_abs > 0 else 0
        color = "#dc2626" if value >= 0 else "#2563eb"
        draw.text((30, y), str(row["分解项"]), fill="black", font=label_font)
        if value >= 0:
            draw.rectangle([center, y + 6, center + length, y + 32], fill=color)
            draw.text((center + length + 12, y + 5), f"{value:.2f}", fill="black", font=value_font)
        else:
            draw.rectangle([center - length, y + 6, center, y + 32], fill=color)
            draw.text((center - length - 80, y + 5), f"{value:.2f}", fill="black", font=value_font)
    base.save_image(image, "第三问价格差异分解图.png")


def build_result_markdown(
    q2_params: TunedParams,
    assumptions: ChinaAssumptions,
    adjusted_params: dict[str, float],
    layer_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    pricing_summary: pd.DataFrame,
    compare_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
) -> str:
    china_layer = layer_df[layer_df["country_region"] == CHINA_MARKET].copy()
    theta = float(decomposition_df.loc[decomposition_df["分解项"] == "谈判权重theta", "金额(百万美元)"].iloc[0])
    return "\n".join(
        [
            "# 第三问中国内地转播权报价结果",
            "",
            "## 第二问沿用参数",
            base.dataframe_to_markdown(
                pd.DataFrame(
                    [
                        {"参数": "参考市场数量k", "取值": q2_params.k},
                        {"参数": "结构价值弹性alpha*", "取值": q2_params.alpha},
                        {"参数": "GDP敏感系数gamma1*", "取值": q2_params.gamma1},
                        {"参数": "人口敏感系数gamma2*", "取值": q2_params.gamma2},
                        {"参数": "层级溢价强度eta*", "取值": q2_params.eta},
                        {"参数": "内容增量弹性beta*", "取值": q2_params.beta},
                        {"参数": "参赛上浮系数q", "取值": q2_params.qualified_bonus},
                        {"参数": "未参赛折扣系数q0", "取值": q2_params.unqualified_discount},
                        {"参数": "竞购系数c*", "取值": q2_params.competition_c},
                        {"参数": "权利结构系数omega*", "取值": q2_params.rights_split_step},
                    ]
                ),
                digits=4,
            ),
            "",
            "## 中国内地修正设定",
            base.dataframe_to_markdown(
                pd.DataFrame(
                    [
                        {"项目": "买方集中度HHI_C", "数值": assumptions.hhi},
                        {"项目": "商业化成熟度m_C", "数值": assumptions.maturity_score},
                        {"项目": "有效竞购方数量n_C", "数值": assumptions.effective_bidders},
                        {"项目": "修正结构价值弹性alpha_C", "数值": adjusted_params["alpha_c"]},
                        {"项目": "修正层级溢价强度eta_C", "数值": adjusted_params["eta_c"]},
                        {"项目": "修正竞购系数c_C", "数值": adjusted_params["c_c"]},
                        {"项目": "修正权利结构系数omega_C", "数值": adjusted_params["omega_c"]},
                    ]
                ),
                digits=4,
            ),
            "",
            "## 中国内地层级判定",
            base.dataframe_to_markdown(
                china_layer[
                    [
                        "市场",
                        "原始层级评分S",
                        "修正层级评分S_adj",
                        "定量层级",
                        "规则修正层级",
                        "层级标签",
                    ]
                ],
                digits=4,
            ),
            "",
            "## 参考市场修正权重",
            base.dataframe_to_markdown(
                reference_df[
                    [
                        "参考市场",
                        "参考层级",
                        "原始灰色关联度",
                        "商业机制成熟度",
                        "机制惩罚因子",
                        "修正灰色关联度",
                        "修正权重",
                    ]
                ],
                digits=4,
            ),
            "",
            "## 动态修正因子",
            base.dataframe_to_markdown(dynamic_df, digits=4),
            "",
            "## 报价结果",
            base.dataframe_to_markdown(pricing_summary, digits=4),
            "",
            "## 现实价格对比",
            base.dataframe_to_markdown(compare_df, digits=4),
            "",
            "## 差异分解",
            base.dataframe_to_markdown(decomposition_df, digits=4),
            "",
            "## 说明",
            "1. FIFA最初报价采用 2.5 亿到 3 亿美元区间的中点 2.75 亿美元作为建模输入。",
            "2. 央视还价在模型中作为谈判锚点，单独取 0.55 亿美元，避免与最终成交价完全重合。",
            "3. 最终成交价按 0.6 亿美元的公开报道口径换算为 2026 单届等价价格。",
            "4. 报价区间的 delta1 = 0.22、delta2 = 0.27 是对第二问留一交叉验证结果的圆整化处理，便于论文表述。",
            f"5. 谈判权重 theta = {theta:.4f}，越接近 0 表示结果越接近央视还价，越接近 1 表示越接近 FIFA 报价。",
            "",
        ]
    )


def main() -> None:
    ensure_output_dir()
    assumptions = ChinaAssumptions()
    q2_params = read_best_q2_params()
    observed_df, full_df = build_extended_master_table()

    observed_matrix = structural_indicator_matrix(observed_df)
    full_matrix = structural_indicator_matrix(full_df)
    observed_normalized, full_normalized = normalize_with_observed_reference(observed_matrix, full_matrix)

    ahp_matrix, subjective_weights, ahp_summary = base.build_ahp_result(list(observed_matrix.columns))
    objective_weights = base.entropy_weight(observed_normalized)
    weight_df = base.combine_weights(subjective_weights, objective_weights)
    weights = weight_df["组合权重"]
    market_values = base.topsis_score(full_normalized, weights)

    layer_df = build_layer_profile(observed_df, full_df, assumptions)
    adjusted_params = china_adjusted_params(q2_params, assumptions)
    reference_df, dynamic_df, pricing_summary = compute_china_pricing(
        observed_df=observed_df,
        full_df=full_df,
        full_normalized=full_normalized,
        market_values=market_values,
        weights=weights,
        layer_df=layer_df,
        q2_params=q2_params,
        assumptions=assumptions,
    )
    compare_df, decomposition_df = build_comparison_and_decomposition(pricing_summary, dynamic_df, adjusted_params, assumptions)

    china_input_df = full_df[full_df["country_region"] == CHINA_MARKET].copy()
    china_input_df["版权商业化成熟度"] = assumptions.maturity_score
    china_input_df["买方集中度HHI_C"] = assumptions.hhi
    china_input_df["有效竞购方数量n_C"] = assumptions.effective_bidders
    china_input_export = china_input_df.rename(columns={"country_region": "市场代码"})

    market_value_export = pd.DataFrame(
        {
            "市场": [base.market_to_cn(name) if name != CHINA_MARKET else "中国内地" for name in market_values.index],
            "结构TOPSIS市场价值指数": market_values.values,
        }
    )
    normalized_export = full_normalized.reset_index().rename(columns={"index": "市场代码", "country_region": "市场代码"})
    normalized_export.insert(1, "市场", normalized_export["市场代码"].map(lambda name: base.market_to_cn(name) if name != CHINA_MARKET else "中国内地"))
    layer_export = layer_df.rename(columns={"country_region": "市场代码"}).copy()

    save_csv(china_input_export, "第三问中国内地输入数据表.csv")
    save_csv(ahp_matrix.reset_index().rename(columns={"index": "指标"}), "第三问AHP判断矩阵.csv")
    save_csv(ahp_summary, "第三问AHP一致性检验结果.csv")
    save_csv(weight_df.reset_index().rename(columns={"index": "指标"}), "第三问结构TOPSIS组合赋权结果.csv")
    save_csv(normalized_export, "第三问结构TOPSIS标准化矩阵.csv")
    save_csv(market_value_export, "第三问结构TOPSIS市场价值指数.csv")
    save_csv(layer_export, "第三问中国内地层级判定表.csv")
    save_csv(reference_df, "第三问中国内地参考市场权重表.csv")
    save_csv(dynamic_df, "第三问中国内地动态修正因子表.csv")
    save_csv(pricing_summary, "第三问中国内地报价结果表.csv")
    save_csv(compare_df, "第三问中国内地现实价格对比表.csv")
    save_csv(decomposition_df, "第三问中国内地价格差异分解表.csv")

    plot_reference_weights(reference_df)
    plot_pricing_comparison(compare_df)
    plot_decomposition(decomposition_df)

    result_md = build_result_markdown(
        q2_params=q2_params,
        assumptions=assumptions,
        adjusted_params=adjusted_params,
        layer_df=layer_df,
        reference_df=reference_df,
        dynamic_df=dynamic_df,
        pricing_summary=pricing_summary,
        compare_df=compare_df,
        decomposition_df=decomposition_df,
    )
    save_markdown(result_md, "第三问模型结果说明.md")


if __name__ == "__main__":
    main()
