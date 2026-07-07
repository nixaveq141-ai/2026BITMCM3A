import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "q2" / "run_q2_analysis.py"
OUTPUT_DIR = ROOT / "q2" / "outputs_layered_market"

ALPHA_GRID = np.round(np.arange(0.4, 1.41, 0.2), 2)
GAMMA1_GRID = np.round(np.arange(0.05, 0.36, 0.05), 2)
GAMMA2_GRID = np.round(np.arange(0.05, 0.26, 0.05), 2)
ETA_GRID = [0.15, 0.20, 0.25, 0.30]

LAYER_LABELS = {5: "S", 4: "A", 3: "B", 2: "C", 1: "D"}
MANUAL_LAYER_MAP = {
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


def load_base_module():
    spec = importlib.util.spec_from_file_location("q2_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()
base.OUTPUT_DIR = OUTPUT_DIR
base.COMPETITION_C_GRID = [0.04, 0.08, 0.12, 0.16]


@dataclass
class LayeredModelConfig:
    alpha: float
    gamma1: float
    gamma2: float
    eta: float
    k: int
    dynamic: object


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_ratio(numerator: float, denominator: float) -> float:
    return max(float(numerator), 1e-6) / max(float(denominator), 1e-6)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = float(values.std(ddof=0))
    if np.isclose(std, 0.0):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - float(values.mean())) / std


def right_structure_complexity(df: pd.DataFrame) -> pd.Series:
    structure_count = (
        df["合同结构"]
        .fillna("")
        .astype(str)
        .map(lambda text: len([part for part in text.split("、") if part.strip()]))
    )
    package_bonus = df["合同结构"].fillna("").astype(str).str.contains("package", case=False).astype(int)
    return structure_count + 0.5 * package_bonus + 0.5 * pd.to_numeric(df["广播商数量"], errors="coerce").fillna(1.0)


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


def build_layer_profile(df: pd.DataFrame) -> pd.DataFrame:
    layer_df = df[["country_region", "市场", "总GDP", "人口规模", "体育媒体市场规模", "历史转播权价格", "主要竞购方数量", "广播商数量", "合同结构"]].copy()
    layer_df["结构复杂度"] = right_structure_complexity(df)
    layer_df["Z_lnGDP"] = zscore(np.log(layer_df["总GDP"].clip(lower=1e-6)))
    layer_df["Z_lnPOP"] = zscore(np.log(layer_df["人口规模"].clip(lower=1e-6)))
    layer_df["Z_lnMedia"] = zscore(np.log(layer_df["体育媒体市场规模"].clip(lower=1e-6)))
    layer_df["Z_lnHistPrice"] = zscore(np.log(layer_df["历史转播权价格"].clip(lower=1e-6)))
    layer_df["Z_Bidder"] = zscore(layer_df["主要竞购方数量"])
    layer_df["Z_RightStructure"] = zscore(layer_df["结构复杂度"])

    score_cols = ["Z_lnGDP", "Z_lnPOP", "Z_lnMedia", "Z_lnHistPrice", "Z_Bidder", "Z_RightStructure"]
    layer_df["层级评分S"] = layer_df[score_cols].mean(axis=1)

    q20, q40, q60, q80 = layer_df["层级评分S"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()

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

    layer_df["定量层级"] = layer_df["层级评分S"].map(score_to_layer)
    layer_df["规则修正层级"] = layer_df["country_region"].map(MANUAL_LAYER_MAP).fillna(layer_df["定量层级"]).astype(int)
    layer_df["层级标签"] = layer_df["规则修正层级"].map(LAYER_LABELS)
    return layer_df


def layer_factor(layer: pd.Series, eta: float) -> pd.Series:
    return np.exp(eta * (pd.to_numeric(layer, errors="coerce") - 3.0))


def eligible_reference_markets(layer_map: dict[str, int], target_market: str, all_markets: list[str]) -> list[str]:
    target_layer = layer_map[target_market]
    refs = [market for market in all_markets if market != target_market and abs(layer_map[market] - target_layer) <= 1]
    if refs:
        return refs
    refs = [market for market in all_markets if market != target_market]
    return refs


def prepare_prediction_cache(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    market_values: pd.Series,
    weights: pd.Series,
    layer_map: dict[str, int],
) -> dict[tuple[str, int], dict[str, object]]:
    cache: dict[tuple[str, int], dict[str, object]] = {}
    markets = df["country_region"].tolist()
    market_index = {market: idx for idx, market in enumerate(markets)}
    df_map = df.set_index("country_region")
    for target_market in markets:
        ref_markets = eligible_reference_markets(layer_map, target_market, markets)
        grey_scores = base.grey_relation_scores(normalized, weights, target_market, ref_markets)
        target_gdp = float(df_map.loc[target_market, "总GDP"])
        target_population = float(df_map.loc[target_market, "人口规模"])
        target_value = float(market_values.loc[target_market])
        for k in base.K_GRID:
            selected = grey_scores.head(min(k, len(grey_scores)))
            ref_list = selected.index.tolist()
            grey_weights = (selected / selected.sum()).to_numpy(dtype=float)
            ref_gdp = df_map.loc[ref_list, "总GDP"].to_numpy(dtype=float)
            ref_population = df_map.loc[ref_list, "人口规模"].to_numpy(dtype=float)
            ref_value = market_values.loc[ref_list].to_numpy(dtype=float)
            gdp_ratio = np.array([safe_ratio(target_gdp, value) for value in ref_gdp], dtype=float)
            population_ratio = np.array([safe_ratio(target_population, value) for value in ref_population], dtype=float)
            quality_ratio = np.array([safe_ratio(target_value, value) for value in ref_value], dtype=float)
            cache[(target_market, k)] = {
                "target_idx": market_index[target_market],
                "ref_indices": np.array([market_index[ref] for ref in ref_list], dtype=int),
                "ref_markets": ref_list,
                "grey_weights": grey_weights,
                "quality_ratio": quality_ratio,
                "gdp_log_ratio": np.log(gdp_ratio),
                "population_log_ratio": np.log(population_ratio),
            }
    return cache


def predict_one_market_layered(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    market_values: pd.Series,
    weights: pd.Series,
    dynamic: pd.DataFrame,
    layer_df: pd.DataFrame,
    target_market: str,
    alpha: float,
    gamma1: float,
    gamma2: float,
    eta: float,
    k: int,
) -> tuple[float, float, pd.DataFrame]:
    layer_map = layer_df.set_index("country_region")["规则修正层级"].to_dict()
    ref_markets = eligible_reference_markets(layer_map, target_market, df["country_region"].tolist())
    grey_scores = base.grey_relation_scores(normalized, weights, target_market, ref_markets)
    selected = grey_scores.head(min(k, len(grey_scores)))

    dynamic_map = dynamic.set_index("country_region")
    df_map = df.set_index("country_region")
    layer_factor_map = layer_factor(layer_df.set_index("country_region")["规则修正层级"], eta)

    target_gdp = float(df_map.loc[target_market, "总GDP"])
    target_population = float(df_map.loc[target_market, "人口规模"])
    target_value = float(market_values.loc[target_market])
    target_dynamic = float(dynamic_map.loc[target_market, "动态修正总因子"])
    target_layer_factor = float(layer_factor_map.loc[target_market])

    detail = pd.DataFrame({"灰色关联度": selected})
    detail["灰色权重"] = detail["灰色关联度"] / detail["灰色关联度"].sum()
    detail["参考市场价值指数"] = market_values.loc[detail.index]
    detail["参考市场层级"] = layer_df.set_index("country_region").loc[detail.index, "层级标签"]

    ref_actual_price = df_map.loc[detail.index, "2026成交价"]
    ref_dynamic = dynamic_map.loc[detail.index, "动态修正总因子"]
    ref_layer_factor = layer_factor_map.loc[detail.index]
    ref_base_price = ref_actual_price / (ref_dynamic * ref_layer_factor)

    ref_gdp = df_map.loc[detail.index, "总GDP"]
    ref_population = df_map.loc[detail.index, "人口规模"]
    gdp_log_ratio = np.log(ref_gdp.map(lambda value: safe_ratio(target_gdp, value)))
    population_log_ratio = np.log(ref_population.map(lambda value: safe_ratio(target_population, value)))
    scale_multiplier = np.maximum(0.1, 1.0 + gamma1 * gdp_log_ratio + gamma2 * population_log_ratio)
    quality_ratio = detail["参考市场价值指数"].map(lambda value: safe_ratio(target_value, value))

    detail["参考市场基础价格"] = ref_base_price
    detail["GDP对数差值"] = gdp_log_ratio
    detail["人口对数差值"] = population_log_ratio
    detail["体量平滑乘数"] = scale_multiplier
    detail["结构价值比"] = quality_ratio
    detail["结构弹性项"] = np.power(quality_ratio, alpha)
    detail["校准后目标基础价格"] = ref_base_price * detail["体量平滑乘数"] * detail["结构弹性项"]

    base_price = float((detail["灰色权重"] * detail["校准后目标基础价格"]).sum())
    final_price = base_price * target_dynamic * target_layer_factor

    detail = detail.reset_index().rename(columns={"index": "country_region"})
    detail["参考市场"] = detail["country_region"].map(base.market_to_cn)
    return final_price, base_price, detail


def run_loocv_layered(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    market_values: pd.Series,
    weights: pd.Series,
    dynamic: pd.DataFrame,
    layer_df: pd.DataFrame,
    config: LayeredModelConfig,
) -> pd.DataFrame:
    rows = []
    df_map = df.set_index("country_region")
    dynamic_map = dynamic.set_index("country_region")
    layer_factor_map = layer_factor(layer_df.set_index("country_region")["规则修正层级"], config.eta)
    layer_label_map = layer_df.set_index("country_region")["层级标签"].to_dict()
    for market in df["country_region"]:
        pred_price, base_price, detail = predict_one_market_layered(
            df=df,
            normalized=normalized,
            market_values=market_values,
            weights=weights,
            dynamic=dynamic,
            layer_df=layer_df,
            target_market=market,
            alpha=config.alpha,
            gamma1=config.gamma1,
            gamma2=config.gamma2,
            eta=config.eta,
            k=config.k,
        )
        actual_price = float(df_map.loc[market, "2026成交价"])
        actual_base = actual_price / (
            float(dynamic_map.loc[market, "动态修正总因子"]) * float(layer_factor_map.loc[market])
        )
        abs_error = abs(pred_price - actual_price)
        rel_error = abs_error / actual_price
        rows.append(
            {
                "市场": base.market_to_cn(market),
                "市场层级": layer_label_map[market],
                "实际成交价": actual_price,
                "预测成交价": pred_price,
                "实际基准价格": actual_base,
                "预测基准价格": base_price,
                "绝对误差": abs_error,
                "相对误差": rel_error,
                "参考市场数量": config.k,
                "结构价值弹性系数alpha": config.alpha,
                "GDP对数敏感系数gamma1": config.gamma1,
                "人口对数敏感系数gamma2": config.gamma2,
                "层级溢价强度eta": config.eta,
                "参考市场": "、".join(detail["参考市场"].tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("相对误差").reset_index(drop=True)


def tune_parameters_layered(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    market_values: pd.Series,
    weights: pd.Series,
    layer_df: pd.DataFrame,
) -> tuple[LayeredModelConfig, pd.DataFrame, pd.DataFrame]:
    markets = df["country_region"].tolist()
    actual_prices = df["2026成交价"].to_numpy(dtype=float)
    layer_map = layer_df.set_index("country_region")["规则修正层级"].to_dict()
    prediction_cache = prepare_prediction_cache(df, normalized, market_values, weights, layer_map)
    rows = []
    best_config = None
    best_key = None
    for beta in base.BETA_GRID:
        for q in base.QUALIFIED_BONUS_GRID:
            for q0 in base.UNQUALIFIED_DISCOUNT_GRID:
                for c in base.COMPETITION_C_GRID:
                    for rights_step in base.RIGHTS_SPLIT_STEP_GRID:
                        dynamic_params = base.DynamicParams(beta, q, q0, c, rights_step)
                        dynamic = base.compute_dynamic_factors(df, dynamic_params)
                        dynamic_values = dynamic["动态修正总因子"].to_numpy(dtype=float)
                        for eta in ETA_GRID:
                            layer_factors = layer_factor(layer_df["规则修正层级"], eta).to_numpy(dtype=float)
                            ref_base_prices_by_market = actual_prices / (dynamic_values * layer_factors)
                            for k in base.K_GRID:
                                for alpha in ALPHA_GRID:
                                    for gamma1 in GAMMA1_GRID:
                                        for gamma2 in GAMMA2_GRID:
                                            rel_errors = []
                                            for market in markets:
                                                cached = prediction_cache[(market, k)]
                                                ref_base_prices = ref_base_prices_by_market[cached["ref_indices"]]
                                                scale_multiplier = np.maximum(
                                                    0.1,
                                                    1.0
                                                    + gamma1 * cached["gdp_log_ratio"]
                                                    + gamma2 * cached["population_log_ratio"],
                                                )
                                                base_price = float(
                                                    np.sum(
                                                        cached["grey_weights"]
                                                        * ref_base_prices
                                                        * scale_multiplier
                                                        * np.power(cached["quality_ratio"], alpha)
                                                    )
                                                )
                                                target_idx = cached["target_idx"]
                                                pred_price = base_price * dynamic_values[target_idx] * layer_factors[target_idx]
                                                actual_price = actual_prices[target_idx]
                                                rel_errors.append(abs(pred_price - actual_price) / actual_price)
                                            rel_errors_array = np.array(rel_errors, dtype=float)
                                            mean_error = float(rel_errors_array.mean())
                                            median_error = float(np.median(rel_errors_array))
                                            max_error = float(rel_errors_array.max())
                                            rows.append(
                                                {
                                                    "参考市场数量": k,
                                                    "结构价值弹性系数alpha": alpha,
                                                    "GDP对数敏感系数gamma1": gamma1,
                                                    "人口对数敏感系数gamma2": gamma2,
                                                    "层级溢价强度eta": eta,
                                                    "beta": beta,
                                                    "参赛上浮系数": q,
                                                    "未参赛折扣系数": q0,
                                                    "竞购系数": c,
                                                    "结构系数": rights_step,
                                                    "平均相对误差": mean_error,
                                                    "中位相对误差": median_error,
                                                    "最大相对误差": max_error,
                                                }
                                            )
                                            key = (mean_error, median_error, max_error)
                                            if best_key is None or key < best_key:
                                                best_key = key
                                                best_config = LayeredModelConfig(
                                                    alpha=float(alpha),
                                                    gamma1=float(gamma1),
                                                    gamma2=float(gamma2),
                                                    eta=float(eta),
                                                    k=int(k),
                                                    dynamic=dynamic_params,
                                                )
    tuning_df = pd.DataFrame(rows).sort_values(["平均相对误差", "中位相对误差", "最大相对误差"]).reset_index(drop=True)
    best_eta = float(tuning_df.iloc[0]["层级溢价强度eta"])
    best_gamma2 = float(tuning_df.iloc[0]["人口对数敏感系数gamma2"])
    heatmap_df = (
        tuning_df[
            (tuning_df["层级溢价强度eta"] == best_eta)
            & (tuning_df["人口对数敏感系数gamma2"] == best_gamma2)
        ]
        .groupby(["GDP对数敏感系数gamma1", "结构价值弹性系数alpha"], as_index=False)["平均相对误差"]
        .mean()
        .sort_values(["GDP对数敏感系数gamma1", "结构价值弹性系数alpha"])
        .reset_index(drop=True)
    )
    return best_config, tuning_df, heatmap_df


def plot_market_value_rank(market_values: pd.Series, price_map: pd.Series) -> None:
    data = pd.DataFrame(
        {
            "市场": [base.market_to_cn(name) for name in market_values.index],
            "结构价值指数": market_values.values,
            "2026成交价": price_map.loc[market_values.index].values,
        }
    ).sort_values("结构价值指数", ascending=True)
    width, height = 1300, 100 + 65 * len(data)
    image, draw = base.make_canvas(width, height)
    title_font = base.get_font(28)
    label_font = base.get_font(20)
    value_font = base.get_font(16)
    left, right = 240, 1080
    top = 70
    max_value = float(data["结构价值指数"].max())
    draw.text((width // 2 - 180, 20), "结构TOPSIS市场价值指数排序图", fill="black", font=title_font)
    for idx, row in data.reset_index(drop=True).iterrows():
        y = top + idx * 55
        bar_width = int((row["结构价值指数"] / max_value) * (right - left)) if max_value > 0 else 0
        draw.text((30, y), str(row["市场"]), fill="black", font=label_font)
        draw.rectangle([left, y + 4, left + bar_width, y + 30], fill="#2563eb")
        draw.text((left + bar_width + 12, y + 1), f"{row['结构价值指数']:.4f}", fill="black", font=value_font)
    base.save_image(image, "结构TOPSIS市场价值指数排序图.png")


def plot_heatmap(heatmap_df: pd.DataFrame) -> None:
    pivot = heatmap_df.pivot(index="GDP对数敏感系数gamma1", columns="结构价值弹性系数alpha", values="平均相对误差")
    gamma_values = list(pivot.index)
    alpha_values = list(pivot.columns)
    cell_w, cell_h = 130, 60
    left, top = 220, 120
    width = left + cell_w * len(alpha_values) + 80
    height = top + cell_h * len(gamma_values) + 100
    image, draw = base.make_canvas(width, height)
    title_font = base.get_font(28)
    axis_font = base.get_font(18)
    value_font = base.get_font(16)

    draw.text((width // 2 - 260, 24), "参数寻优热力图（固定最优eta与gamma2）", fill="black", font=title_font)
    values = pivot.to_numpy(dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    for j, alpha in enumerate(alpha_values):
        x = left + j * cell_w
        draw.text((x + 25, top - 40), f"a={alpha:.1f}", fill="black", font=axis_font)
    for i, gamma1 in enumerate(gamma_values):
        y = top + i * cell_h
        draw.text((40, y + 15), f"g1={gamma1:.2f}", fill="black", font=axis_font)
        for j, alpha in enumerate(alpha_values):
            value = float(pivot.iloc[i, j])
            ratio = 0.0 if np.isclose(vmax, vmin) else (value - vmin) / (vmax - vmin)
            red = int(230 * ratio + 20)
            green = int(180 * (1 - ratio) + 40)
            blue = 90
            x = left + j * cell_w
            draw.rectangle([x, y, x + cell_w - 8, y + cell_h - 8], fill=(red, green, blue), outline="black")
            draw.text((x + 18, y + 18), f"{value:.3f}", fill="white", font=value_font)
    base.save_image(image, "参数寻优平均相对误差热力图_分层版.png")


def build_summary_markdown(
    config: LayeredModelConfig,
    loocv_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    interval_sensitivity_df: pd.DataFrame,
) -> str:
    summary_df = pd.DataFrame(
        [
            {"指标": "最优参考市场数量k", "结果": config.k},
            {"指标": "最优结构价值弹性系数alpha", "结果": config.alpha},
            {"指标": "最优GDP对数敏感系数gamma1", "结果": config.gamma1},
            {"指标": "最优人口对数敏感系数gamma2", "结果": config.gamma2},
            {"指标": "最优层级溢价强度eta", "结果": config.eta},
            {"指标": "最优beta", "结果": config.dynamic.beta},
            {"指标": "最优参赛上浮系数q", "结果": config.dynamic.qualified_bonus},
            {"指标": "最优未参赛折扣系数q0", "结果": config.dynamic.unqualified_discount},
            {"指标": "最优竞购系数c", "结果": config.dynamic.competition_c},
            {"指标": "最优结构系数", "结果": config.dynamic.rights_split_step},
            {"指标": "平均相对误差", "结果": loocv_df["相对误差"].mean()},
            {"指标": "中位相对误差", "结果": loocv_df["相对误差"].median()},
            {"指标": "最大相对误差", "结果": loocv_df["相对误差"].max()},
        ]
    )
    return "\n".join(
        [
            "# 第二问分层改进版模型测试结果",
            "",
            "## 改进说明",
            "本版本在对数体量平滑模型基础上，新增市场层级变量、邻层参考市场约束与层级溢价因子，用于刻画不同市场的版权商业化能力差异。",
            "",
            "## 核心结果",
            base.dataframe_to_markdown(summary_df, digits=4),
            "",
            "## 留一交叉验证结果",
            base.dataframe_to_markdown(
                loocv_df[
                    [
                        "市场",
                        "市场层级",
                        "实际成交价",
                        "预测成交价",
                        "相对误差",
                        "结构价值弹性系数alpha",
                        "GDP对数敏感系数gamma1",
                        "人口对数敏感系数gamma2",
                        "层级溢价强度eta",
                        "参考市场",
                    ]
                ],
                digits=4,
            ),
            "",
            "## 参数寻优前十组",
            base.dataframe_to_markdown(tuning_df.head(10), digits=4),
            "",
            "## 报价区间敏感性",
            base.dataframe_to_markdown(interval_sensitivity_df, digits=4),
            "",
        ]
    )


def main() -> None:
    ensure_output_dir()
    master_df = base.build_master_table()
    layer_df = build_layer_profile(master_df)
    indicator_matrix = structural_indicator_matrix(master_df)
    normalized = indicator_matrix.apply(base.minmax, axis=0)
    ahp_matrix, subjective_weights, ahp_summary = base.build_ahp_result(list(indicator_matrix.columns))
    objective_weights = base.entropy_weight(normalized)
    weight_df = base.combine_weights(subjective_weights, objective_weights)
    combined_weights = weight_df["组合权重"]
    market_values = base.topsis_score(normalized, combined_weights)

    best_config, tuning_df, heatmap_df = tune_parameters_layered(master_df, normalized, market_values, combined_weights, layer_df)
    dynamic_df = base.compute_dynamic_factors(master_df, best_config.dynamic)
    loocv_df = run_loocv_layered(master_df, normalized, market_values, combined_weights, dynamic_df, layer_df, best_config)
    interval_sensitivity_df = base.build_interval_sensitivity(loocv_df)

    best_interval = interval_sensitivity_df.iloc[(interval_sensitivity_df["样本覆盖率"] - 0.8).abs().argsort()].iloc[0]
    interval_df = loocv_df.copy()
    interval_df["保守报价"] = interval_df["预测成交价"] * (1 - best_interval["delta1"])
    interval_df["基准报价"] = interval_df["预测成交价"]
    interval_df["进取报价"] = interval_df["预测成交价"] * (1 + best_interval["delta2"])

    indexed_master = master_df.set_index("country_region")
    market_value_export = pd.DataFrame(
        {
            "市场": [base.market_to_cn(name) for name in market_values.index],
            "结构TOPSIS价值指数": market_values.values,
            "2026成交价": indexed_master.loc[market_values.index, "2026成交价"].values,
            "总GDP": indexed_master.loc[market_values.index, "总GDP"].values,
            "人口规模": indexed_master.loc[market_values.index, "人口规模"].values,
        }
    ).sort_values("结构TOPSIS价值指数", ascending=False)

    normalized_export = normalized.reset_index().rename(columns={"index": "市场代码", "country_region": "市场代码"})
    normalized_export.insert(1, "市场", normalized_export["市场代码"].map(base.market_to_cn))
    feature_export = master_df.rename(columns={"country_region": "市场代码"}).copy()
    factor_export = dynamic_df.rename(columns={"country_region": "市场代码"}).copy()
    layer_export = layer_df.copy()
    layer_export["层级因子(eta=0.15)"] = layer_factor(layer_export["规则修正层级"], 0.15)
    layer_export["层级因子(eta=0.20)"] = layer_factor(layer_export["规则修正层级"], 0.20)
    layer_export["层级因子(eta=0.25)"] = layer_factor(layer_export["规则修正层级"], 0.25)
    layer_export["层级因子(eta=0.30)"] = layer_factor(layer_export["规则修正层级"], 0.30)
    layer_export["最优层级因子"] = layer_factor(layer_export["规则修正层级"], best_config.eta)
    model_note_df = pd.DataFrame(
        [
            {"项目": "TOPSIS剔除指标", "说明": "总GDP、人口规模"},
            {"项目": "层级判定方式", "说明": "定量评分S + 规则修正"},
            {"项目": "参考市场筛选", "说明": "仅在同层或邻层市场中计算灰色关联并选择前k个参考市场"},
            {"项目": "层级因子公式", "说明": "L_i = exp[eta * (layer_i - 3)]"},
        ]
    )

    base.save_csv(feature_export, "第二问综合指标数据表_分层版.csv")
    base.save_csv(normalized_export, "第二问结构TOPSIS标准化矩阵_分层版.csv")
    base.save_csv(ahp_matrix.reset_index().rename(columns={"index": "指标"}), "第二问AHP判断矩阵_分层版.csv")
    base.save_csv(ahp_summary, "第二问AHP一致性检验结果_分层版.csv")
    base.save_csv(weight_df.reset_index().rename(columns={"index": "指标"}), "第二问结构TOPSIS组合赋权结果_分层版.csv")
    base.save_csv(market_value_export, "第二问结构TOPSIS市场价值指数_分层版.csv")
    base.save_csv(factor_export, "第二问动态修正因子表_分层版.csv")
    base.save_csv(layer_export, "第二问市场层级判定表_分层版.csv")
    base.save_csv(model_note_df, "第二问模型改进说明_分层版.csv")
    base.save_csv(tuning_df, "第二问参数寻优结果_分层版.csv")
    base.save_csv(heatmap_df, "第二问参数热力图数据_分层版.csv")
    base.save_csv(interval_sensitivity_df, "第二问报价区间敏感性分析_分层版.csv")
    base.save_csv(loocv_df, "第二问留一交叉验证结果_分层版.csv")
    base.save_csv(interval_df, "第二问留一交叉验证报价区间_分层版.csv")

    plot_market_value_rank(market_values, indexed_master["2026成交价"])
    plot_heatmap(heatmap_df)
    base.plot_prediction_bar(loocv_df)
    base.plot_relative_error_bar(loocv_df)

    summary_md = build_summary_markdown(best_config, loocv_df, tuning_df, interval_sensitivity_df)
    base.save_markdown(summary_md, "第二问模型测试结果说明_分层版.md")


if __name__ == "__main__":
    main()
