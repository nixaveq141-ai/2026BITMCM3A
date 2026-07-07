import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


_FONT_PATHS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datas"
OUTPUT_DIR = ROOT / "q1" / "outputs"

TARGET_MARKET = "Japan"
TARGET_LABEL = "日本"
PRICE_UNIT = "百万美元"
DISCOUNT_RATE = 0.04
GREY_RHO = 0.5
COMBINATION_LAMBDA = 0.5
CORRELATION_THRESHOLD = 0.85

ALPHA_GRID = np.round(np.arange(0.40, 1.41, 0.10), 2)
GAMMA1_GRID = np.round(np.arange(0.05, 0.31, 0.05), 2)
GAMMA2_GRID = np.round(np.arange(0.05, 0.26, 0.05), 2)
ETA_GRID = [0.15, 0.20, 0.25, 0.30]
K_GRID = [2, 3, 4]

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

DYNAMIC_BETA = 0.25
QUALIFIED_BONUS = 0.10
UNQUALIFIED_DISCOUNT = 0.05
COMPETITION_C = 0.08
RIGHTS_STEP = 0.04

MARKET_NAME_MAP = {
    "United_States": "美国",
    "United_Kingdom": "英国",
    "Japan": "日本",
    "South_Korea": "韩国",
    "Germany": "德国",
    "Brazil": "巴西",
    "Hong_Kong": "中国香港",
    "Vietnam": "越南",
    "Australia": "澳大利亚",
    "China": "中国内地",
    "France": "法国",
    "Spain": "西班牙",
    "Italy": "意大利",
    "India": "印度",
    "Malaysia": "马来西亚",
    "Thailand": "泰国",
}

Q1_ALLOWED_MARKETS = {
    "United_States",
    "United_Kingdom",
    "Japan",
    "South_Korea",
    "Germany",
    "Vietnam",
    "Brazil",
    "Australia",
    "Hong_Kong",
}
SINGLE_TOURNAMENT_MARKETS = {"Brazil"}

LEAGUE_STRENGTH_MAP = {"Top": 3.0, "Medium": 2.0, "Low": 1.0}
COMPETITION_LEVEL_MAP = {"High": 3.0, "Medium": 2.0, "Low": 1.0}
GROWTH_TREND_MAP = {"Strong_Growth": 3.0, "Growing": 2.0, "Stable": 1.5, "Declining_2022": 1.0}
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

BASE_AHP_SCORES = {
    "潜在观众基数": 8.0,
    "历史观赛代理值": 7.0,
    "球迷人数": 8.0,
    "人均GDP": 6.0,
    "足球热度指数": 8.0,
    "历史转播权价格": 8.0,
}


@dataclass
class ModelConfig:
    use_combined_audience: bool
    correlation_value: float
    indicators: List[str]
    best_alpha: float
    best_k: int
    best_gamma1: float
    best_gamma2: float
    best_eta: float


@dataclass
class AhpResult:
    matrix: pd.DataFrame
    weights: pd.Series
    lambda_max: float
    ci: float
    cr: float
    consistency_passed: bool


@dataclass
class PredictionResult:
    estimated_price: float
    base_price: float
    selected_refs: pd.DataFrame
    grey_scores: pd.Series


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def market_to_cn(name: str) -> str:
    return MARKET_NAME_MAP.get(name, name)


def parse_numeric_maybe_range(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if "_" in text and text.replace("_", "").replace(".", "").isdigit():
        parts = [float(x) for x in text.split("_") if x]
        return float(sum(parts) / len(parts))
    return float(text)


def bundle_to_single(price: float, package_coverage: str, country_region: str) -> float:
    if country_region in SINGLE_TOURNAMENT_MARKETS or "2026_2030" not in str(package_coverage):
        return price
    factor = 1.0 + (1.0 + DISCOUNT_RATE) ** -4
    return price / factor


def minmax(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    min_v = float(numeric.min())
    max_v = float(numeric.max())
    if math.isclose(min_v, max_v):
        return pd.Series(np.ones(len(numeric)), index=numeric.index)
    return (numeric - min_v) / (max_v - min_v)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std(ddof=0))
    if np.isclose(std, 0.0):
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - float(numeric.mean())) / std


def safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(series, errors="coerce").clip(lower=0.0))


def safe_ratio(numerator: float, denominator: float) -> float:
    return max(float(numerator), 1e-6) / max(float(denominator), 1e-6)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


def save_text(content: str, filename: str) -> None:
    (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")


def save_markdown(content: str, filename: str) -> None:
    (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def save_image(image: Image.Image, filename: str) -> None:
    image.save(OUTPUT_DIR / filename)


def draw_horizontal_bar_chart(series: pd.Series, filename: str, title: str, subtitle: str, color: str) -> None:
    data = series.sort_values(ascending=True)
    width, height = 1360, max(560, 180 + 62 * len(data))
    image, draw = make_canvas(width, height)
    title_font = get_font(30)
    label_font = get_font(20)
    value_font = get_font(16)
    draw.text((width // 2 - 220, 20), title, fill="black", font=title_font)
    draw.text((60, 70), subtitle, fill="#444444", font=get_font(16))
    left, right, top = 280, 1160, 120
    max_value = float(data.max()) if len(data) else 1.0
    for idx, (label, value) in enumerate(data.items()):
        y = top + idx * 56
        bar_width = int((float(value) / max_value) * (right - left)) if max_value > 0 else 0
        draw.text((35, y), str(label), fill="black", font=label_font)
        draw.rectangle([left, y + 8, left + bar_width, y + 34], fill=color)
        draw.text((left + bar_width + 14, y + 6), f"{float(value):.4f}", fill="black", font=value_font)
    save_image(image, filename)


def draw_line_chart(df: pd.DataFrame, x_col: str, y_col: str, filename: str, title: str) -> None:
    width, height = 1260, 720
    image, draw = make_canvas(width, height)
    title_font = get_font(30)
    axis_font = get_font(18)
    value_font = get_font(15)
    draw.text((width // 2 - 260, 20), title, fill="black", font=title_font)
    left, right, top, bottom = 120, 1120, 120, 620
    draw.line([left, bottom, right, bottom], fill="black", width=2)
    draw.line([left, top, left, bottom], fill="black", width=2)
    x_values = df[x_col].astype(float).to_list()
    y_values = df[y_col].astype(float).to_list()
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    points: list[tuple[int, int]] = []
    for x_val, y_val in zip(x_values, y_values):
        x = int(left + (x_val - min_x) / span_x * (right - left))
        y = int(bottom - (y_val - min_y) / span_y * (bottom - top))
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill="#2563eb", width=3)
    for x, y, x_val, y_val in zip([p[0] for p in points], [p[1] for p in points], x_values, y_values):
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="#2563eb")
        draw.text((x - 14, bottom + 12), f"{x_val:.2f}", fill="black", font=value_font)
        draw.text((40, y - 8), f"{y_val:.3f}", fill="black", font=value_font)
    best_row = df.loc[df[y_col].idxmin()]
    best_x = int(left + (float(best_row[x_col]) - min_x) / span_x * (right - left))
    best_y = int(bottom - (float(best_row[y_col]) - min_y) / span_y * (bottom - top))
    draw.ellipse([best_x - 7, best_y - 7, best_x + 7, best_y + 7], fill="#dc2626")
    draw.text((best_x + 16, best_y - 20), f"最优点 alpha={float(best_row[x_col]):.2f}\n误差={float(best_row[y_col]):.4f}", fill="#dc2626", font=value_font)
    draw.text((width // 2 - 60, 660), x_col, fill="black", font=axis_font)
    draw.text((20, 90), y_col, fill="black", font=axis_font)
    save_image(image, filename)


def draw_grouped_bar_chart(df: pd.DataFrame, filename: str, title: str) -> None:
    plot_df = df.copy()
    plot_df.index = plot_df.index.map(str)
    width, height = 1480, 780
    image, draw = make_canvas(width, height)
    title_font = get_font(30)
    label_font = get_font(16)
    draw.text((width // 2 - 220, 20), title, fill="black", font=title_font)
    left, right, top, bottom = 120, 1360, 140, 640
    draw.line([left, bottom, right, bottom], fill="black", width=2)
    max_value = float(plot_df[["主观权重", "客观权重", "组合权重"]].to_numpy().max())
    group_width = (right - left) / max(len(plot_df), 1)
    bar_width = group_width / 5
    colors = [("#f59e0b", "主观权重"), ("#0ea5e9", "客观权重"), ("#10b981", "组合权重")]
    for idx, (name, row) in enumerate(plot_df.iterrows()):
        group_left = left + idx * group_width
        for offset, (color, col) in enumerate(colors):
            bar_left = group_left + (offset + 1) * bar_width
            bar_top = bottom - (float(row[col]) / max_value) * (bottom - top) if max_value > 0 else bottom
            draw.rectangle([bar_left, bar_top, bar_left + bar_width, bottom], fill=color)
        draw.text((group_left + 5, bottom + 12), str(name), fill="black", font=label_font)
    legend_y = 90
    for idx, (color, label) in enumerate(colors):
        x = 180 + idx * 220
        draw.rectangle([x, legend_y, x + 28, legend_y + 20], fill=color)
        draw.text((x + 40, legend_y - 2), label, fill="black", font=label_font)
    save_image(image, filename)


def draw_target_calibration_chart(selected: pd.DataFrame, final_price: float, filename: str) -> None:
    plot_df = selected.sort_values("灰色权重", ascending=True).copy()
    width, height = 1480, max(640, 180 + 62 * len(plot_df))
    image, draw = make_canvas(width, height)
    title_font = get_font(30)
    label_font = get_font(20)
    value_font = get_font(16)
    draw.text((width // 2 - 210, 20), f"{TARGET_LABEL}成交价复盘分解图", fill="black", font=title_font)
    left, right, top = 320, 1240, 120
    max_value = max(float(plot_df["校准后目标基准价格"].max()), float(final_price))
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        y = top + idx * 56
        value = float(row["校准后目标基准价格"])
        bar_width = int((value / max_value) * (right - left)) if max_value > 0 else 0
        draw.text((40, y), str(row["国家地区"]), fill="black", font=label_font)
        draw.rectangle([left, y + 8, left + bar_width, y + 34], fill="#60a5fa")
        draw.text((left + bar_width + 12, y + 6), f"{value:.2f}", fill="black", font=value_font)
    line_x = int(left + (float(final_price) / max_value) * (right - left)) if max_value > 0 else left
    draw.line([line_x, top - 8, line_x, top + len(plot_df) * 56], fill="#dc2626", width=3)
    draw.text((line_x + 8, top - 18), f"加权基准价 {final_price:.2f}", fill="#dc2626", font=value_font)
    save_image(image, filename)


def clean_country_deals() -> pd.DataFrame:
    deals = pd.read_csv(DATA_DIR / "01_broadcasting_rights_2026.csv")
    deals = deals[(deals["deal_status"] == "confirmed") & (deals["country_region"].isin(Q1_ALLOWED_MARKETS))].copy()
    deals["price_usd_million"] = deals["price_usd_million"].map(parse_numeric_maybe_range)
    deals["单届折算价格"] = deals.apply(
        lambda row: bundle_to_single(row["price_usd_million"], row["package_coverage"], row["country_region"]),
        axis=1,
    )
    summary = (
        deals.groupby("country_region", as_index=False)
        .agg(
            **{
                "2026成交价格_百万美元": ("单届折算价格", "sum"),
                "广播商数量": ("broadcaster", "nunique"),
                "广播商列表": ("broadcaster", lambda x: "、".join(sorted(set(x)))),
                "套餐类型": ("package_coverage", lambda x: "、".join(sorted(set(x)))),
            }
        )
    )
    return summary


def latest_historical_price() -> pd.DataFrame:
    hist = pd.read_csv(DATA_DIR / "02_historical_broadcasting_rights.csv")
    hist = hist[hist["country_region"].isin(Q1_ALLOWED_MARKETS)].copy()
    hist["price_usd_million"] = hist["price_usd_million"].map(parse_numeric_maybe_range)
    latest = (
        hist.sort_values(["country_region", "world_cup_year"], ascending=[True, False])
        .groupby("country_region", as_index=False)
        .agg(
            **{
                "历史转播权价格_百万美元": ("price_usd_million", "first"),
                "历史价格对应年份": ("world_cup_year", "first"),
            }
        )
    )
    return latest


def build_viewership_proxy() -> pd.DataFrame:
    view = pd.read_csv(DATA_DIR / "05_worldcup_viewership.csv")
    view = view[view["country_region"].isin(Q1_ALLOWED_MARKETS)].copy()
    unit_factor = {
        "million_people": 1.0,
        "million_viewers": 1.0,
        "billion_views": 250.0,
    }
    view["单位系数"] = view["unit"].map(unit_factor).fillna(1.0)
    view["观赛代理值"] = view[["value_2022", "value_2018"]].apply(pd.to_numeric, errors="coerce").mean(axis=1) * view["单位系数"]
    return view.groupby("country_region", as_index=False)["观赛代理值"].sum().rename(columns={"观赛代理值": "历史观赛代理值"})


def infer_competition_score(row: pd.Series) -> int:
    broadcasters = str(row["广播商列表"])
    count_rows = int(row["广播商数量"])
    if row["country_region"] == "United_States":
        return 4
    if "BBC_ITV" in broadcasters:
        return 2
    if any(key in broadcasters for key in ["Docomo", "DAZN", "Dentsu", "Globo", "CazeTV", "SBT", "ARD_ZDF", "RAI_DAZN"]):
        return 3
    if count_rows >= 2:
        return 3
    if float(row["主要竞购方数量"]) >= 3 and str(row["country_region"]) not in {"Australia", "Vietnam", "Hong_Kong"}:
        return 2
    return 1


def build_master_table() -> pd.DataFrame:
    deals = clean_country_deals()
    hist = latest_historical_price()
    econ = pd.read_csv(DATA_DIR / "03_economic_indicators.csv")
    econ = econ[econ["country_region"].isin(Q1_ALLOWED_MARKETS)].copy()
    football = pd.read_csv(DATA_DIR / "04_football_market_indicators.csv")
    football = football[football["country_region"].isin(Q1_ALLOWED_MARKETS)].copy()
    timezone = pd.read_csv(DATA_DIR / "06_timezone_viewer_convenience.csv")
    timezone = timezone[timezone["country_region"].isin(Q1_ALLOWED_MARKETS)].copy()
    market = pd.read_csv(DATA_DIR / "07_market_structure.csv")
    market = market[market["country_region"].isin(Q1_ALLOWED_MARKETS)].copy()
    view = build_viewership_proxy()

    timezone["viewer_friendliness_score_1_to_5"] = pd.to_numeric(timezone["viewer_friendliness_score_1_to_5"], errors="coerce")
    timezone["prime_time_match_pct"] = pd.to_numeric(timezone["prime_time_match_pct"], errors="coerce")
    timezone["比赛时段友好度"] = timezone["viewer_friendliness_score_1_to_5"] / 5.0 * 0.6 + timezone["prime_time_match_pct"] / 100.0 * 0.4
    timezone = timezone.groupby("country_region", as_index=False)["比赛时段友好度"].mean()

    merged = deals.merge(econ, on="country_region", how="left")
    merged = merged.merge(football, on="country_region", how="left")
    merged = merged.merge(timezone, on="country_region", how="left")
    merged = merged.merge(
        market[
            [
                "country_region",
                "total_gdp_billion_usd_2024",
                "competition_level",
                "num_major_bidders",
                "historical_wc_viewing_growth_trend",
                "media_broadcast_market_maturity",
            ]
        ],
        on="country_region",
        how="left",
    )
    merged = merged.merge(hist, on="country_region", how="left")
    merged = merged.merge(view, on="country_region", how="left")

    merged["国家地区"] = merged["country_region"].map(market_to_cn)
    merged["人口规模_百万人"] = pd.to_numeric(merged["population_million_2024"], errors="coerce")
    merged["GDP总量_十亿美元"] = pd.to_numeric(merged["total_gdp_billion_usd_2024"], errors="coerce")
    merged["人均GDP_美元"] = pd.to_numeric(merged["gdp_per_capita_usd_2024"], errors="coerce")
    merged["互联网渗透率"] = pd.to_numeric(merged["internet_penetration_pct_2024"], errors="coerce")
    merged["体育媒体市场规模_十亿美元"] = pd.to_numeric(merged["sports_media_market_billion_usd_est"], errors="coerce")
    merged["球迷占比"] = pd.to_numeric(merged["football_fan_pct_of_population"], errors="coerce")
    merged["球迷人数_百万人"] = pd.to_numeric(merged["football_fans_million"], errors="coerce")
    merged["主要竞购方数量"] = pd.to_numeric(merged["num_major_bidders"], errors="coerce")
    merged["历史转播权价格_百万美元"] = pd.to_numeric(merged["历史转播权价格_百万美元"], errors="coerce")
    merged["历史观赛代理值"] = pd.to_numeric(merged["历史观赛代理值"], errors="coerce")

    merged["是否参赛"] = merged["qualified_2026_wc"].fillna("No").str.startswith("Yes").astype(int)
    merged["联赛强度分值"] = merged["domestic_league_strength_tier"].map(LEAGUE_STRENGTH_MAP).fillna(1.0)
    merged["竞争层级分值"] = merged["competition_level"].map(COMPETITION_LEVEL_MAP).fillna(1.0)
    merged["增长趋势分值"] = merged["historical_wc_viewing_growth_trend"].map(GROWTH_TREND_MAP).fillna(1.5)

    merged["比赛时段友好度"] = merged["比赛时段友好度"].fillna(merged["比赛时段友好度"].median())
    merged["历史观赛代理值"] = merged["历史观赛代理值"].fillna(
        merged["球迷人数_百万人"] * (0.35 + 0.65 * merged["比赛时段友好度"].fillna(0.4))
    )

    rank_series = pd.to_numeric(merged["fifa_ranking_june_2026"], errors="coerce")
    max_rank = float(rank_series.max(skipna=True))
    merged["FIFA排名得分"] = np.where(rank_series.notna(), (max_rank + 1 - rank_series) / max_rank, 0.0)
    merged["球迷人数对数得分"] = minmax(np.log1p(merged["球迷人数_百万人"]))
    merged["足球热度指数"] = (
        0.25 * merged["FIFA排名得分"]
        + 0.25 * minmax(merged["球迷占比"])
        + 0.30 * merged["球迷人数对数得分"]
        + 0.20 * minmax(merged["联赛强度分值"])
    )

    merged["媒体竞购强度"] = merged.apply(infer_competition_score, axis=1)

    ratio = (
        merged.loc[merged["历史转播权价格_百万美元"].notna(), "历史转播权价格_百万美元"]
        / merged.loc[merged["历史转播权价格_百万美元"].notna(), "2026成交价格_百万美元"]
    ).median()
    merged["历史价格是否插补"] = 0
    missing_mask = merged["历史转播权价格_百万美元"].isna()
    merged.loc[missing_mask, "历史转播权价格_百万美元"] = merged.loc[missing_mask, "2026成交价格_百万美元"] * ratio
    merged.loc[missing_mask, "历史价格是否插补"] = 1

    merged["潜在观众基数"] = np.sqrt(
        np.log1p(merged["人口规模_百万人"].clip(lower=0.01)) * np.log1p(merged["历史观赛代理值"].clip(lower=0.01))
    )

    keep_cols = [
        "country_region",
        "国家地区",
        "2026成交价格_百万美元",
        "广播商数量",
        "广播商列表",
        "套餐类型",
        "人口规模_百万人",
        "历史观赛代理值",
        "潜在观众基数",
        "GDP总量_十亿美元",
        "人均GDP_美元",
        "体育媒体市场规模_十亿美元",
        "球迷占比",
        "球迷人数_百万人",
        "足球热度指数",
        "是否参赛",
        "媒体竞购强度",
        "比赛时段友好度",
        "历史转播权价格_百万美元",
        "历史价格对应年份",
        "历史价格是否插补",
        "主要竞购方数量",
        "media_broadcast_market_maturity",
    ]
    return merged[keep_cols].sort_values("2026成交价格_百万美元", ascending=False).reset_index(drop=True)


def should_combine_audience(df: pd.DataFrame) -> tuple[bool, float]:
    corr = np.log1p(df["人口规模_百万人"]).corr(np.log1p(df["历史观赛代理值"]))
    return bool(corr > CORRELATION_THRESHOLD), float(corr)


def build_indicator_matrix(df: pd.DataFrame, use_combined_audience: bool) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    if use_combined_audience:
        data["潜在观众基数"] = np.log1p(df["潜在观众基数"]).to_numpy()
    else:
        data["历史观赛代理值"] = np.log1p(df["历史观赛代理值"]).to_numpy()
        data["球迷人数"] = np.log1p(df["球迷人数_百万人"]).to_numpy()
    data["人均GDP"] = np.log1p(df["人均GDP_美元"]).to_numpy()
    data["足球热度指数"] = df["足球热度指数"].to_numpy()
    data["历史转播权价格"] = np.log1p(df["历史转播权价格_百万美元"]).to_numpy()
    return pd.DataFrame(data, index=df["country_region"])


def build_ahp_result(indicators: Sequence[str]) -> AhpResult:
    scores = np.array([BASE_AHP_SCORES[name] for name in indicators], dtype=float)
    matrix = np.outer(scores, 1 / scores)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues[max_idx].real)
    principal = np.abs(eigenvectors[:, max_idx].real)
    weights = principal / principal.sum()
    n = len(indicators)
    ci = 0.0 if n <= 2 else (lambda_max - n) / (n - 1)
    ri = RI_TABLE.get(n, 1.49)
    cr = 0.0 if ri == 0 else ci / ri
    return AhpResult(
        matrix=pd.DataFrame(matrix, index=indicators, columns=indicators),
        weights=pd.Series(weights, index=indicators, name="主观权重"),
        lambda_max=lambda_max,
        ci=ci,
        cr=cr,
        consistency_passed=cr < 0.10,
    )


def entropy_weight(normalized: pd.DataFrame) -> pd.Series:
    x = normalized.to_numpy(dtype=float)
    col_sums = x.sum(axis=0)
    p = np.divide(x, col_sums, out=np.full_like(x, 1.0 / len(normalized)), where=col_sums > 0)
    k = 1.0 / math.log(len(normalized))
    plogp = np.zeros_like(p)
    mask = p > 0
    plogp[mask] = p[mask] * np.log(p[mask])
    entropy = -k * np.nansum(plogp, axis=0)
    diff = 1 - entropy
    weights = diff / diff.sum() if not np.isclose(diff.sum(), 0) else np.full(len(diff), 1.0 / len(diff))
    return pd.Series(weights, index=normalized.columns, name="客观权重")


def combine_weights(subjective: pd.Series, objective: pd.Series) -> pd.DataFrame:
    final = COMBINATION_LAMBDA * subjective + (1 - COMBINATION_LAMBDA) * objective
    final = final / final.sum()
    return pd.DataFrame({"主观权重": subjective, "客观权重": objective, "组合权重": final})


def topsis_score(normalized: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weighted = normalized.mul(weights, axis=1)
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    return (dist_worst / (dist_best + dist_worst)).rename("结构价值指数")


def grey_relation_scores(normalized: pd.DataFrame, weights: pd.Series, target: str, ref_pool: Iterable[str]) -> pd.Series:
    ref_list = list(ref_pool)
    delta = (normalized.loc[ref_list] - normalized.loc[target]).abs()
    delta_min = float(delta.min().min())
    delta_max = float(delta.max().max())
    coeff = (delta_min + GREY_RHO * delta_max) / (delta + GREY_RHO * delta_max)
    scores = coeff.mul(weights, axis=1).sum(axis=1)
    return scores.rename("灰色关联度").sort_values(ascending=False)


def right_structure_complexity(df: pd.DataFrame) -> pd.Series:
    structure_count = df["套餐类型"].fillna("").astype(str).map(lambda text: len([part for part in text.split("、") if part.strip()]))
    package_bonus = df["套餐类型"].fillna("").astype(str).str.contains("package", case=False).astype(int)
    return structure_count + 0.5 * package_bonus + 0.5 * pd.to_numeric(df["广播商数量"], errors="coerce").fillna(1.0)


def build_layer_profile(df: pd.DataFrame) -> pd.DataFrame:
    layer_df = df[
        [
            "country_region",
            "国家地区",
            "GDP总量_十亿美元",
            "人口规模_百万人",
            "体育媒体市场规模_十亿美元",
            "历史转播权价格_百万美元",
            "主要竞购方数量",
            "广播商数量",
            "套餐类型",
        ]
    ].copy()
    layer_df["权利结构复杂度"] = right_structure_complexity(df)
    layer_df["Z_lnGDP"] = zscore(np.log(layer_df["GDP总量_十亿美元"].clip(lower=1e-6)))
    layer_df["Z_ln人口"] = zscore(np.log(layer_df["人口规模_百万人"].clip(lower=1e-6)))
    layer_df["Z_ln媒体规模"] = zscore(np.log(layer_df["体育媒体市场规模_十亿美元"].clip(lower=1e-6)))
    layer_df["Z_ln历史价格"] = zscore(np.log(layer_df["历史转播权价格_百万美元"].clip(lower=1e-6)))
    layer_df["Z_竞购方数量"] = zscore(layer_df["主要竞购方数量"])
    layer_df["Z_权利复杂度"] = zscore(layer_df["权利结构复杂度"])
    score_cols = ["Z_lnGDP", "Z_ln人口", "Z_ln媒体规模", "Z_ln历史价格", "Z_竞购方数量", "Z_权利复杂度"]
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
    return refs if refs else [market for market in all_markets if market != target_market]


def compute_dynamic_factors(df: pd.DataFrame) -> pd.DataFrame:
    dynamic = df[["country_region", "国家地区", "是否参赛", "媒体竞购强度", "比赛时段友好度", "广播商数量"]].copy()
    dynamic["参赛因子"] = np.where(dynamic["是否参赛"] == 1, 1.0 + QUALIFIED_BONUS, 1.0 - UNQUALIFIED_DISCOUNT)
    dynamic["竞购因子"] = 1.0 + COMPETITION_C * (pd.to_numeric(dynamic["媒体竞购强度"], errors="coerce") - 1.0)
    dynamic["时差因子"] = 0.65 + 0.35 * pd.to_numeric(dynamic["比赛时段友好度"], errors="coerce").clip(lower=0.0, upper=1.0)
    dynamic["权利结构因子"] = 1.0 + RIGHTS_STEP * (pd.to_numeric(dynamic["广播商数量"], errors="coerce").clip(lower=1.0) - 1.0)
    dynamic["动态修正总因子"] = (
        dynamic["参赛因子"] ** DYNAMIC_BETA
        * dynamic["竞购因子"]
        * dynamic["时差因子"]
        * dynamic["权利结构因子"]
    )
    return dynamic


def predict_target_price(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    market_scores: pd.Series,
    weights: pd.Series,
    dynamic_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    target: str,
    alpha: float,
    gamma1: float,
    gamma2: float,
    eta: float,
    k_refs: int,
) -> PredictionResult:
    layer_map = layer_df.set_index("country_region")["规则修正层级"].to_dict()
    ref_pool = eligible_reference_markets(layer_map, target, df["country_region"].tolist())
    grey_scores = grey_relation_scores(normalized, weights, target, ref_pool)
    refs = list(grey_scores.head(min(k_refs, len(grey_scores))).index)

    df_map = df.set_index("country_region")
    dynamic_map = dynamic_df.set_index("country_region")
    layer_factor_map = layer_factor(layer_df.set_index("country_region")["规则修正层级"], eta)

    target_gdp = float(df_map.loc[target, "GDP总量_十亿美元"])
    target_population = float(df_map.loc[target, "人口规模_百万人"])
    target_value = float(market_scores.loc[target])
    target_dynamic = float(dynamic_map.loc[target, "动态修正总因子"])
    target_layer_factor = float(layer_factor_map.loc[target])

    selected = pd.DataFrame(index=refs)
    selected["国家地区"] = [market_to_cn(x) for x in refs]
    selected["灰色关联度"] = grey_scores.loc[refs].values
    selected["灰色权重"] = selected["灰色关联度"] / selected["灰色关联度"].sum()
    selected["结构价值指数"] = market_scores.loc[refs].values
    selected["统一口径成交价_百万美元"] = df_map.loc[refs, "2026成交价格_百万美元"].values
    selected["参考市场层级"] = layer_df.set_index("country_region").loc[refs, "层级标签"].values

    ref_dynamic = dynamic_map.loc[refs, "动态修正总因子"]
    ref_layer_factor = layer_factor_map.loc[refs]
    selected["动态因子"] = ref_dynamic.values
    selected["层级因子"] = ref_layer_factor.values
    selected["参考市场基准价格"] = selected["统一口径成交价_百万美元"] / (selected["动态因子"] * selected["层级因子"])

    ref_gdp = df_map.loc[refs, "GDP总量_十亿美元"]
    ref_population = df_map.loc[refs, "人口规模_百万人"]
    selected["GDP对数差值"] = np.log(ref_gdp.map(lambda value: safe_ratio(target_gdp, value)))
    selected["人口对数差值"] = np.log(ref_population.map(lambda value: safe_ratio(target_population, value)))
    selected["体量平滑乘数"] = np.maximum(
        0.1,
        1.0 + gamma1 * selected["GDP对数差值"] + gamma2 * selected["人口对数差值"],
    )
    selected["结构价值比"] = selected["结构价值指数"].map(lambda value: safe_ratio(target_value, value))
    selected["结构弹性项"] = np.power(selected["结构价值比"], alpha)
    selected["校准后目标基准价格"] = (
        selected["参考市场基准价格"] * selected["体量平滑乘数"] * selected["结构弹性项"]
    )

    base_price = float((selected["灰色权重"] * selected["校准后目标基准价格"]).sum())
    final_price = base_price * target_dynamic * target_layer_factor

    return PredictionResult(
        estimated_price=final_price,
        base_price=base_price,
        selected_refs=selected.reset_index().rename(columns={"index": "country_region"}),
        grey_scores=grey_scores,
    )


def tune_layered_parameters(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    market_scores: pd.Series,
    weights: pd.Series,
    dynamic_df: pd.DataFrame,
    layer_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    markets = df["country_region"].tolist()
    actual_map = df.set_index("country_region")["2026成交价格_百万美元"]
    for k in K_GRID:
        for alpha in ALPHA_GRID:
            for gamma1 in GAMMA1_GRID:
                for gamma2 in GAMMA2_GRID:
                    for eta in ETA_GRID:
                        errors = []
                        for market in markets:
                            pred = predict_target_price(
                                df=df,
                                normalized=normalized,
                                market_scores=market_scores,
                                weights=weights,
                                dynamic_df=dynamic_df,
                                layer_df=layer_df,
                                target=market,
                                alpha=float(alpha),
                                gamma1=float(gamma1),
                                gamma2=float(gamma2),
                                eta=float(eta),
                                k_refs=int(k),
                            )
                            actual = float(actual_map.loc[market])
                            errors.append(abs(pred.estimated_price - actual) / actual)
                        rows.append(
                            {
                                "参考市场数量k": int(k),
                                "结构价格弹性系数alpha": float(alpha),
                                "GDP对数敏感系数gamma1": float(gamma1),
                                "人口对数敏感系数gamma2": float(gamma2),
                                "层级溢价强度eta": float(eta),
                                "平均相对误差": float(np.mean(errors)),
                                "中位相对误差": float(np.median(errors)),
                                "最大相对误差": float(np.max(errors)),
                            }
                        )
    return pd.DataFrame(rows).sort_values(["平均相对误差", "中位相对误差", "最大相对误差"]).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    temp = df.copy()
    for col in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[col]):
            temp[col] = temp[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    headers = [str(col) for col in temp.columns]
    rows = [[str(value) for value in row] for row in temp.fillna("").to_numpy()]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def build_result_markdown(
    summary_df: pd.DataFrame,
    market_value_df: pd.DataFrame,
    target_ref_df: pd.DataFrame,
    ahp_summary_df: pd.DataFrame,
    config: ModelConfig,
) -> str:
    lines = [
        "# 第一问代码直接运行结果",
        "",
        "## 核心结论",
        f"- 研究对象：{TARGET_LABEL}",
        "- 建模口径：保留“日本成交价复盘”任务，但估价公式已切换为第二问的分层动态版",
        f"- 观众指标合并策略：{'人口规模与历史观赛代理值已合并为潜在观众基数' if config.use_combined_audience else '人口规模相关信息未合并，采用历史观赛代理值与球迷人数入模'}",
        f"- 对数相关系数：{config.correlation_value:.4f}",
        f"- 最优参考市场数量 k：{config.best_k}",
        f"- 最优结构价格弹性系数 alpha：{config.best_alpha:.2f}",
        f"- 最优 GDP 对数敏感系数 gamma1：{config.best_gamma1:.2f}",
        f"- 最优人口对数敏感系数 gamma2：{config.best_gamma2:.2f}",
        f"- 最优层级溢价强度 eta：{config.best_eta:.2f}",
        "",
        f"## {TARGET_LABEL}解释结果表",
        dataframe_to_markdown(summary_df, digits=4),
        "",
        "## AHP 一致性检验结果",
        dataframe_to_markdown(ahp_summary_df, digits=6),
        "",
        "## 结构价值指数排序",
        dataframe_to_markdown(market_value_df, digits=4),
        "",
        f"## {TARGET_LABEL}复盘参考市场",
        dataframe_to_markdown(target_ref_df, digits=4),
        "",
    ]
    return "\n".join(lines)


def build_analysis_markdown(
    config: ModelConfig,
    used_data_df: pd.DataFrame,
    chart_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> str:
    lines = [
        "# 第一问数据、图表与结果说明",
        "",
        "## 使用的数据文件",
        dataframe_to_markdown(used_data_df, digits=4),
        "",
        "## 图表说明",
        dataframe_to_markdown(chart_df, digits=4),
        "",
        "## 本轮模型修改重点",
        "- 保留“日本成交价复盘”的任务，但把原先直接按结构价值指数比例缩放价格的做法，改成“先剥离参考市场动态因子与层级因子，再回乘日本因子”的分层动态复盘模型。",
        "- TOPSIS 现在只衡量结构价值，不再把总量 GDP、人口规模、参赛、媒体竞购强度、时差友好度直接混入结构价值指数。",
        "- 参考市场集合不再写死，而是依据市场层级满足“同层或相邻层”自动筛选。",
        "- 体量修正改成 GDP 与人口的对数平滑项，并和结构价值弹性项一起作用在参考市场基准价格上。",
        "",
        "## 结果分析",
        f"- 观众代理变量相关系数为 {config.correlation_value:.4f}，{'高于' if config.use_combined_audience else '低于'} 0.85，因此{'采用' if config.use_combined_audience else '未采用'}“潜在观众基数”合并指标。",
        f"- 留一法交叉验证选出的最优参数为 k={config.best_k}、alpha={config.best_alpha:.2f}、gamma1={config.best_gamma1:.2f}、gamma2={config.best_gamma2:.2f}、eta={config.best_eta:.2f}。",
        "- 该模型先把参考市场成交价还原为“基准价格”，再根据日本的时差、竞购、参赛和层级位置回推成交价，因此解释口径比旧版更贴近论文公式。",
        "",
        "## 参数敏感性前十组",
        dataframe_to_markdown(sensitivity_df.head(10), digits=4),
        "",
        f"## {TARGET_LABEL}结果摘要",
        dataframe_to_markdown(summary_df, digits=4),
        "",
    ]
    return "\n".join(lines)


def run_analysis() -> None:
    ensure_output_dir()
    master = build_master_table()
    use_combined_audience, corr_value = should_combine_audience(master)
    indicator_matrix = build_indicator_matrix(master, use_combined_audience)
    normalized = indicator_matrix.apply(minmax, axis=0)

    ahp_result = build_ahp_result(list(indicator_matrix.columns))
    objective_weight = entropy_weight(normalized)
    weight_df = combine_weights(ahp_result.weights, objective_weight)
    weights = weight_df["组合权重"]
    market_scores = topsis_score(normalized, weights).sort_values(ascending=False)

    layer_df = build_layer_profile(master)
    dynamic_df = compute_dynamic_factors(master)
    sensitivity_df = tune_layered_parameters(master, normalized, market_scores, weights, dynamic_df, layer_df)
    best_row = sensitivity_df.iloc[0]
    config = ModelConfig(
        use_combined_audience=use_combined_audience,
        correlation_value=corr_value,
        indicators=list(indicator_matrix.columns),
        best_alpha=float(best_row["结构价格弹性系数alpha"]),
        best_k=int(best_row["参考市场数量k"]),
        best_gamma1=float(best_row["GDP对数敏感系数gamma1"]),
        best_gamma2=float(best_row["人口对数敏感系数gamma2"]),
        best_eta=float(best_row["层级溢价强度eta"]),
    )

    prediction = predict_target_price(
        df=master,
        normalized=normalized,
        market_scores=market_scores,
        weights=weights,
        dynamic_df=dynamic_df,
        layer_df=layer_df,
        target=TARGET_MARKET,
        alpha=config.best_alpha,
        gamma1=config.best_gamma1,
        gamma2=config.best_gamma2,
        eta=config.best_eta,
        k_refs=config.best_k,
    )

    master_map = master.set_index("country_region")
    actual_target_price = float(master_map.loc[TARGET_MARKET, "2026成交价格_百万美元"])
    abs_error = abs(prediction.estimated_price - actual_target_price)
    rel_error = abs_error / actual_target_price

    target_layer_row = layer_df.set_index("country_region").loc[TARGET_MARKET]
    target_dynamic_row = dynamic_df.set_index("country_region").loc[TARGET_MARKET]
    target_layer_factor = float(layer_factor(pd.Series([target_layer_row["规则修正层级"]]), config.best_eta).iloc[0])

    market_value_df = pd.DataFrame(
        {
            "排名": np.arange(1, len(market_scores) + 1),
            "国家地区": [market_to_cn(x) for x in market_scores.index],
            "结构价值指数": market_scores.values,
            "2026统一口径成交价_百万美元": master_map.loc[market_scores.index, "2026成交价格_百万美元"].values,
        }
    )

    grey_df = prediction.grey_scores.reset_index()
    grey_df.columns = ["country_region", "灰色关联度"]
    grey_df["国家地区"] = grey_df["country_region"].map(market_to_cn)
    grey_df["是否进入参考集"] = np.where(grey_df["country_region"].isin(prediction.selected_refs["country_region"]), "是", "否")
    grey_df = grey_df[["国家地区", "灰色关联度", "是否进入参考集"]]

    target_ref_df = prediction.selected_refs.copy()
    target_ref_df = target_ref_df[
        [
            "国家地区",
            "参考市场层级",
            "灰色关联度",
            "灰色权重",
            "结构价值指数",
            "统一口径成交价_百万美元",
            "动态因子",
            "层级因子",
            "参考市场基准价格",
            "GDP对数差值",
            "人口对数差值",
            "体量平滑乘数",
            "结构价值比",
            "结构弹性项",
            "校准后目标基准价格",
        ]
    ]

    summary_df = pd.DataFrame(
        [
            {"指标": "人口-历史观赛代理值相关系数", "结果": corr_value},
            {"指标": "是否合并观众类指标", "结果": "是" if use_combined_audience else "否"},
            {"指标": "AHP一致性是否通过", "结果": "是" if ahp_result.consistency_passed else "否"},
            {"指标": "最优参考市场数量k", "结果": config.best_k},
            {"指标": "最优结构价格弹性系数alpha", "结果": config.best_alpha},
            {"指标": "最优GDP对数敏感系数gamma1", "结果": config.best_gamma1},
            {"指标": "最优人口对数敏感系数gamma2", "结果": config.best_gamma2},
            {"指标": "最优层级溢价强度eta", "结果": config.best_eta},
            {"指标": f"{TARGET_LABEL}市场层级标签", "结果": target_layer_row["层级标签"]},
            {"指标": f"{TARGET_LABEL}市场层级数值", "结果": target_layer_row["规则修正层级"]},
            {"指标": f"{TARGET_LABEL}层级因子", "结果": target_layer_factor},
            {"指标": f"{TARGET_LABEL}动态修正总因子", "结果": target_dynamic_row["动态修正总因子"]},
            {"指标": f"{TARGET_LABEL}模型基准价格_百万美元", "结果": prediction.base_price},
            {"指标": f"{TARGET_LABEL}模型复盘成交价_百万美元", "结果": prediction.estimated_price},
            {"指标": f"{TARGET_LABEL}真实成交价_百万美元", "结果": actual_target_price},
            {"指标": "绝对误差_百万美元", "结果": abs_error},
            {"指标": "相对误差", "结果": f"{rel_error:.2%}"},
        ]
    )

    ahp_summary_df = pd.DataFrame(
        [
            {"指标": "最大特征值lambda_max", "结果": ahp_result.lambda_max},
            {"指标": "一致性指标CI", "结果": ahp_result.ci},
            {"指标": "一致性比例CR", "结果": ahp_result.cr},
            {"指标": "一致性是否通过", "结果": "是" if ahp_result.consistency_passed else "否"},
        ]
    )

    feature_df = master.copy()
    normalized_export = normalized.reset_index().rename(columns={"index": "country_region"})
    normalized_export["国家地区"] = normalized_export["country_region"].map(market_to_cn)
    normalized_export = normalized_export[["国家地区"] + [col for col in normalized_export.columns if col not in {"country_region", "国家地区"}]]

    layer_export = layer_df.copy()
    dynamic_export = dynamic_df.copy()
    weight_export = weight_df.reset_index().rename(columns={"index": "指标"})
    ahp_matrix_export = ahp_result.matrix.reset_index().rename(columns={"index": "指标"})

    save_csv(feature_df, "综合指标数据表.csv")
    save_csv(normalized_export, "标准化指标矩阵.csv")
    save_csv(weight_export, "组合赋权结果.csv")
    save_csv(ahp_matrix_export, "AHP判断矩阵.csv")
    save_csv(ahp_summary_df, "AHP一致性检验结果.csv")
    save_csv(market_value_df, "TOPSIS结构价值指数.csv")
    save_csv(layer_export, "市场层级划分结果.csv")
    save_csv(dynamic_export, "动态修正因子结果.csv")
    save_csv(grey_df, f"{TARGET_LABEL}灰色关联度.csv")
    save_csv(target_ref_df, f"{TARGET_LABEL}价格复盘表.csv")
    save_csv(sensitivity_df, "分层动态参数敏感性分析.csv")
    save_csv(summary_df, f"{TARGET_LABEL}成交价解释结果.csv")

    draw_horizontal_bar_chart(
        market_value_df.set_index("国家地区")["结构价值指数"],
        "结构价值指数排序图.png",
        "结构TOPSIS市场价值指数排序图",
        "横轴为结构价值指数",
        "#2563eb",
    )
    draw_horizontal_bar_chart(
        grey_df.set_index("国家地区")["灰色关联度"],
        f"{TARGET_LABEL}相似市场灰色关联图.png",
        f"{TARGET_LABEL}相似市场灰色关联图",
        "横轴为灰色关联度",
        "#059669",
    )
    draw_line_chart(
        sensitivity_df[sensitivity_df["参考市场数量k"] == config.best_k].sort_values("结构价格弹性系数alpha"),
        "结构价格弹性系数alpha",
        "平均相对误差",
        "分层动态价格弹性系数留一法误差图.png",
        f"分层动态价格弹性系数留一法误差图（k={config.best_k}）",
    )
    draw_grouped_bar_chart(weight_df, "指标权重对比图.png", "主客观组合赋权结果对比图")
    draw_target_calibration_chart(target_ref_df, prediction.base_price, f"{TARGET_LABEL}成交价复盘分解图.png")

    used_data_df = pd.DataFrame(
        [
            {"原始数据文件": "01_broadcasting_rights_2026.csv", "用途": "提取 2026 已成交版权价格，并折算成 2026 单届统一口径成交价"},
            {"原始数据文件": "02_historical_broadcasting_rights.csv", "用途": "提取各市场最近一届历史转播权价格，作为结构价值锚点"},
            {"原始数据文件": "03_economic_indicators.csv", "用途": "提供人口、GDP、人均GDP、体育媒体市场规模等体量变量"},
            {"原始数据文件": "04_football_market_indicators.csv", "用途": "提供 FIFA 排名、是否参赛、球迷规模、联赛强度等足球市场信息"},
            {"原始数据文件": "05_worldcup_viewership.csv", "用途": "构造历史观赛代理值，衡量过往收视需求"},
            {"原始数据文件": "06_timezone_viewer_convenience.csv", "用途": "构造比赛时段友好度，用于时差动态修正"},
            {"原始数据文件": "07_market_structure.csv", "用途": "补充市场成熟度、竞购方数量与市场增长趋势"},
        ]
    )

    chart_df = pd.DataFrame(
        [
            {"图表文件": "结构价值指数排序图", "含义": "比较各样本市场在结构价值维度上的相对高低"},
            {"图表文件": f"{TARGET_LABEL}相似市场灰色关联图", "含义": f"展示各市场与{TARGET_LABEL}在结构指标上的相似程度"},
            {"图表文件": "分层动态价格弹性系数留一法误差图", "含义": "展示在最优 k 下，不同 alpha 对留一法平均误差的影响"},
            {"图表文件": "指标权重对比图", "含义": "对比 AHP 主观权重、熵权法客观权重和组合权重"},
            {"图表文件": f"{TARGET_LABEL}成交价复盘分解图", "含义": f"展示参考市场经体量与结构弹性修正后，对{TARGET_LABEL}基准价格的推导结果"},
        ]
    )

    result_md = build_result_markdown(summary_df, market_value_df, target_ref_df, ahp_summary_df, config)
    analysis_md = build_analysis_markdown(config, used_data_df, chart_df, summary_df, sensitivity_df)
    save_markdown(result_md, "代码运行直接结果.md")
    save_markdown(analysis_md, "数据与图表说明及结果分析.md")
    save_text(
        "\n".join(
            [
                "第一问运行说明",
                f"研究对象：{TARGET_LABEL}",
                f"价格单位：{PRICE_UNIT}",
                "任务定位：保留日本成交价复盘任务，但估价公式已改为第二问的分层动态版",
                f"打包合同折现率 r = {DISCOUNT_RATE:.2%}",
                f"灰色分辨系数 rho = {GREY_RHO:.2f}",
                f"组合赋权系数 lambda = {COMBINATION_LAMBDA:.2f}",
                f"人口-观赛代理值相关系数 = {corr_value:.4f}",
                f"是否合并观众类指标 = {'是' if use_combined_audience else '否'}",
                f"最优 k = {config.best_k}",
                f"最优 alpha = {config.best_alpha:.2f}",
                f"最优 gamma1 = {config.best_gamma1:.2f}",
                f"最优 gamma2 = {config.best_gamma2:.2f}",
                f"最优 eta = {config.best_eta:.2f}",
                f"{TARGET_LABEL}模型基准价格 = {prediction.base_price:.2f} {PRICE_UNIT}",
                f"{TARGET_LABEL}模型复盘成交价 = {prediction.estimated_price:.2f} {PRICE_UNIT}",
                f"{TARGET_LABEL}真实成交价 = {actual_target_price:.2f} {PRICE_UNIT}",
                f"相对误差 = {rel_error:.2%}",
            ]
        ),
        "分析说明.txt",
    )


if __name__ == "__main__":
    run_analysis()
