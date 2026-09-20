import math
from collections import Counter

import numpy as np
import pandas as pd
from pandas import DataFrame


def calculate_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def distribution_summary(s: pd.Series) -> pd.Series:
    counts = s.dropna().value_counts()
    total = counts.sum()
    unique = len(counts)
    if total == 0:
        return pd.Series({
            'unique_ratio': 0.0,
            'repeat_ratio': 0.0,
            'singleton_share': 0.0,
            'top1_share': 0.0,
            'top3_share': 0.0,
            'simpson': 0.0,
            'norm_entropy': 0.0,
            'effective_ratio': 0.0,
        })

    probabilities = counts.to_numpy(dtype=float) / total
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum()
    return pd.Series({
        'unique_ratio': unique / total,
        'repeat_ratio': 1.0 - unique / total,
        'singleton_share': counts.eq(1).sum() / total,
        'top1_share': probabilities.max(),
        'top3_share': np.sort(probabilities)[-3:].sum(),
        'simpson': np.square(probabilities).sum(),
        'norm_entropy': entropy / np.log(unique) if unique > 1 else 0.0,
        'effective_ratio': np.exp(entropy) / unique if unique else 0.0,
    })

def add_ua_features(df: pd.DataFrame, col: str = 'user_agent') -> DataFrame:
    df[col] = df[col].fillna('')

    df['ua_length'] = df[col].str.len()
    df['count_slash'] = df[col].str.count('/')
    df['count_semicolon'] = df[col].str.count(';')
    df['count_dots'] = df[col].str.count(r'\.')
    df['count_open_paren'] = df[col].str.count(r'\(')
    df['count_spaces'] = df[col].str.count(' ')

    df['digit_count'] = df[col].str.count(r'\d')
    df['digit_ratio'] = df['digit_count'] / df['ua_length'].replace(0, 1)
    df['entropy'] = df[col].apply(calculate_entropy)

    tools_pattern = r'(?i)(?:scrapy|curl|wget|node-fetch|python-requests|urllib|aiohttp|axios|httpclient|postman|go-http|java/)'
    headless_pattern = r'(?i)(?:headlesschrome|phantomjs|selenium|puppeteer|playwright)'

    df['is_known_tool'] = df[col].str.contains(tools_pattern, regex=True).astype(int)
    df['is_headless'] = df[col].str.contains(headless_pattern, regex=True).astype(int)
    df['has_url'] = df[col].str.contains(r'https?://', regex=True).astype(int)
    df['starts_with_mozilla'] = df[col].str.startswith('Mozilla/5.0').astype(int)

    df['is_app_header'] = df[col].str.contains(r'^[A-Za-z0-9_-]+/\d+', regex=True) & (~df['starts_with_mozilla'].astype(bool))
    df['is_app_header'] = df['is_app_header'].astype(int)

    os_conditions = [
        df[col].str.contains('Android', case = False),
        df[col].str.contains('iPad', case = False),
        df[col].str.contains(r'iPhone|iOs', regex=True, case=False),
        df[col].str.contains('Windows NT', case = False),
        df[col].str.contains(r'Macintosh|Mac OS X', regex=True, case=False),
        df[col].str.contains('Linux', case = False),
    ]
    os_choices = ['Android', 'iPadOS', 'iOS', 'Windows', 'macOS', 'Linux']
    df['ua_os'] = pd.Series(
        np.select(os_conditions, os_choices, default='Other'),
        index=df.index
    ).astype('category')

    device_conditions = [
        df[col].str.contains('iPad', case = False),
        df[col].str.contains('iPhone', case = False),
        df[col].str.contains(r'Android.*Mobile|Mobile.*Android', regex=True, case=False),
        df[col].str.contains('Android', case = False),
        df[col].str.contains(r'Windows NT|Macintosh|Linux|Mac OS X', regex=True, case=False)
    ]
    device_choices = ['tablet', 'mobile', 'mobile', 'tablet', 'desktop']

    df['ua_device_type'] = pd.Series(
        np.select(device_conditions, device_choices, default='unknown'),
        index=df.index
    ).astype('category')

    platform = df['platform'].str.lower().replace('iphone', 'ios').fillna('')

    ua_platform = df['ua_os'].astype('string').map({
        'Android': 'android',
        'iOS': 'ios', 'iPadOS': 'ios',
        'Windows': 'desktop', 'macOS': 'desktop', 'Linux': 'desktop',
    }).fillna('')

    checkable = platform.isin(['android', 'ios', 'desktop'])

    df['ua_device_conflict'] = (
        checkable & (platform != ua_platform)
    ).astype('uint8')

    chrome_ver = df[col].str.extract(r'Chrome/(\d+)\.')[0]
    df['chrome_major_version'] = pd.to_numeric(chrome_ver, errors='coerce').fillna(0).astype(int)

    return df


def add_temporal_features(df: pd.DataFrame, col: str = 'event_ts') -> DataFrame:
    df[col] = pd.to_datetime(df[col])
    hour = df[col].dt.hour.astype('int16')
    angle = 2 * np.pi * hour / 24.0

    df['event_hour'] = hour
    df['hour_sin_1'] = np.sin(angle)
    df['hour_cos_1'] = np.cos(angle)
    df['hour_sin_2'] = np.sin(2 * angle)
    df['hour_cos_2'] = np.cos(2 * angle)
    df['hour_distance_to_23'] = np.minimum(np.abs(hour - 23), 24 - np.abs(hour - 23))
    df['is_night'] = hour.isin([0, 1, 2, 3, 4, 5, 6]).astype('uint8')
    df['is_evening'] = hour.isin([18, 19, 20, 21, 22, 23]).astype('uint8')
    df['is_late_evening'] = hour.isin([22, 23]).astype('uint8')
    return df


BASE_AGGREGATIONS = {
    'eid': 'count',
    'item_id': 'nunique',
    'item_category': 'nunique',
    'item_location': 'nunique',
    'search_query': 'nunique',
    'search_page': ['max', 'mean'],
    'entropy': ['mean', 'min', 'max'],
    'is_known_tool': 'max',
    'is_headless': 'max',
    'has_url': 'max',
    'starts_with_mozilla': 'mean',
    'is_app_header': 'max',
    'ua_device_conflict': 'max',
    'ua_length': ['mean', 'std'],
    'digit_ratio': 'mean',
    'chrome_major_version': ['min', 'max', 'nunique'],
    'time_diff': ['mean', 'std', 'min', 'median'],
    'gap_le_1s': 'mean',
    'gap_le_5s': 'mean',
    'gap_le_10s': 'mean',
    'gap_le_30s': 'mean',
    'gap_gt_30m': ['mean', 'sum'],
    'pointer_x': lambda x: x.notna().mean(),
    'hour_sin_1': 'mean',
    'hour_cos_1': 'mean',
    'hour_sin_2': 'mean',
    'hour_cos_2': 'mean',
    'hour_distance_to_23': ['mean', 'min', 'std'],
    'is_night': 'mean',
    'is_evening': 'mean',
    'is_late_evening': 'mean',
}

DIVERSITY_COLUMNS = ['item_category', 'item_location', 'item_id', 'search_query']
DYNAMICS_COLUMNS = ['item_category', 'item_location']
CAT_TEXT_COLUMNS = [
    'cat_platform_mode',
    'cat_event_name_mode',
    'cat_item_category_mode',
    'cat_item_location_mode',
    'cat_seller_type_mode',
    'text_user_agent',
    'text_search_queries',
]


def add_inter_event_features(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(['cookie_id', 'event_ts']).copy()
    ev['time_diff'] = ev.groupby('cookie_id')['event_ts'].diff().dt.total_seconds()
    ev['time_diff_rounded'] = ev['time_diff'].round()
    for seconds in (1, 5, 10, 30):
        ev[f'gap_le_{seconds}s'] = ev['time_diff'].le(seconds).astype(float)
    ev['gap_gt_30m'] = ev['time_diff'].gt(1800).astype(float)
    return ev


def aggregate_base_features(ev: pd.DataFrame) -> pd.DataFrame:
    features = ev.groupby('cookie_id').agg(BASE_AGGREGATIONS)
    features.columns = ['_'.join(column).strip('_') for column in features.columns]
    return features


def build_diversity_features(ev: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for column in DIVERSITY_COLUMNS:
        part = ev.groupby('cookie_id')[column].apply(distribution_summary).unstack()
        part.columns = [f'div_{column}_{metric}' for metric in part.columns]
        parts.append(part)
    return pd.concat(parts, axis=1)

def build_dynamics_features(ev: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for column in DYNAMICS_COLUMNS:
        part = ev.groupby('cookie_id')[column].apply(diversity_dynamics).unstack()
        part.columns = [f'dyn_{column}_{metric}' for metric in part.columns]
        parts.append(part)
    return pd.concat(parts, axis=1)

def diversity_dynamics(s: pd.Series) -> pd.Series:
    names = [
        'unique_first25_ratio',
        'unique_first50_ratio',
        'unique_first75_ratio',
        'novelty_first_half',
        'novelty_second_half',
        'novelty_delta',
        'cumulative_unique_auc',
        'half_jaccard',
        'switch_rate',
        'max_run_share',
    ]
    values = s.dropna().astype(str).to_numpy()
    n_values = len(values)
    if n_values == 0:
        return pd.Series(dict.fromkeys(names, 0))

    seen, cumulative, is_new = set(), [], []
    for value in values:
        new = value not in seen
        seen.add(value)
        is_new.append(float(new))
        cumulative.append(len(seen))

    final_unique = max(len(seen), 1)
    cuts = [max(1, int(np.ceil(n_values * quantile))) for quantile in (0.25, 0.50, 0.75)]
    half = max(1, n_values // 2)
    first, second = set(values[:half]), set(values[half:])
    union = first | second
    switches = np.mean(values[1:] != values[:-1]) if n_values > 1 else 0

    run_lengths, run = [], 1
    for index in range(1, n_values):
        if values[index] == values[index - 1]:
            run += 1
        else:
            run_lengths.append(run)
            run = 1
    run_lengths.append(run)

    novelty_first = np.mean(is_new[:half])
    novelty_second = np.mean(is_new[half:]) if half < n_values else 0
    return pd.Series({
        'unique_first25_ratio': len(set(values[:cuts[0]])) / final_unique,
        'unique_first50_ratio': len(set(values[:cuts[1]])) / final_unique,
        'unique_first75_ratio': len(set(values[:cuts[2]])) / final_unique,
        'novelty_first_half': novelty_first,
        'novelty_second_half': novelty_second,
        'novelty_delta': novelty_second - novelty_first,
        'cumulative_unique_auc': np.mean(cumulative) / final_unique,
        'half_jaccard': len(first & second) / max(len(union), 1),
        'switch_rate': switches,
        'max_run_share': max(run_lengths) / n_values,
    })





def build_cross_diversity(ev: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=features.index)
    pairs = [
        ('item_category', 'item_location', 'category_location'),
        ('item_category', 'item_id', 'category_item'),
        ('event_name', 'item_category', 'event_category'),
    ]
    for left, right, name in pairs:
        valid = ev.dropna(subset=[left, right])
        result[f'cross_{name}_nunique'] = valid.groupby('cookie_id')[[left, right]].apply(
            lambda frame: len(frame.drop_duplicates())
        )
    category_count = features.item_category_nunique.clip(lower=1)
    result['cross_items_per_category'] = features.item_id_nunique / category_count
    result['cross_locations_per_category'] = features.item_location_nunique / category_count
    return result


def build_gap_features(ev: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    gaps = ev.dropna(subset=['time_diff'])
    result = gaps.groupby('cookie_id').agg(
        time_diff_q10=('time_diff', lambda s: s.quantile(0.10)),
        time_diff_q25=('time_diff', lambda s: s.quantile(0.25)),
        time_diff_q75=('time_diff', lambda s: s.quantile(0.75)),
        time_diff_q90=('time_diff', lambda s: s.quantile(0.90)),
        time_diff_unique=('time_diff_rounded', 'nunique'),
        time_diff_count=('time_diff_rounded', 'size'),
    )
    result['time_diff_iqr'] = result.time_diff_q75 - result.time_diff_q25
    result['time_diff_repeat_ratio'] = 1 - result.time_diff_unique / result.time_diff_count.clip(lower=1)
    result['time_diff_cv'] = features.time_diff_std / features.time_diff_mean.clip(lower=1e-6)
    result['time_diff_burstiness'] = (features.time_diff_std - features.time_diff_mean) / (
        features.time_diff_std + features.time_diff_mean
    ).clip(lower=1e-6)
    result['session_count'] = 1 + features.gap_gt_30m_sum
    return result


def build_hour_profile(ev: pd.DataFrame) -> pd.DataFrame:
    profile = pd.crosstab(ev['cookie_id'], ev['event_hour'], normalize='index')
    profile = profile.reindex(columns=range(24), fill_value=0)
    profile.columns = [f'hour_share_{hour:02d}' for hour in profile.columns]
    values = profile.to_numpy()

    lag_columns = {}
    lag_summary = pd.DataFrame(index=profile.index)
    for lag in (1, 2, 3):
        difference = values - np.roll(values, lag, axis=1)
        for hour in range(24):
            lag_columns[f'hour_diff_lag{lag}_{hour:02d}'] = difference[:, hour]
        lag_summary[f'hour_diff_lag{lag}_mean_abs'] = np.abs(difference).mean(axis=1)
        lag_summary[f'hour_diff_lag{lag}_max_abs'] = np.abs(difference).max(axis=1)

    summary = pd.DataFrame(index=profile.index)
    summary['active_hour_count'] = (values > 0).sum(axis=1)
    summary['hour_share_max'] = values.max(axis=1)
    summary['hour_share_top3'] = np.sort(values, axis=1)[:, -3:].sum(axis=1)
    summary['hour_entropy'] = -(values * np.log(values + 1e-12)).sum(axis=1)
    summary['hour_uniform_l1'] = np.abs(values - 1 / 24).sum(axis=1)
    summary['hour_profile_roughness'] = np.abs(values - np.roll(values, 1, axis=1)).mean(axis=1)
    centered_hours = np.arange(24) - 11.5
    summary['hour_linear_slope'] = (
        (values - values.mean(axis=1, keepdims=True)) * centered_hours
    ).sum(axis=1) / np.square(centered_hours).sum()
    summary['night_share_00_05'] = values[:, 0:6].sum(axis=1)
    summary['morning_share_06_11'] = values[:, 6:12].sum(axis=1)
    summary['day_share_12_17'] = values[:, 12:18].sum(axis=1)
    summary['evening_share_18_23'] = values[:, 18:24].sum(axis=1)
    summary['late_minus_early_share'] = values[:, 18:24].sum(axis=1) - values[:, 0:6].sum(axis=1)
    return profile.join(pd.DataFrame(lag_columns, index=profile.index)).join(lag_summary).join(summary)


def mode_or_missing(s: pd.Series) -> str:
    mode = s.astype('string').fillna('__MISSING__').mode()
    return str(mode.iloc[0]) if len(mode) else '__MISSING__'


def query_text(s: pd.Series) -> str:
    values = s.dropna().astype(str).drop_duplicates().head(100)
    return ' '.join(values) if len(values) else '__EMPTY_QUERY__'


def build_categorical_features(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns = ['platform', 'event_name', 'item_category', 'item_location', 'seller_type', 'user_agent']
    names = [
        'cat_platform_mode',
        'cat_event_name_mode',
        'cat_item_category_mode',
        'cat_item_location_mode',
        'cat_seller_type_mode',
        'text_user_agent',
    ]
    categorical = ev.groupby('cookie_id')[columns].agg(mode_or_missing)
    categorical.columns = names
    queries = ev.groupby('cookie_id')['search_query'].agg(query_text).rename('text_search_queries')
    return categorical, queries


def merge_feature_blocks(meta: pd.DataFrame, blocks: list[pd.DataFrame | pd.Series]) -> pd.DataFrame:
    result = meta[['cookie_id', 'cookie_age_days']]
    for block in blocks:
        result = result.merge(block.reset_index(), on='cookie_id', how='left')
    return result


def sanitize_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    for column in CAT_TEXT_COLUMNS:
        features[column] = features[column].astype('object').fillna('__MISSING__').astype(str)
    numeric_columns = [
        column for column in features.columns if column not in CAT_TEXT_COLUMNS + ['cookie_id']
    ]
    features[numeric_columns] = features[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def extract_all_features(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta['cookie_age_days'] = (
        meta['window_start_ts'] - meta['cookie_created_at']
    ).dt.total_seconds() / 86400.0
    ev = add_inter_event_features(ev)
    base = aggregate_base_features(ev)
    categorical, queries = build_categorical_features(ev)
    event_frequency = pd.crosstab(ev['cookie_id'], ev['event_name'], normalize='index')
    event_frequency.columns = [f'freq_{event}' for event in event_frequency.columns]
    blocks = [
        base,
        build_diversity_features(ev),
        build_dynamics_features(ev),
        build_cross_diversity(ev, base),
        build_gap_features(ev, base),
        event_frequency,
        build_hour_profile(ev),
        categorical,
        queries,
    ]
    return sanitize_feature_frame(merge_feature_blocks(meta, blocks))
