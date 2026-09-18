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
    """Добавляет признаки времени события без использования target.

    Час считается циклическим: расстояние от 23:00 до 00:00 равно одному часу.
    Часовой профиль cookie и его лаги агрегируются позже на уровне cookie.
    """
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
