"""
Extracts my gold standard venues
"""

import os
from dotenv import load_dotenv
import requests
from enum import Enum
import pandas as pd
import json
from datetime import datetime
from db import store_to_table
import time
load_dotenv()


def extract_date(mention: str) -> datetime:
    """extract date from mention string (YYYY-MM-DD at the end)"""
    try:
        date_str = mention.rsplit(' - ', 1)[-1].strip()
        return datetime.strptime(date_str, '%Y-%m-%d')
    except Exception:
        return datetime(1900, 1, 1)


def process_mentions(mentions_raw, n_old=3, n_recent=6, n_newest=4):
    """get sorted mentions + oldest/newest slices"""

    if not isinstance(mentions_raw, str) or not mentions_raw.strip():
        return {
            'sorted': [],
            'oldest': [],
            'recent': [],
            'newest': []
        }

    try:
        mentions_list = json.loads(mentions_raw)
    except json.JSONDecodeError:
        return {
            'sorted': [],
            'oldest': [],
            'recent': [],
            'newest': []
        }

    if not isinstance(mentions_list, list):
        return {
            'sorted': [],
            'oldest': [],
            'recent': [],
            'newest': []
        }

    # Sort chronologically (oldest to newest)
    sorted_mentions = sorted(mentions_list, key=extract_date)

    result = {
        'sorted': sorted_mentions,
        'oldest': sorted_mentions[:n_old],          # first 3 (oldest)
        'recent': sorted_mentions[-n_recent:],      # last 6 (most recent)
        # last 4 (newest, for Variant C)
        'newest': sorted_mentions[-n_newest:]
    }

    return result


def geocode_google(venue_name):
    """For each venue, extract their Coordinates"""
    API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY")

    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={venue_name}&key={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data['results']:
        location = data['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    return None, None


class Mode(Enum):
    """
    Differences between the two modes:
    - SRTM: Radar-based (radio waves measure bounce-back time), ±16m accuracy, coverage -60° to +60° latitude
    - ASTER: Optical-based (stereo photos calculate parallax - how things shift between viewpoints to measure height), ±17m accuracy, near-global coverage -83° to +83°
    """

    SRTM = "srtm30m"
    ASTER = "aster30m"


def elevation_data(lat, lon, mode: Mode = Mode.SRTM):
    """
    Fetch elevation from OpenTopography API using the specified dataset.
    Returns (elevation_value, full_url) or (None, None) if failed/missing.
    """
    if lat is None or lon is None:
        return None, None

    service = mode.value
    url = f"https://api.opentopodata.org/v1/{service}?locations={lat},{lon}"

    try:
        time.sleep(2)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'OK' and data.get('results'):
            elevation = data['results'][0].get('elevation')

            return elevation, url
        else:
            return None, url

    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Error fetching elevation from {url}: {e}")
        return None, url


def get_best_elevation(lat, srtm, aster):
    # No SRTM data? Use ASTER
    if srtm is None or srtm == '':
        return aster, 'ASTER_ONLY', None

    # No ASTER data? Use SRTM
    if aster is None or aster == '':
        return srtm, 'SRTM_ONLY', None

    diff = abs(srtm - aster)

    # High latitude? Prefer ASTER
    if abs(lat) > 55:
        return aster, 'ASTER_HIGH_LAT', diff

    # Good agreement? Use SRTM
    if diff <= 10:
        return srtm, 'SRTM_PREFERRED', diff

    # Large disagreement? Flag it
    if diff > 15:
        return (srtm + aster) / 2, 'MANUAL_CHECK', diff

    # Medium disagreement? Use average
    return (srtm + aster) / 2, 'CONSENSUS', diff


def select_gold_standard(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    main_candidates = df[df['mention_count'] >= 20].copy()

    # negative filters
    exclude_patterns = [
        # specialized throwing fields
        r"(?i)Throw Town|Millican|Ramona|Field at Throw",
        r"(?i)Training|Practice|Club|School|College|High School",  # too local
        # indoor venues often have different altitude logic
        r"(?i)Indoor|Hall|Sports Centre|Centre Sportif"
    ]

    exclude_mask = main_candidates['canonical_venue'].str.contains(
        '|'.join(exclude_patterns), regex=True, na=False)

    main_candidates = main_candidates[~exclude_mask]

    print("Number of selected venues:", len(main_candidates))

    gold_df = main_candidates.copy()

    # Extract oldest, recent and newest venues mentions
    gold_df['mention_data'] = gold_df['mentions'].apply(
        process_mentions)

    # flatten into separate columns for easier access
    gold_df['oldest_mentions'] = gold_df['mention_data'].apply(
        lambda x: x['oldest'])
    gold_df['recent_mentions'] = gold_df['mention_data'].apply(
        lambda x: x['recent'])
    gold_df['newest_mentions'] = gold_df['mention_data'].apply(
        lambda x: x['newest'])

    gold_df = gold_df.drop(columns=['mention_data'])

    results = extract_gold_standard_elevation(gold_df)

    return results


def extract_gold_standard_elevation(gold_df: pd.DataFrame) -> pd.DataFrame:
    df = gold_df.copy()
    results = []

    print("=" * 50)
    for idx, row in df.iterrows():
        venue = row['canonical_venue']
        print(f"{idx + 1}. Extracting coords for: {venue}")
        print("-" * 5)

        lat, lon = geocode_google(venue)

        if lat is None or lon is None:
            print(f"Geocoding failed for {venue}")
            elevation, reason, diff = None, 'NO_COORDS', None
            srtm_e = aster_e = None
            used_url = None
        else:
            # Fetch both datasets explicitly
            srtm_elevation, srtm_url = elevation_data(lat, lon, Mode.SRTM)
            aster_elevation, aster_url = elevation_data(lat, lon, Mode.ASTER)

            print(f"{venue} - Latitude: {lat}, Longitude: {lon}, "
                  f"SRTM Elevation: {srtm_elevation}, ASTER Elevation: {aster_elevation}")

            elevation, reason, diff = get_best_elevation(
                lat, srtm_elevation, aster_elevation)

            # For the result row, prefer ASTER url when that's what we chose
            used_url = aster_url if reason and 'ASTER' in reason else srtm_url

            srtm_e = srtm_elevation
            aster_e = aster_elevation

        results.append({
            "canonical_venue": venue,
            "latitude": float(lat) if lat is not None else None,
            "longitude": float(lon) if lon is not None else None,
            "elevation": float(elevation) if elevation is not None else None,
            "elevation_reason": reason,
            "url": used_url,
            "srtm_elevation": float(srtm_e) if srtm_e is not None else None,
            "aster_elevation": float(aster_e) if aster_e is not None else None,
            "diff": float(diff) if diff is not None else None
        })

        print("-" * 50)

    df_results = pd.DataFrame(results)
    df_out = df.merge(df_results, on="canonical_venue", how="left")

    # CSV backup
    output_path = "./data/gold_venues_with_elevation.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(f"Wrote {len(df_out)} rows with elevation data to {output_path}")

    # db storage
    table = "venues_gold_standard"

    cols = ["canonical_venue", "gold_altitude_m", "gold_confidence",
            "gold_source", "mention_count", "notes"]

    df_out['gold_altitude_m'] = df_out['elevation'] if 'elevation' in df_out.columns else -1

    def _map_conf(reason):
        if pd.isna(reason) or reason is None:
            return 'unknown'
        if reason in ('SRTM_ONLY', 'ASTER_ONLY', 'SRTM_PREFERRED', 'ASTER_HIGH_LAT'):
            return 'high'
        if reason == 'CONSENSUS':
            return 'medium'
        if reason == 'MANUAL_CHECK':
            return 'low'
        return 'unknown'

    if 'elevation_reason' in df_out.columns:
        df_out['gold_confidence'] = df_out['elevation_reason'].apply(_map_conf)
    else:
        df_out['gold_confidence'] = 'unknown'

    df_out['gold_source'] = df_out['url'] if 'url' in df_out.columns else None

    def _make_notes(row):
        """include SRTM/ASTER values and the elevation_reason for traceability"""
        parts = []
        if 'srtm_elevation' in row and pd.notnull(row['srtm_elevation']):
            parts.append(f"SRTM={row['srtm_elevation']}")
        if 'aster_elevation' in row and pd.notnull(row['aster_elevation']):
            parts.append(f"ASTER={row['aster_elevation']}")
        if 'elevation_reason' in row and row['elevation_reason']:
            parts.append(f"reason={row['elevation_reason']}")
        return '; '.join(parts)

    df_out['notes'] = df_out.apply(_make_notes, axis=1)

    store_to_table(table_name=table, columns=cols, values=df_out)

    return df_out
