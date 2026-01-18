# Variant A: Short baseline (canonical name only — cheap/fast)
PROMPT_1_TEMPLATE_SHORT = """
You are a sports geography expert. Determine the best single altitude (meters above sea level) for this athletics venue across its history. If relocated/renovated, prefer the most recent/main configuration.

Venue: {canonical_venue}

CRITICAL RULES:
- If uncertain, use "Unknown" for altitude_meters rather than guessing
- Prioritize official stadium/venue elevation over city average
- For indoor venues, use building/floor elevation
- Consider performance impact (e.g., Mexico City ~2240m)

Response format (JSON object, no extra text):
{{
    "canonical_venue": "{canonical_venue}",
    "altitude_meters": 134 or "Unknown",
    "confidence": "High" or "Medium" or "Low",
    "source": "Brief reasoning/source"
}}
"""

# Variant B: With recent mentions (sweet spot — adds context without overload)
PROMPT_2_TEMPLATE_RECENT = """
You are a sports geography expert. Provide ONE best altitude (meters above sea level) representing this venue across uses. Prefer most recent/common if relocated/renovated.

Canonical venue: {canonical_venue}

Recent/important competitions: {recent_mentions}

CRITICAL RULES:
- If uncertain, use "Unknown" for altitude_meters rather than guessing
- Prioritize official stadium/venue elevation over city average
- For indoor venues, use building/floor elevation
- Historical context: some venues renovated/relocated

Response format (JSON object, no extra text):
{{
"canonical_venue": "{canonical_venue}",
"altitude_meters": 134 or "Unknown",
"confidence": "High" or "Medium" or "Low",
"source": "Reasoning/source, note any changes"
}}
"""

# Variant C: Old + new mentions (for relocation detection)
PROMPT_3_TEMPLATE_OLD_NEW = """
You are a sports geography expert. Determine ONE representative altitude (meters above sea level) for this venue. If mentions show relocation/major change, prefer most recent and note in source.

Canonical venue: {canonical_venue}

Oldest competitions: {oldest_mentions}
Newest competitions: {newest_mentions}

CRITICAL RULES:
- If uncertain, use "Unknown" for altitude_meters rather than guessing
- Prioritize official stadium/venue elevation over city average
- For indoor venues, use building/floor elevation
- Altitude affects performance (e.g., high altitude like Nairobi ~1795m)

Response format (JSON object, no extra text):
{{
"canonical_venue": "{canonical_venue}",
"altitude_meters": 134 or "Unknown",
"confidence": "High" or "Medium" or "Low",
"source": "Reasoning/source, including any historical changes"
}}
"""

# Variant D: Few-shot (adds examples — for better consistency)
PROMPT_4_TEMPLATE_FEW_SHOT = """
You are a sports geography expert. Provide ONE best altitude (meters above sea level) for each venue. Prefer recent if changed.

Examples:
- Venue: Hayward Field, Eugene (USA)
  Altitude: 134, Confidence: High, Source: Official after 2020 renovation, USGS verified
- Venue: Estadio Olímpico Universitario, Mexico City (MEX)
  Altitude: 2240, Confidence: High, Source: High-altitude venue, consistent across Olympics/competitions
- Venue: Unknown Local Track, Small Town (XYZ)
  Altitude: Unknown, Confidence: Low, Source: No reliable data found

Now for: {canonical_venue}
Mentions: {recent_mentions}

CRITICAL RULES:
- If uncertain, use "Unknown" for altitude_meters rather than guessing
- Prioritize official stadium/venue elevation over city average
- For indoor venues, use building/floor elevation
- Altitude affects performance (e.g., high altitude like Nairobi ~1795m)

Response format (JSON object, no extra text):
{{
"canonical_venue": "{canonical_venue}",
"altitude_meters": 134 or "Unknown",
"confidence": "High" or "Medium" or "Low",
"source": "Reasoning/source, including any historical changes"
}}
"""
