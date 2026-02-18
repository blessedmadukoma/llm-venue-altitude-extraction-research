# Variant A: Short baseline (canonical name only — cheap/fast)
PROMPT_1_TEMPLATE_SHORT = """
You are a sports geography expert. Determine the best single altitude (meters above sea level) for this athletics venue across its history. If relocated/renovated, prefer the most recent/main configuration.

Venue: {canonical_venue}

CRITICAL RULES:
- altitude_meters MUST be an integer. If uncertain, use -1 instead of guessing.
- Prioritize official stadium/venue elevation over city average
- For indoor venues, use building/floor elevation
- Consider performance impact (e.g., Mexico City ~2240m)

Response format (JSON object, no extra text):
{{
    "canonical_venue": "{canonical_venue}",
    "altitude_meters": 134,
    "confidence": "High" or "Medium" or "Low",
    "source": "Brief reasoning/source"
}}
"""

# Variant B: With recent mentions (adds context without overload)
PROMPT_2_TEMPLATE_RECENT = """
You are a sports geography expert. Provide ONE best altitude (meters above sea level) representing this venue across uses. Prefer most recent/common if relocated/renovated.

Canonical venue: {canonical_venue}

Recent/important competitions: {recent_mentions}

CRITICAL RULES:
- altitude_meters MUST be an integer. If uncertain, use -1 instead of guessing.
- Prioritize official stadium/venue elevation over city average
- For indoor venues, use building/floor elevation
- Historical context: some venues renovated/relocated

Response format (JSON object, no extra text):
{{
"canonical_venue": "{canonical_venue}",
"altitude_meters": 134,
"confidence": "High" or "Medium" or "Low",
"source": "Reasoning/source, note any changes"
}}
"""

# Variant C: Old + new mentions (includes relocation detection)
PROMPT_3_TEMPLATE_OLD_NEW = """
You are a sports geography expert. Determine ONE representative altitude (meters above sea level) for this venue. If mentions show relocation/major change, prefer most recent and note in source.

Canonical venue: {canonical_venue}

Oldest competitions: {oldest_mentions}
Newest competitions: {newest_mentions}

CRITICAL RULES:
- altitude_meters MUST be an integer. If uncertain, use -1 instead of guessing.
- Prioritize official stadium/venue elevation over city average
- For indoor venues, use building/floor elevation
- Altitude affects performance (e.g., high altitude like Nairobi ~1795m)

Response format (JSON object, no extra text):
{{
"canonical_venue": "{canonical_venue}",
"altitude_meters": 134,
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
  Altitude: -1, Confidence: Low, Source: No reliable data found

Now for: {canonical_venue}
Mentions: {recent_mentions}

CRITICAL RULES:
- altitude_meters MUST be an integer. If uncertain, use -1 instead of guessing.
- Prioritize official stadium/venue elevation over city average
- For indoor venues, use building/floor elevation
- Altitude affects performance (e.g., high altitude like Nairobi ~1795m)

Response format (JSON object, no extra text):
{{
"canonical_venue": "{canonical_venue}",
"altitude_meters": 134,
"confidence": "High" or "Medium" or "Low",
"source": "Reasoning/source, including any historical changes"
}}
"""

# Bulk inference: raw crawled venue string → canonical name + altitude
# Used for the ~4k unmatched competition venues in venue_altitudes pipeline.
# Returns -1 for unknown so altitude_m is always a parseable integer.
PROMPT_BULK_TEMPLATE = """
You are a sports geography expert specialising in athletics venues worldwide.

Given a raw venue string scraped from a competition database, you must:
1. Produce a normalised canonical venue name (Title Case, no parenthetical country codes, no special characters)
2. Estimate the altitude in metres above sea level

Examples:
- Raw: "Hayward Field, Eugene (USA)"
  canonical_venue: "Hayward Field, Eugene", altitude_meters: 134, confidence: "high"
- Raw: "Estadio Olímpico Universitario, Ciudad de México (MEX)"
  canonical_venue: "Estadio Olimpico Universitario, Mexico City", altitude_meters: 2240, confidence: "high"
- Raw: "Stade du Roi Baudouin, Bruxelles (BEL)"
  canonical_venue: "Stade du Roi Baudouin, Brussels", altitude_meters: 15, confidence: "high"
- Raw: "Some Unknown Track, Smallville (XYZ)"
  canonical_venue: "Unknown Track, Smallville", altitude_meters: -1, confidence: "low"

Venue to process: {raw_venue}

CRITICAL RULES:
- altitude_meters MUST be an integer. NEVER a string or null.
- If altitude cannot be determined with reasonable confidence, use -1. Do NOT guess.
- Prioritise official stadium/venue elevation over city average elevation.
- For indoor venues, use the building/floor elevation.

Respond with ONLY a JSON object, no extra text or markdown fences:
{{
    "raw_venue_string": "{raw_venue}",
    "canonical_venue": "Normalised venue name",
    "altitude_meters": 134,
    "confidence": "high",
    "reasoning": "Brief source or reasoning"
}}
"""
