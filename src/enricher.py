import os
import json
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = os.path.join("data", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "scene_summaries.json")


def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _get_chunk_hash(chunk):
    key_str = f"{chunk.get('movie', '')}_{chunk.get('start', '')}_{chunk.get('end', '')}_{chunk.get('text', '')}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def call_gemini_api(prompt, api_key):
    """Calls Gemini REST API directly using requests to avoid extra dependency issues."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 150
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        res_json = response.json()
        try:
            return res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            return None
    else:
        # Fallback to gemini-1.5-flash if 2.5 endpoint differs in some regions
        url_fallback = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        res_fb = requests.post(url_fallback, headers=headers, json=payload, timeout=30)
        if res_fb.status_code == 200:
            res_json = res_fb.json()
            try:
                return res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                return None
    return None


def generate_scene_summary(chunk_text, start_time, end_time, api_key=None):
    """
    Generates a 1-2 sentence scene context summary for a subtitle chunk.
    If no API key is provided, returns None or a basic heuristic summary.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    prompt = (
        f"You are analyzing subtitle dialogue from a movie scene between timestamp {start_time} and {end_time}.\n\n"
        f"Subtitle Dialogue:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
        f"Task: Provide a concise 1-2 sentence scene context summary describing what is happening, "
        f"who is involved (names if present in dialogue or inferred context), and key actions/events. "
        f"Do NOT include conversational filler, preamble, or quotes."
    )

    try:
        summary = call_gemini_api(prompt, api_key)
        return summary
    except Exception as e:
        print(f"LLM Enrichment Error: {e}")
        return None


def enrich_chunks(chunks, api_key=None, verbose=True):
    """
    Enriches a list of subtitle chunks with LLM-generated scene summaries.
    Uses local disk caching so each chunk is summarized only once.
    """
    cache = _load_cache()
    api_key = api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key and verbose:
        print("[Enricher] Warning: GEMINI_API_KEY not found in environment. Returning raw chunks without new LLM summaries.")

    enriched_chunks = []
    new_summaries_count = 0

    for i, chunk in enumerate(chunks):
        chunk_hash = _get_chunk_hash(chunk)
        chunk_copy = dict(chunk)

        if chunk_hash in cache:
            chunk_copy["summary"] = cache[chunk_hash]
        elif api_key:
            if verbose:
                print(f"[Enricher] Summarizing chunk {i+1}/{len(chunks)} ({chunk['start']} -> {chunk['end']})...")
            summary = generate_scene_summary(chunk["text"], chunk["start"], chunk["end"], api_key)
            if summary:
                cache[chunk_hash] = summary
                chunk_copy["summary"] = summary
                new_summaries_count += 1
            else:
                chunk_copy["summary"] = ""
        else:
            chunk_copy["summary"] = ""

        enriched_chunks.append(chunk_copy)

    if new_summaries_count > 0:
        _save_cache(cache)
        if verbose:
            print(f"[Enricher] Saved {new_summaries_count} new summaries to cache.")

    return enriched_chunks


if __name__ == "__main__":
    from parser import parse_srt
    from chunker import chunk_subtitles

    subtitles = parse_srt("data/movies/captain_america2.srt")
    chunks = chunk_subtitles(subtitles)

    print(f"Total chunks: {len(chunks)}")
    enriched = enrich_chunks(chunks[:5])

    for c in enriched:
        print("\n--- CHUNK ---")
        print(f"Time: {c['start']} -> {c['end']}")
        print(f"Summary: {c.get('summary', 'N/A')}")
        print(f"Text:\n{c['text'][:150]}...")
