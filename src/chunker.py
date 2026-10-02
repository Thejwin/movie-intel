from datetime import timedelta


def chunk_subtitles(subtitles, max_duration=30):
    chunks = []
    current_entries = []

    for subtitle in subtitles:
        current_entries.append(subtitle)

        start = current_entries[0]["start"]
        end = subtitle["end"]

        duration = end - start

        if duration.total_seconds() >= max_duration:
            chunks.append(create_chunk(current_entries))
            current_entries = []

    if current_entries:
        chunks.append(create_chunk(current_entries))

    return chunks


def create_chunk(entries):
    return {
        "movie": entries[0]["movie"],
        "start": entries[0]["start"],
        "end": entries[-1]["end"],
        "text": "\n".join(entry["text"] for entry in entries)
    }

# Test script

if __name__ == "__main__":
    from parser import parse_srt

    file_path = "data/movies/captain_america2.srt"

    subtitles = parse_srt(file_path)
    chunks = chunk_subtitles(subtitles)

    print(f"Subtitle entries: {len(subtitles)}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n--- CHUNK ---")
        print(chunk["movie"])
        print(chunk["start"], "→", chunk["end"])
        print(chunk["text"])