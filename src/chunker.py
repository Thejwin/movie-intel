from datetime import timedelta


def chunk_subtitles(subtitles, max_duration=45, overlap_duration=15):
    chunks = []
    if not subtitles:
        return chunks

    i = 0
    n = len(subtitles)

    while i < n:
        chunk_entries = []
        start_time = subtitles[i]["start"]

        j = i
        while j < n:
            chunk_entries.append(subtitles[j])
            duration = (subtitles[j]["end"] - start_time).total_seconds()
            if duration >= max_duration:
                break
            j += 1

        chunks.append(create_chunk(chunk_entries))

        # Advance i such that there is overlap
        next_i = i + 1
        while next_i < j:
            if (subtitles[next_i]["start"] - start_time).total_seconds() >= (max_duration - overlap_duration):
                break
            next_i += 1

        if next_i <= i:
            next_i = i + 1

        i = next_i

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