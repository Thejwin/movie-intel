import srt
from pathlib import Path

def format_timestamp(timestamp):
    # Human readable timestamps
    total_seconds = timestamp.total_seconds()

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(timestamp.microseconds / 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def parse_srt(file_path):
    movie_title = Path(file_path).stem

    with open(file_path, "r", encoding="utf-8-sig") as file:
        content = file.read()

    subtitles = list(srt.parse(content))

    results = []

    for subtitle in subtitles:
        results.append({
            "movie": movie_title,
            "start": subtitle.start,
            "end": subtitle.end,
            "text": subtitle.content
        })

    return results


if __name__ == "__main__":
    file_path = Path("data/movies/captain_america2.srt")

    subtitles = parse_srt(file_path)

    print(f"Parsed {len(subtitles)} subtitle entries.")

    for subtitle in subtitles[:5]:
        print(
        subtitle["movie"],
        format_timestamp(subtitle["start"]),
        "→",
        format_timestamp(subtitle["end"]),
        subtitle["text"]
        )