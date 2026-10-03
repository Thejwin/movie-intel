import chromadb
from sentence_transformers import SentenceTransformer


def create_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="chroma_db")

    try:
        client.delete_collection(name="movie_subtitles")
    except Exception:
        pass

    collection = client.create_collection(
        name="movie_subtitles",
        metadata={"hnsw:space": "cosine"}
    )

    documents = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "movie": chunk["movie"],
            "start": str(chunk["start"]),
            "end": str(chunk["end"])
        }
        for chunk in chunks
    ]

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    embeddings = model.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return collection, model

# Test script

if __name__ == "__main__":
    from parser import parse_srt
    from chunker import chunk_subtitles

    file_path = "data/movies/captain_america2.srt"

    subtitles = parse_srt(file_path)
    chunks = chunk_subtitles(subtitles)

    collection, model = create_vector_store(chunks)

    questions = [
    "Who does Steve meet while running?",
    "Who is Sam Wilson?",
    "What happens when Steve goes running?",
    "Who does Steve meet during his morning run?"
]

for question in questions:
    print(f"\n\nQUESTION: {question}")

    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print("\n--- RESULT ---")
        print("Distance:", round(distance, 4))
        print(metadata["movie"])
        print(metadata["start"], "->", metadata["end"])
        print(document[:300])



    # question = "Who does Steve meet while running?"

    # query_embedding = model.encode([question]).tolist()

    # results = collection.query(
    #     query_embeddings=query_embedding,
    #     n_results=3
    # )

    # for i, document in enumerate(results["documents"][0]):
    #     metadata = results["metadatas"][0][i]

    #     print("\n--- RESULT ---")
    #     print(metadata["movie"])
    #     print(metadata["start"], "→", metadata["end"])
    #     print(document)



    # print("\n\nSEARCHING FOR SAM WILSON...")

    # for i, chunk in enumerate(chunks):
    #     if "Sam Wilson" in chunk["text"]:
    #         print("\n--- MATCH ---")
    #         print("Chunk:", i)
    #         print(chunk["start"], "→", chunk["end"])
    #         print(chunk["text"])