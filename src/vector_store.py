import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def strip_html_tags(text):
    """Remove HTML/subtitle formatting tags like <i>, </i>, etc."""
    return re.sub(r"<[^>]+>", "", text)


def tokenize(text):
    """Lowercase, strip punctuation, and split text into tokens for BM25."""
    cleaned = strip_html_tags(text)
    tokens = cleaned.lower().split()
    tokens = [re.sub(r'[^\w]', '', t) for t in tokens]
    return [t for t in tokens if t]


def create_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="chroma_db")

    try:
        client.delete_collection(name="movie_subtitles")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="movie_subtitles",
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    for chunk in chunks:
        summary = chunk.get("summary", "")
        if summary:
            doc_text = f"[Scene Summary]\n{summary}\n\n[Dialogue]\n{chunk['text']}"
        else:
            doc_text = chunk["text"]
        documents.append(doc_text)

    metadatas = [
        {
            "movie": chunk["movie"],
            "start": str(chunk["start"]),
            "end": str(chunk["end"]),
            "summary": chunk.get("summary", "")
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

    # Build BM25 index from the same documents
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return collection, model, bm25, documents, metadatas


def hybrid_search(query, collection, model, bm25, documents, metadatas, n_results=5, k=60):
    """
    Hybrid search combining BM25 keyword matching and vector similarity.

    Uses Reciprocal Rank Fusion (RRF) to merge both rankings.
    k=60 is the standard RRF constant that controls how much weight
    lower-ranked results get (higher k = more even weighting).
    """

    # --- Vector search ---
    query_embedding = model.encode([query]).tolist()
    try:
        vector_results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(len(documents), 50),
            include=["documents", "metadatas", "distances"]
        )
    except Exception:
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection(name="movie_subtitles")
        vector_results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(len(documents), 50),
            include=["documents", "metadatas", "distances"]
        )

    # Build vector ranking: chunk_id -> rank (0-indexed)
    vector_ranks = {}
    for rank, doc_id in enumerate(vector_results["ids"][0]):
        vector_ranks[doc_id] = rank

    # --- BM25 search ---
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    # Rank all documents by BM25 score (highest first)
    bm25_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    # Build BM25 ranking: chunk_id -> rank (0-indexed)
    bm25_ranks = {}
    for rank, doc_index in enumerate(bm25_ranking):
        chunk_id = f"chunk_{doc_index}"
        bm25_ranks[chunk_id] = rank

    # --- Reciprocal Rank Fusion ---
    all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores = {}

    for chunk_id in all_chunk_ids:
        vector_rank = vector_ranks.get(chunk_id, len(documents))
        bm25_rank = bm25_ranks.get(chunk_id, len(documents))

        # RRF formula: score = 1/(k + rank) for each system, then sum
        rrf_scores[chunk_id] = (1 / (k + vector_rank)) + (1 / (k + bm25_rank))

    # Sort by RRF score (highest = most relevant)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final results
    final_results = []
    for chunk_id, rrf_score in sorted_results[:n_results]:
        doc_index = int(chunk_id.split("_")[1])
        final_results.append({
            "chunk_id": chunk_id,
            "document": documents[doc_index],
            "metadata": metadatas[doc_index],
            "rrf_score": round(rrf_score, 6),
            "vector_rank": vector_ranks.get(chunk_id, None),
            "bm25_rank": bm25_ranks.get(chunk_id, None),
            "bm25_score": round(bm25_scores[doc_index], 4)
        })

    return final_results


# Test script

if __name__ == "__main__":
    from parser import parse_srt
    from chunker import chunk_subtitles
    from enricher import enrich_chunks

    file_path = "data/movies/captain_america2.srt"

    subtitles = parse_srt(file_path)
    chunks = chunk_subtitles(subtitles)
    chunks = enrich_chunks(chunks)

    collection, model, bm25, documents, metadatas = create_vector_store(chunks)

    questions = [
        "Who does Steve meet while running?",
        "Who is Sam Wilson?",
        "What happens when Steve goes running?",
        "Who does Steve meet during his morning run?"
    ]

    for question in questions:
        print(f"\n\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        results = hybrid_search(
            question, collection, model, bm25, documents, metadatas,
            n_results=5
        )

        for i, result in enumerate(results):
            print(f"\n--- RESULT {i+1} ---")
            print(f"RRF Score:    {result['rrf_score']}")
            print(f"Vector Rank:  {result['vector_rank']}  |  BM25 Rank: {result['bm25_rank']}  |  BM25 Score: {result['bm25_score']}")
            print(f"Movie:        {result['metadata']['movie']}")
            print(f"Time:         {result['metadata']['start']} -> {result['metadata']['end']}")
            print(f"Text:         {strip_html_tags(result['document'][:200])}")