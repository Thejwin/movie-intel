import os
from parser import parse_srt
from chunker import chunk_subtitles
from enricher import enrich_chunks, call_gemini_api
from vector_store import create_vector_store, hybrid_search


class MovieRAGEngine:
    def __init__(self, movies_dir="data/movies", api_key=None):
        self.movies_dir = movies_dir
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.chunks = []
        self.collection = None
        self.model = None
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        self.is_indexed = False

    def ingest_and_index(self, verbose=True):
        if verbose:
            print(f"[RAG Engine] Scanning SRT files in '{self.movies_dir}'...")

        all_chunks = []

        if os.path.exists(self.movies_dir):
            for filename in os.listdir(self.movies_dir):
                if filename.endswith(".srt"):
                    filepath = os.path.join(self.movies_dir, filename)
                    if verbose:
                        print(f"[RAG Engine] Parsing {filename}...")
                    subtitles = parse_srt(filepath)
                    chunks = chunk_subtitles(subtitles)
                    all_chunks.extend(chunks)

        if not all_chunks:
            if verbose:
                print("[RAG Engine] No subtitle chunks found!")
            return

        if verbose:
            print(f"[RAG Engine] Enriching {len(all_chunks)} chunks with scene summaries...")
        enriched_chunks = enrich_chunks(all_chunks, api_key=self.api_key, verbose=verbose)
        self.chunks = enriched_chunks

        if verbose:
            print("[RAG Engine] Creating vector store & BM25 index...")
        self.collection, self.model, self.bm25, self.documents, self.metadatas = create_vector_store(enriched_chunks)
        self.is_indexed = True

        if verbose:
            print(f"[RAG Engine] Indexing complete! Total chunks indexed: {len(enriched_chunks)}")

    def answer_query(self, query, top_k=4):
        if not self.is_indexed:
            self.ingest_and_index()

        results = hybrid_search(query, self.collection, self.model, self.bm25, self.documents, self.metadatas, n_results=top_k)

        # Build context string and citations list
        context_blocks = []
        citations = []

        for i, res in enumerate(results):
            meta = res["metadata"]
            citation = {
                "movie": meta.get("movie", "Unknown Movie"),
                "start": meta.get("start", "00:00:00"),
                "end": meta.get("end", "00:00:00"),
                "summary": meta.get("summary", ""),
                "dialogue_snippet": res["document"][:200]
            }
            citations.append(citation)

            block = (
                f"--- SOURCE {i+1} ---\n"
                f"Movie: {citation['movie']}\n"
                f"Timestamp: {citation['start']} -> {citation['end']}\n"
                f"Content:\n{res['document']}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

        # Generate grounded answer using Gemini LLM if API key available, else synthetic synthesis
        if self.api_key:
            prompt = (
                f"You are a Movie Intelligence RAG Assistant. Answer the user's question accurately using ONLY "
                f"the provided movie subtitle and scene summary sources.\n\n"
                f"Question: {query}\n\n"
                f"Context Sources:\n{context_str}\n\n"
                f"Instructions:\n"
                f"1. Provide a direct, informative answer.\n"
                f"2. Cite your sources clearly using [Movie Title, Timestamp Start -> End].\n"
                f"3. Do not make up facts outside the provided context."
            )
            answer = call_gemini_api(prompt, self.api_key)
            if not answer:
                answer = self._fallback_answer(query, citations)
        else:
            answer = self._fallback_answer(query, citations)

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "sources": results
        }

    def _fallback_answer(self, query, citations):
        if not citations:
            return "No relevant movie scenes found for your query."

        top = citations[0]
        summary_text = f" Summary: '{top['summary']}'" if top['summary'] else ""
        return (
            f"Based on retrieved movie data for '{top['movie']}' between {top['start']} and {top['end']}:{summary_text}\n"
            f"Key dialogue snippet:\n\"{top['dialogue_snippet'].strip()}...\"\n\n"
            f"[Source Citation: {top['movie']} ({top['start']} -> {top['end']})]"
        )


if __name__ == "__main__":
    engine = MovieRAGEngine()
    engine.ingest_and_index()
    res = engine.answer_query("Who does Steve meet during his morning run?")
    print("\n\n" + "="*50)
    print("ANSWER:")
    print(res["answer"])
    print("\nCITATIONS:")
    for c in res["citations"]:
        print(f"- {c['movie']} [{c['start']} -> {c['end']}]")
