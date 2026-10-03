import os
from rag_engine import MovieRAGEngine
from mcp_email_tool import MCPEmailTool
from router import AgenticRouter


class MovieIntelAgent:
    def __init__(self, movies_dir="data/movies", api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.rag_engine = MovieRAGEngine(movies_dir=movies_dir, api_key=self.api_key)
        self.mcp_email_tool = MCPEmailTool()
        self.router = AgenticRouter(api_key=self.api_key)
        self.is_initialized = False

    def initialize(self, verbose=True):
        if not self.is_initialized:
            self.rag_engine.ingest_and_index(verbose=verbose)
            self.is_initialized = True

    def process_request(self, user_input):
        self.initialize(verbose=False)

        route = self.router.route_request(user_input)
        route_type = route["type"]

        if route_type == "AMBIGUOUS":
            return {
                "status": "CLARIFICATION_NEEDED",
                "route_type": route_type,
                "message": route["clarification_needed"],
                "citations": []
            }

        elif route_type == "INFORMATIONAL":
            rag_result = self.rag_engine.answer_query(route["query"])
            return {
                "status": "SUCCESS",
                "route_type": route_type,
                "answer": rag_result["answer"],
                "citations": rag_result["citations"],
                "sources": rag_result["sources"]
            }

        elif route_type == "ACTION_EMAIL":
            recipient = route["recipient"]
            query = route["query"]

            # First retrieve movie information using RAG Engine
            rag_result = self.rag_engine.answer_query(query)
            answer = rag_result["answer"]
            citations = rag_result["citations"]

            subject = f"Movie Intel Scene Breakdown: {query[:40]}"
            body = (
                f"Hello,\n\n"
                f"Here is the requested movie scene breakdown:\n\n"
                f"{answer}\n\n"
                f"--- Citations ---\n"
            )
            for c in citations:
                body += f"- Movie: {c['movie']} | Timestamp: {c['start']} -> {c['end']}\n"

            # Dispatch via MCP Email Tool
            mcp_response = self.mcp_email_tool.execute(
                recipient=recipient,
                subject=subject,
                body=body,
                citations=citations
            )

            return {
                "status": "SUCCESS",
                "route_type": route_type,
                "answer": f"**[MCP Action Executed]** {mcp_response['message']}\n\n**Email Subject:** {subject}\n\n**Content Sent:**\n{answer}",
                "citations": citations,
                "mcp_dispatch": mcp_response
            }

        return {
            "status": "ERROR",
            "message": "Unknown route type."
        }


if __name__ == "__main__":
    agent = MovieIntelAgent()
    agent.initialize()

    print("\n--- TEST 1: Informational ---")
    res1 = agent.process_request("Who does Steve Rogers meet while running?")
    print("Answer:\n", res1["answer"])

    print("\n--- TEST 2: Email Action ---")
    res2 = agent.process_request("Email the morning run scene breakdown to producer@marvel.com")
    print("Answer:\n", res2["answer"])

    print("\n--- TEST 3: Ambiguous ---")
    res3 = agent.process_request("Send an email")
    print("Answer:\n", res3["message"])
