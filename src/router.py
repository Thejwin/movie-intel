import re
import os
from enricher import call_gemini_api


class AgenticRouter:
    """
    Agentic Decision Making & Routing System:
    1. Informational Queries -> RAG Engine
    2. Action / Email Requests -> RAG Engine + MCP Email Tool
    3. Ambiguous Requests -> Ask user for clarification
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def route_request(self, user_prompt, conversation_context=None):
        prompt_lower = user_prompt.lower().strip()

        # Check for Email / Action intent
        email_keywords = ["email", "send an email", "send email", "mail", "dispatch email", "share via email"]
        is_email_action = any(kw in prompt_lower for kw in email_keywords)

        # Regex for email extraction
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_prompt)
        recipient_email = email_match.group(0) if email_match else None

        if is_email_action:
            if not recipient_email:
                return {
                    "type": "AMBIGUOUS",
                    "reason": "Missing recipient email address",
                    "clarification_needed": "I notice you'd like to send an email, but no recipient email address was provided. Could you please specify who to send it to? (e.g., user@example.com)"
                }
            
            # Check if query is too vague (e.g. "send email to john@x.com" without topic/query)
            clean_query = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", user_prompt)
            clean_query = re.sub(r"\b(email|send|an|to|share|mail|dispatch)\b", "", clean_query, flags=re.IGNORECASE).strip()

            if len(clean_query) < 4:
                return {
                    "type": "AMBIGUOUS",
                    "reason": "Vague email request topic",
                    "clarification_needed": "Could you please specify which movie topic, dialogue, or scene breakdown you would like included in the email?"
                }

            return {
                "type": "ACTION_EMAIL",
                "recipient": recipient_email,
                "query": clean_query,
                "raw_prompt": user_prompt
            }

        # Check for ambiguous informational queries
        vague_phrases = ["tell me about that scene", "what did he say", "send that quote", "show me that movie"]
        if prompt_lower in vague_phrases or len(prompt_lower.split()) <= 2 and prompt_lower not in ["hi", "hello"]:
            return {
                "type": "AMBIGUOUS",
                "reason": "Vague or underspecified query",
                "clarification_needed": "Could you provide a few more details or keywords about the movie, character, or dialogue you are inquiring about?"
            }

        return {
            "type": "INFORMATIONAL",
            "query": user_prompt
        }


if __name__ == "__main__":
    router = AgenticRouter()
    
    test_prompts = [
        "Who does Steve Rogers meet while jogging in Washington DC?",
        "Send an email to director@marvel.com",
        "Email the morning run scene breakdown to producer@studio.com",
        "what did he say"
    ]

    for p in test_prompts:
        print(f"\nPrompt: '{p}'")
        decision = router.route_request(p)
        print(f"Decision: {decision}")
