import re
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enricher import call_gemini_api


class AgenticRouter:
    """
    Dynamic Agentic Decision Making & Routing System:
    1. INFORMATIONAL Queries -> RAG Engine
    2. ACTION_EMAIL Requests -> RAG Engine + MCP Email Tool
    3. AMBIGUOUS Requests -> Ask user for clarification

    Uses LLM classification when API key is available, with a robust pattern fallback.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def route_request(self, user_prompt, conversation_context=None):
        # Normalize whitespace (collapse multiple spaces, tabs, newlines)
        normalized_prompt = " ".join(user_prompt.split()).strip()

        # Try LLM-based dynamic classification if API key is present
        if self.api_key:
            llm_decision = self._classify_with_llm(normalized_prompt)
            if llm_decision:
                return llm_decision

        # Fallback to robust dynamic rule/pattern classifier
        return self._rule_fallback_classify(normalized_prompt)

    def _classify_with_llm(self, prompt):
        system_prompt = (
            "You are an AI Agent Decision Router for a Movie Intelligence Assistant.\n"
            "Classify the user input into ONE of three categories:\n"
            "1. 'INFORMATIONAL': Questions about movie plots, dialogues, scenes, or characters with sufficient detail to search.\n"
            "2. 'ACTION_EMAIL': Requests to email/send a scene breakdown, quote, or dialogue summary TO a specific recipient email address.\n"
            "3. 'AMBIGUOUS': Requests that are vague, missing context (e.g. 'tell me about that scene', 'what did he say', 'show that movie'), or email requests missing recipient email address or missing topic.\n\n"
            "Respond ONLY with a valid JSON object formatted as follows:\n"
            "{\n"
            '  "type": "INFORMATIONAL" | "ACTION_EMAIL" | "AMBIGUOUS",\n'
            '  "query": "cleaned query string",\n'
            '  "recipient": "email address or null",\n'
            '  "clarification_needed": "clarification question if AMBIGUOUS or null"\n'
            "}\n\n"
            f"User Input: \"{prompt}\""
        )
        
        raw_res = call_gemini_api(system_prompt, self.api_key)
        if raw_res:
            try:
                # Extract JSON block
                json_str = raw_res.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                parsed = json.loads(json_str)
                rtype = parsed.get("type")

                if rtype == "AMBIGUOUS":
                    return {
                        "type": "AMBIGUOUS",
                        "reason": "Vague or missing information detected by LLM router",
                        "clarification_needed": parsed.get("clarification_needed") or "Could you please specify more details about the movie, character, or recipient email?"
                    }
                elif rtype == "ACTION_EMAIL":
                    return {
                        "type": "ACTION_EMAIL",
                        "recipient": parsed.get("recipient"),
                        "query": parsed.get("query") or prompt,
                        "raw_prompt": prompt
                    }
                elif rtype == "INFORMATIONAL":
                    return {
                        "type": "INFORMATIONAL",
                        "query": parsed.get("query") or prompt
                    }
            except Exception:
                pass

        return None

    def _rule_fallback_classify(self, prompt):
        prompt_lower = prompt.lower()

        # Extract email address via regex
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", prompt)
        recipient_email = email_match.group(0) if email_match else None

        # Check for email action keywords
        email_keywords = ["email", "send email", "send an email", "mail", "dispatch email", "share via email"]
        is_email_action = any(re.search(r"\b" + re.escape(kw) + r"\b", prompt_lower) for kw in email_keywords)

        if is_email_action:
            if not recipient_email:
                return {
                    "type": "AMBIGUOUS",
                    "reason": "Missing recipient email address",
                    "clarification_needed": "I notice you'd like to send an email, but no recipient email address was provided. Could you please specify who to send it to? (e.g., user@example.com)"
                }

            clean_query = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", prompt)
            clean_query = re.sub(r"\b(email|send|an|to|share|mail|dispatch)\b", "", clean_query, flags=re.IGNORECASE).strip()
            clean_query = " ".join(clean_query.split())

            if len(clean_query) < 3:
                return {
                    "type": "AMBIGUOUS",
                    "reason": "Vague email request topic",
                    "clarification_needed": "Could you please specify which movie topic, dialogue, or scene breakdown you would like included in the email?"
                }

            return {
                "type": "ACTION_EMAIL",
                "recipient": recipient_email,
                "query": clean_query,
                "raw_prompt": prompt
            }

        # Dynamic check for vague demonstrative phrases ("that scene", "that quote", "that movie", "what did he say")
        vague_patterns = [
            r"\b(tell me|show me|what about|explain)\s+(about\s+)?(that|this|the)\s+(scene|movie|quote|part|dialogue|clip)\b",
            r"\bwhat\s+did\s+(he|she|they|it)\s+(say|do)\b",
            r"\bsend\s+(that|this)\s+(quote|scene|dialogue)\b"
        ]

        has_proper_nouns = bool(re.search(r"\b[A-Z][a-z]+\b", prompt))
        is_vague_demonstrative = any(re.search(pat, prompt_lower) for pat in vague_patterns) and not has_proper_nouns

        tokens = prompt_lower.split()
        if is_vague_demonstrative or (len(tokens) <= 2 and prompt_lower not in ["hi", "hello"]):
            return {
                "type": "AMBIGUOUS",
                "reason": "Vague or underspecified query",
                "clarification_needed": "Could you provide a few more details or keywords about the movie title, character, or specific scene you are inquiring about?"
            }

        return {
            "type": "INFORMATIONAL",
            "query": prompt
        }


if __name__ == "__main__":
    router = AgenticRouter()

    test_prompts = [
        "tell me  about   that scene ",
        "tell me about that scene",
        "Who does Steve Rogers meet while jogging in Washington DC?",
        "Send an email to director@marvel.com",
        "Email the morning run scene breakdown to producer@studio.com",
        "what did he say"
    ]

    for p in test_prompts:
        decision = router.route_request(p)
        print(f"\nPrompt: '{p}' -> Route: {decision['type']}")
