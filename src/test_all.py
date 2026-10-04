import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_srt
from chunker import chunk_subtitles
from mcp_email_tool import MCPEmailTool
from router import AgenticRouter


class TestMovieIntelComponents(unittest.TestCase):

    def test_srt_parser(self):
        srt_path = os.path.join("data", "movies", "captain_america2.srt")
        if os.path.exists(srt_path):
            subtitles = parse_srt(srt_path)
            self.assertTrue(len(subtitles) > 0)
            self.assertIn("movie", subtitles[0])
            self.assertIn("start", subtitles[0])
            self.assertIn("end", subtitles[0])
            self.assertIn("text", subtitles[0])

    def test_subtitle_chunker(self):
        dummy_subtitles = [
            {"movie": "TestMovie", "start": 0, "end": 10, "text": "Line 1"},
            {"movie": "TestMovie", "start": 15, "end": 25, "text": "Line 2"},
        ]
        # Test helper functionality
        from datetime import timedelta
        dummy_sub_parsed = [
            {"movie": "TestMovie", "start": timedelta(seconds=0), "end": timedelta(seconds=10), "text": "Line 1"},
            {"movie": "TestMovie", "start": timedelta(seconds=15), "end": timedelta(seconds=25), "text": "Line 2"},
        ]
        chunks = chunk_subtitles(dummy_sub_parsed, max_duration=60, overlap_duration=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["movie"], "TestMovie")
        self.assertIn("Line 1", chunks[0]["text"])

    def test_mcp_email_tool(self):
        tool = MCPEmailTool(log_file="data/test_mcp_log.json")
        res = tool.execute(
            recipient="test@example.com",
            subject="Test Subject",
            body="Test Body"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["dispatch_record"]["recipient"], "test@example.com")

        # Test missing recipient validation
        invalid_res = tool.execute(recipient="", subject="Test", body="Test")
        self.assertFalse(invalid_res["success"])

        if os.path.exists("data/test_mcp_log.json"):
            os.remove("data/test_mcp_log.json")

    def test_agentic_router_rules(self):
        router = AgenticRouter()
        
        # Test Informational
        r1 = router._rule_fallback_classify("Who is Steve Rogers?")
        self.assertEqual(r1["type"], "INFORMATIONAL")

        # Test Action Email
        r2 = router._rule_fallback_classify("Email morning run scene breakdown to director@marvel.com")
        self.assertEqual(r2["type"], "ACTION_EMAIL")
        self.assertEqual(r2["recipient"], "director@marvel.com")

        # Test Ambiguous - missing email
        r3 = router._rule_fallback_classify("Send an email")
        self.assertEqual(r3["type"], "AMBIGUOUS")

        # Test Ambiguous - vague demonstrative query
        r4 = router._rule_fallback_classify("tell me about that scene")
        self.assertEqual(r4["type"], "AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()
