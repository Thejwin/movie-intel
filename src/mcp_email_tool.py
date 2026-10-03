import os
import json
import datetime


class MCPEmailTool:
    """
    Model Context Protocol (MCP) tool implementation for email dispatch.
    Complies with MCP tool definition standards:
      name: send_email
      parameters: recipient, subject, body, citations
    """

    def __init__(self, log_file="data/mcp_email_dispatch_log.json"):
        self.name = "send_email"
        self.description = "Sends an email containing movie RAG scene breakdowns, dialogue summaries, or quote analysis."
        self.log_file = log_file

    def get_tool_definition(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Recipient email address (e.g. user@example.com)"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject line of the email"
                    },
                    "body": {
                        "type": "string",
                        "description": "Main body content of the email including movie analysis and citations"
                    }
                },
                "required": ["recipient", "subject", "body"]
            }
        }

    def execute(self, recipient, subject, body, citations=None):
        if not recipient or "@" not in recipient:
            return {
                "success": False,
                "error": "Invalid or missing recipient email address."
            }
        if not subject:
            return {
                "success": False,
                "error": "Subject line is required."
            }
        if not body:
            return {
                "success": False,
                "error": "Email body content is required."
            }

        timestamp = datetime.datetime.now().isoformat()
        dispatch_record = {
            "timestamp": timestamp,
            "mcp_tool": self.name,
            "status": "DISPATCHED",
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "citations": citations or []
        }

        # Log dispatch record locally
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        logs = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []

        logs.append(dispatch_record)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "message": f"Email successfully dispatched via MCP tool to '{recipient}'.",
            "dispatch_record": dispatch_record
        }


if __name__ == "__main__":
    tool = MCPEmailTool()
    res = tool.execute(
        recipient="director@studio.com",
        subject="Scene Analysis: Steve & Sam Morning Run",
        body="Here is the breakdown for the opening scene...",
        citations=[{"movie": "Captain America 2", "start": "0:00:48", "end": "0:01:49"}]
    )
    print(res)
