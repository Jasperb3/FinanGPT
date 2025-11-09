#!/usr/bin/env python3
"""Tests for conversational query interface (Phase 3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Import functions to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import (
    trim_conversation_history,
    MAX_HISTORY_LENGTH,
)


class TestConversationHistory:
    """Test conversation history management."""

    def test_trim_conversation_history_short(self):
        """Test that short history is not trimmed."""
        history = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        trimmed = trim_conversation_history(history)

        assert len(trimmed) == 3
        assert trimmed == history

    def test_trim_conversation_history_long(self):
        """Test that long history is trimmed correctly."""
        # Create a history longer than MAX_HISTORY_LENGTH
        history = [{"role": "system", "content": "System prompt"}]

        # Add many user/assistant exchanges
        for i in range(MAX_HISTORY_LENGTH + 5):
            history.append({"role": "user", "content": f"Question {i}"})
            history.append({"role": "assistant", "content": f"Answer {i}"})

        trimmed = trim_conversation_history(history)

        # Should keep system prompt + MAX_HISTORY_LENGTH recent messages
        assert len(trimmed) <= MAX_HISTORY_LENGTH + 1
        assert trimmed[0]["role"] == "system"
        assert trimmed[0]["content"] == "System prompt"

        # Check that we kept the most recent messages
        assert "Question 0" not in str(trimmed)  # Old messages removed
        assert f"Question {MAX_HISTORY_LENGTH + 4}" in str(trimmed)  # Recent kept

    def test_trim_conversation_history_preserves_system_prompt(self):
        """Test that system prompt is always preserved."""
        history = [{"role": "system", "content": "Important system prompt"}]

        # Add many messages
        for i in range(50):
            history.append({"role": "user", "content": f"Message {i}"})

        trimmed = trim_conversation_history(history)

        assert trimmed[0]["role"] == "system"
        assert trimmed[0]["content"] == "Important system prompt"

    def test_trim_conversation_history_no_system_prompt(self):
        """Test trimming when no system prompt exists."""
        # Create history without system prompt
        history = []
        for i in range(MAX_HISTORY_LENGTH + 10):
            history.append({"role": "user", "content": f"Question {i}"})
            history.append({"role": "assistant", "content": f"Answer {i}"})

        trimmed = trim_conversation_history(history)

        # Should keep only MAX_HISTORY_LENGTH most recent messages
        assert len(trimmed) == MAX_HISTORY_LENGTH
        # Should keep most recent messages
        assert f"Question {MAX_HISTORY_LENGTH + 9}" in str(trimmed)


class TestCallOllamaChat:
    """Test Ollama chat API integration."""

    @patch('chat.requests.post')
    def test_call_ollama_chat_success(self, mock_post):
        """Test successful Ollama chat call."""
        from chat import call_ollama_chat

        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "SELECT * FROM financials.annual LIMIT 10"}
        }
        mock_post.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are a SQL assistant"},
            {"role": "user", "content": "Show me some data"}
        ]

        result = call_ollama_chat("http://localhost:11434", "phi4:latest", messages)

        assert result == "SELECT * FROM financials.annual LIMIT 10"
        mock_post.assert_called_once()

    @patch('chat.requests.post')
    def test_call_ollama_chat_empty_response(self, mock_post):
        """Test handling of empty Ollama response."""
        from chat import call_ollama_chat

        # Mock empty response
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {}}
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="empty response"):
            call_ollama_chat("http://localhost:11434", "phi4:latest", messages)


class TestQueryRetryLogic:
    """Test intelligent error recovery with retry."""

    @patch('chat.call_ollama_chat')
    @patch('chat.duckdb.DuckDBPyConnection')
    def test_execute_query_success_first_try(self, mock_conn, mock_ollama):
        """Test query succeeds on first attempt."""
        from chat import execute_query_with_retry

        # Mock successful SQL generation and execution
        mock_ollama.return_value = "SELECT ticker, date FROM financials.annual LIMIT 5"

        mock_result = MagicMock()
        mock_result.description = [["ticker"], ["date"]]
        mock_result.fetchall.return_value = [("AAPL", "2024-09-30")]
        mock_conn.execute.return_value = mock_result

        conversation_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Show me data"}
        ]

        schema = {"financials.annual": ["ticker", "date"]}
        logger = MagicMock()

        result = execute_query_with_retry(
            mock_conn,
            "http://localhost:11434",
            "phi4:latest",
            conversation_history,
            schema,
            logger,
            None,
            True,
        )

        assert result is not None
        columns, rows, sql = result
        assert "ticker" in columns
        assert len(rows) > 0

    @patch('chat.call_ollama_chat')
    @patch('chat.duckdb.DuckDBPyConnection')
    def test_execute_query_retry_on_error(self, mock_conn, mock_ollama):
        """Test query retries on validation error."""
        from chat import execute_query_with_retry

        # First attempt fails, second succeeds
        mock_ollama.side_effect = [
            "SELECT * FROM invalid_table",  # First attempt - invalid
            "SELECT ticker FROM financials.annual LIMIT 5",  # Second attempt - valid
        ]

        mock_result = MagicMock()
        mock_result.description = [["ticker"]]
        mock_result.fetchall.return_value = [("AAPL",)]
        mock_conn.execute.return_value = mock_result

        conversation_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Show me data"}
        ]

        schema = {"financials.annual": ["ticker", "date"]}
        logger = MagicMock()

        result = execute_query_with_retry(
            mock_conn,
            "http://localhost:11434",
            "phi4:latest",
            conversation_history,
            schema,
            logger,
            None,
            True,
        )

        # Should succeed on retry
        assert result is not None

        # Should have added error feedback to history
        assert len([msg for msg in conversation_history if msg["role"] == "system" and "failed" in msg["content"].lower()]) > 0

    @patch('chat.call_ollama_chat')
    @patch('chat.duckdb.DuckDBPyConnection')
    def test_execute_query_max_retries_exceeded(self, mock_conn, mock_ollama):
        """Test query fails after max retries."""
        from chat import execute_query_with_retry, MAX_RETRIES

        # All attempts fail
        mock_ollama.return_value = "SELECT * FROM invalid_table"

        conversation_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Show me data"}
        ]

        schema = {"financials.annual": ["ticker", "date"]}
        logger = MagicMock()

        result = execute_query_with_retry(
            mock_conn,
            "http://localhost:11434",
            "phi4:latest",
            conversation_history,
            schema,
            logger,
            None,
            True,
        )

        # Should return None after max retries
        assert result is None

        # Should have called Ollama MAX_RETRIES times
        assert mock_ollama.call_count == MAX_RETRIES


class TestChatCommands:
    """Test chat interface commands."""

    def test_welcome_message_display(self):
        """Test that welcome message is properly formatted."""
        from chat import print_welcome_message
        import io
        from contextlib import redirect_stdout

        schema = {
            "financials.annual": ["ticker", "date"],
            "prices.daily": ["ticker", "date", "close"],
        }

        # Capture printed output
        f = io.StringIO()
        with redirect_stdout(f):
            print_welcome_message(schema)

        output = f.getvalue()

        # Check for key elements
        assert "FinanGPT" in output
        assert "Available tables" in output
        assert "financials.annual" in output
        assert "prices.daily" in output
        assert "/help" in output
        assert "/exit" in output

    def test_help_message_display(self):
        """Test that help message is properly formatted."""
        from chat import print_help
        import io
        from contextlib import redirect_stdout

        # Capture printed output
        f = io.StringIO()
        with redirect_stdout(f):
            print_help()

        output = f.getvalue()

        # Check for key elements
        assert "Help" in output
        assert "/clear" in output
        assert "/exit" in output
        assert "Tips" in output or "Example" in output


class TestIntegration:
    """Integration tests for chat interface."""

    def test_chat_imports(self):
        """Test that all required functions are imported correctly."""
        from chat import (
            build_system_prompt,
            check_data_freshness,
            extract_sql,
            extract_tickers_from_sql,
            introspect_schema,
            load_mongo_database,
            pretty_print,
            validate_sql,
        )

        # All imports should succeed
        assert callable(build_system_prompt)
        assert callable(check_data_freshness)
        assert callable(extract_sql)
        assert callable(extract_tickers_from_sql)
        assert callable(introspect_schema)
        assert callable(load_mongo_database)
        assert callable(pretty_print)
        assert callable(validate_sql)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
