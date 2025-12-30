import anthropic
from typing import List, Optional

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content** - Search for specific content within course materials
2. **get_course_outline** - Get a course's structure: title, link, and complete lesson list

Tool Usage Guidelines:
- Use **get_course_outline** for questions about:
  - What lessons are in a course
  - Course structure or outline
  - What topics a course covers
  - How many lessons a course has
- Use **search_course_content** for questions about:
  - Specific concepts or topics within course content
  - Detailed information from lessons
- **Up to two tool calls per query** - Use first to gather info, second to refine if needed
- If no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**: Provide direct answers only

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        MAX_TOOL_ROUNDS = 2

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message list with user query
        messages = [{"role": "user", "content": query}]

        # Prepare base API parameters
        api_params = {
            **self.base_params,
            "system": system_content
        }

        # Add tools if available (kept for all iterations within the loop)
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Iterative tool loop
        round_count = 0
        while round_count < MAX_TOOL_ROUNDS:
            # Update messages in params
            api_params["messages"] = messages

            # Call Claude API
            response = self.client.messages.create(**api_params)

            # Exit if no tool use requested
            if response.stop_reason != "tool_use":
                return self._extract_text_response(response)

            # Exit if no tool_manager to execute tools
            if not tool_manager:
                return self._extract_text_response(response)

            # Execute tools and collect results
            tool_results, _ = self._execute_tool_calls(response, tool_manager)

            # Append assistant's tool use response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Append tool results to messages
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            round_count += 1

        # Force final response after max rounds (without tools)
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        final_response = self.client.messages.create(**final_params)
        return self._extract_text_response(final_response)

    def _execute_tool_calls(self, response, tool_manager) -> tuple:
        """
        Execute all tool calls from a response.

        Args:
            response: Claude API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results list, has_error boolean)
        """
        tool_results = []
        has_error = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    tool_result = f"Tool execution error: {str(e)}"
                    has_error = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })

        return tool_results, has_error

    def _extract_text_response(self, response) -> str:
        """
        Extract text content from Claude response.

        Args:
            response: Claude API response

        Returns:
            Text content from response, or empty string if no text found
        """
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        return ""