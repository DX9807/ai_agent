"""
Simple LangChain Agent using Groq
Features: Web search, calculator, and conversational abilities
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
# LangChain imports
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.tools.simple import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain_classic.memory import ConversationBufferMemory
import wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import create_react_agent


wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

# Load environment variables
load_dotenv()

# Verify API key
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables")

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information.
    """

    try:
        return wikipedia.summary(query, sentences=3)

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {e.options[:5]}"

    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found."

    except Exception as e:
        return f"Wikipedia error: {str(e)}"

# Tool definitionsCalculate 
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression as a string (e.g., '2 + 2', 'sqrt(16)', '100 * 0.5')
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Safe evaluation using restricted builtins
        import math
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'sqrt': math.sqrt, 'pi': math.pi,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10, 'exp': math.exp,
            'ceil': math.ceil, 'floor': math.floor, 'factorial': math.factorial
        }
        
        # Evaluate expression with access to math functions
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@tool
def get_current_time() -> str:
    """
    Get the current date and time.
    
    Returns:
        A string with the current date and time.
    """
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def word_counter(text: str) -> str:
    """
    Count words, characters, and sentences in a text.
    
    Args:
        text: The text to analyze
    """
    words = len(text.split())
    chars = len(text)
    sentences = len([s for s in text.split('.') if s.strip()])
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"

# ============================================================================
# SIMPLE AGENT CLASS
# ============================================================================

class GroqAgent:
    """
    A simple conversational agent powered by Groq's LLM.
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = True,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Groq agent.
        
        Args:
            model_name: Groq model to use (llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.)
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            verbose: Print debug information
            system_prompt: Custom system prompt
        """
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=API_KEY
        )
        
        # Define tools
        self.tools = [
            DuckDuckGoSearchRun(),
            ArxivQueryRun(),
            wikipedia_search,
            calculator,
            get_current_time,
            word_counter
        ]
        
        # Create prompt template
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant with access to various tools.
            
            You can:
            1. Search the web for current information
            2. Perform mathematical calculations
            3. Check current time
            4. Analyze text (count words, characters)

            Always think step by step:
            - Understand what the user is asking
            - Decide if you need to use a tool
            - Use tools when you need real-time or factual information
            - Explain your reasoning clearly
            - If you don't know something, be honest about it

            Be friendly and conversational in your responses."""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=5,  # Prevent infinite loops
            early_stopping_method="generate"  # Generate response if stuck
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        print(f"=>Agent initialized with model: {model_name}")
        print(f"=>Available tools: {[tool.name for tool in self.tools]}")
    
    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: User's input message
            
        Returns:
            Agent's response
        """
        try:
            # Get response from agent
            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": self.memory.chat_memory.messages
            })
            
            # Update memory
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(response["output"])
            
            return response["output"]
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(f"{error_msg}")
            return f"I encountered an error: {error_msg}"
    
    async def achat(self, message: str) -> str:
        """
        Async version of chat method.
        """
        try:
            response = await self.agent_executor.ainvoke({
                "input": message,
                "chat_history": self.memory.chat_memory.messages
            })
            
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(response["output"])
            
            return response["output"]
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(f"{error_msg}")
            return f"I encountered an error: {error_msg}"
    
    def reset_memory(self):
        """
        Clear conversation history.
        """
        self.memory.clear()
        print(" Memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        Returns:
            A list of messages with roles (user/assistant) and content.
        """
        history = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

# ============================================================================
# SIMPLIFIED AGENT (No Memory, Direct Tool Calling)
# ============================================================================

class SimpleGroqAgent:
    """
    A simpler agent without memory, using direct tool calling.
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """Initialize with a specific Groq model."""
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.7,
            api_key=API_KEY
        )
        
        self.tools = [
            DuckDuckGoSearchRun(),
            ArxivQueryRun(),
            calculator,
            get_current_time
        ]
        
        # Bind tools directly to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        print(f"Simple agent ready with {len(self.tools)} tools")
    
    def chat(self, message: str) -> str:
        """
        Simple chat method that handles tool calling manually.
        """
        messages = [HumanMessage(content=message)]
        
        # Get initial response
        response = self.llm_with_tools.invoke(messages)
        
        # Check if tool calls are needed
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Process each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Find and execute the tool
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if tool:
                    print(f"Using tool: {tool_name}({tool_args})")
                    tool_result = tool.invoke(tool_args)
                    
                    # Add tool result to conversation
                    messages.append(response)
                    messages.append(AIMessage(content=f"Tool result: {tool_result}"))
                    
                    # Get final response with tool context
                    final_response = self.llm_with_tools.invoke(messages)
                    return final_response.content
        
        return response.content

# ============================================================================
# INTERACTIVE CHAT INTERFACE
# ============================================================================

def run_interactive_chat():
    """
    Run an interactive chat session with the agent.
    """
    print("\n" + "="*60)
    print("Groq Agent - Interactive Chat")
    print("="*60)
    print("Commands:")
    print("  /clear - Clear conversation history")
    print("  /history - Show conversation history")
    print("  /quit  - Exit the chat")
    print("="*60 + "\n")
    
    # Create agent
    agent = GroqAgent(
        model_name="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
        temperature=0.7,
        verbose=False  # Set to True for debugging
    )
    
    print("Agent is ready! Start chatting...\n")
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            # Handle commands
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == '/clear':
                agent.reset_memory()
                continue
            elif user_input.lower() == '/history':
                history = agent.get_conversation_history()
                if history:
                    print("\nConversation History:")
                    for msg in history:
                        role_icon = "👤" if msg["role"] == "user" else "Agent"
                        print(f"{role_icon} {msg['role'].title()}: {msg['content'][:100]}...")
                else:
                    print("No conversation history")
                continue
            
            if not user_input:
                continue
            
            # Get agent response
            print("Agent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")

# ============================================================================
# EXAMPLE USAGE FUNCTIONS
# ============================================================================

def example_search():
    """
    Example: Web search functionality.
    """
    print("\n" + "="*60)
    print("Example: Web Search")
    print("="*60)
    
    agent = GroqAgent(verbose=False)
    
    queries = [
        "What are the latest developments in AI today?",
        "What's the current temperature in Tokyo?",
        "Who won the last world cup?"
    ]
    
    for query in queries:
        print(f"\n Query: {query}")
        print(f"Agent Response: {agent.chat(query)[:200]}...")
        print("-"*40)

def example_calculation():
    """
    Example: Mathematical calculations.
    """
    print("\n" + "="*60)
    print("Example: Calculations")
    print("="*60)
    
    agent = GroqAgent(verbose=False)
    
    calculations = [
        "What is 15% of 250?",
        "Calculate the square root of 196",
        "What's 2 to the power of 10?"
    ]
    
    for calc in calculations:
        print(f"\n👤 Query: {calc}")
        print(f"Agent Response: {agent.chat(calc)}")
        print("-"*40)

def example_multi_tool():
    """
    Example: Using multiple tools in one conversation.
    """
    print("\n" + "="*60)
    print("Example: Multi-Tool Usage")
    print("="*60)
    
    agent = GroqAgent(verbose=False)
    
    # Complex query requiring multiple tools
    query = """
    I need help with a few things:
    1. Search for the current population of Tokyo
    2. Calculate what 15% of that population is
    3. Tell me the current time
    """
    
    print(f"\nComplex Query: {query}")
    response = agent.chat(query)
    print(f"Agent Response:\n{response}")

# ============================================================================
# STREAMING VERSION
# ============================================================================

from duckduckgo_search import DDGS


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""

    results = DDGS().text(query, max_results=3)

    output = []

    for r in results:
        output.append(
            f"Title: {r['title']}\n"
            f"Snippet: {r['body']}\n"
            f"URL: {r['href']}\n"
        )

    return "\n\n".join(output)

from langchain_core.prompts import PromptTemplate

template = """
Answer the following questions as best you can.

You have access to the following tools:

{tools}

Use the following format:

Question: the question to answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Question: {input}

Thought: {agent_scratchpad}
"""


class StreamingGroqAgent:

    def __init__(self, model_name="llama-3.3-70b-versatile"):

        self.llm = ChatGroq(
            model=model_name,
            temperature=0.7,
            streaming=False,
            api_key=API_KEY
        )

        self.tools = [
            DuckDuckGoSearchRun(),
            ArxivQueryRun(),
            wikipedia_search,
            calculator
        ]

        prompt = PromptTemplate.from_template(template)

        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )

    def stream_chat(self, message: str):

        print("Streaming: ", end="", flush=True)

        for chunk in self.agent_executor.stream({
            "input": message
        }):

            if "output" in chunk:
                print(chunk["output"], end="", flush=True)
                return chunk["output"]
        print()

def run_streaming_interactive_chat():
    """
    Run an interactive chat session with streaming responses.
    """
    print("\n" + "="*60)
    print("AI Agent - Streaming Interactive Chat")
    print("="*60)
    
    agent = StreamingGroqAgent()
    
    print("Agent is ready! Start chatting...\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("\nGoodbye!")
                break
            if not user_input:
                continue
            
            agent.stream_chat(user_input)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("Please try again.\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔══════════════════════════════════════════╗
    ║     AI Agent using Groq LLM              ║
    ╚══════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        # Run specific examples
        mode = sys.argv[1]
        
        if mode == "examples" or mode == "example":
            example_search()
            example_calculation()
            example_multi_tool()
        elif mode == "stream":
            run_streaming_interactive_chat()
        elif mode == "simple":
            agent = SimpleGroqAgent()
            response = agent.chat("What's 25 * 4?")
            print(f"Simple response: {response}")
    else:
        # Run interactive chat
        run_interactive_chat()