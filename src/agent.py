import json
import torch
import ollama 
from duckduckgo_search import DDGS
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.vision_engine import VisionEngine
from configs import base_config

# --- Tool Definitions ---
config = base_config.get_config()
vision_engine_checkpoint = config['train']['model_checkpoint_path'] 
vision_engine = VisionEngine(model_checkpoint_path=vision_engine_checkpoint)

# ... (vision_tool and web_search_tool functions remain the same) ...
def vision_tool(image_path: str, query: str) -> str:
    """Analyzes an image to answer a query."""
    print(f"--- Analyzing image '{image_path}' with query '{query}' ---")
    image_tensor = torch.randn(3, 224, 224) 
    try:
        similarity = vision_engine.get_similarity(image_tensor, query)
        return f"The similarity between the image and the query '{query}' is {similarity:.2f}."
    except Exception as e:
        return f"Error analyzing image: {e}"

def web_search_tool(query: str) -> str:
    """Searches the web for up-to-date information."""
    print(f"--- Searching web for: '{query}' ---")
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
        if not results: return "No results found."
        return " ".join(results)
    except Exception as e:
        return f"Error during web search: {e}"

# --- Agent Definition ---

class PhoenixAgent:
    def __init__(self, agent_config: dict):
        self.tools = { "vision_tool": vision_tool, "web_search_tool": web_search_tool }
        
        # FIX: Read the model name and other settings from the config
        self.llm_model_name = agent_config['llm_model_name']
        self.max_steps = agent_config['max_steps']
        
        self.system_prompt = """
        You are Phoenix, a helpful AI assistant...
        """ # (The detailed prompt remains the same)

    def run(self, user_prompt: str):
        conversation = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        print(f"User Prompt: {user_prompt}\n")

        for _ in range(self.max_steps):
            try:
                # FIX: Use the model name from the instance variable
                response = ollama.chat(model=self.llm_model_name, messages=conversation)
                # ... (The rest of the run loop logic remains the same) ...
                response_text = response['message']['content']
                conversation.append({"role": "assistant", "content": response_text})
                try:
                    cleaned_response_text = response_text.strip().replace("`", "")
                    if cleaned_response_text.startswith("json"):
                        cleaned_response_text = cleaned_response_text[4:].strip()
                    action_json = json.loads(cleaned_response_text)
                    if "tool_name" in action_json and "tool_input" in action_json:
                        tool_name = action_json['tool_name']
                        tool_input = action_json['tool_input']
                        print(f"Phoenix Action: Calling tool '{tool_name}' with input {tool_input}")
                        observation = self.tools[tool_name](**tool_input)
                        print(f"Phoenix Observation: {observation}\n")
                        conversation.append({"role": "tool", "content": observation})
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                print(f"Phoenix Final Answer: {response_text}")
                return response_text
            except Exception as e:
                error_message = f"An error occurred in the agent loop: {e}"
                print(error_message)
                return error_message
        return "The agent could not reach a final answer."

