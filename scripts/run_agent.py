import os
import sys
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import PhoenixAgent
from configs import base_config

def main():
    """
    A simple command-line interface to interact with the PhoenixAgent.
    """
    print("--- Phoenix Agent Runner ---")
    print("Initializing agent... (This may take a moment)")

    config = base_config.get_config()
    agent_config = config['agent']

    try:
        agent = PhoenixAgent(agent_config)
        print("Agent initialized. You can now ask questions.")
        print("Type 'exit' or 'quit' to end the session.\n")
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Failed to initialize the agent: {e}")
        print("\nPlease ensure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        print("\nAnd that you have a local LLM running (e.g., Ollama with llama3).")
        print("For Ollama, run: ollama serve")
        return

    # --- Main Interaction Loop ---
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ['exit', 'quit']:
                print("Phoenix: Goodbye!")
                break
            
            if not prompt:
                continue

            try:
                agent.run(prompt)
            except Exception as e:
                if 'not found' in str(e) and 'model' in str(e):
                    print(f"\n--- MODEL NOT FOUND ---\n{e}")
                    print(f"To fix: Run 'ollama pull {agent_config['llm_model_name']}' in your terminal.")
                else:
                    print(f"An error occurred: {e}")
            print("-" * 20)

        except KeyboardInterrupt:
            print("\nPhoenix: Session ended by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Phoenix Agent.")
    parser.add_argument('prompt', nargs='?', default=None, help="The initial prompt to send to the agent.")
    args = parser.parse_args()

    config = base_config.get_config()
    agent_config = config['agent']

    if args.prompt:
        try:
            agent = PhoenixAgent(agent_config)
            agent.run(args.prompt)
        except Exception as e:
            if 'not found' in str(e) and 'model' in str(e):
                print(f"\n--- MODEL NOT FOUND ---\n{e}")
                print(f"To fix: Run 'ollama pull {agent_config['llm_model_name']}' in your terminal.")
            else:
                print(f"An error occurred: {e}")
    else:
        main() 