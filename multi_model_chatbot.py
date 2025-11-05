import os
from typing import Dict, List
import anthropic
import openai
from google.generativeai import GenerativeModel
import google.generativeai as genai

class MultiModelChatbot:
    """
    A chatbot that can interact with multiple AI models:
    - Claude (Anthropic)
    - GPT (OpenAI)
    - Gemini (Google)
    """
    
    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {
            'claude': [],
            'gpt': [],
            'gemini': []
        }
        
        # Initialize API clients
        self.anthropic_client = None
        self.openai_client = None
        self.gemini_model = None
        
    def setup_claude(self, api_key: str):
        """Initialize Claude client"""
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        
    def setup_gpt(self, api_key: str):
        """Initialize OpenAI GPT client"""
        self.openai_client = openai.OpenAI(api_key=api_key)
        
    def setup_gemini(self, api_key: str):
        """Initialize Google Gemini client"""
        genai.configure(api_key=api_key)
        self.gemini_model = GenerativeModel('gemini-pro')
        
    def chat_with_claude(self, message: str, model: str = "claude-3-5-sonnet-20241022") -> str:
        """Send message to Claude and get response"""
        if not self.anthropic_client:
            return "Error: Claude API not configured"
        
        self.conversation_history['claude'].append({
            "role": "user",
            "content": message
        })
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=self.conversation_history['claude']
            )
            
            assistant_message = response.content[0].text
            self.conversation_history['claude'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_with_gpt(self, message: str, model: str = "gpt-4") -> str:
        """Send message to GPT and get response"""
        if not self.openai_client:
            return "Error: OpenAI API not configured"
        
        self.conversation_history['gpt'].append({
            "role": "user",
            "content": message
        })
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=self.conversation_history['gpt']
            )
            
            assistant_message = response.choices[0].message.content
            self.conversation_history['gpt'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_with_gemini(self, message: str) -> str:
        """Send message to Gemini and get response"""
        if not self.gemini_model:
            return "Error: Gemini API not configured"
        
        try:
            # Build conversation context
            chat = self.gemini_model.start_chat(history=[])
            
            # Add previous context if exists
            for msg in self.conversation_history['gemini']:
                if msg['role'] == 'user':
                    chat.send_message(msg['content'])
            
            # Send current message
            response = chat.send_message(message)
            assistant_message = response.text
            
            # Update history
            self.conversation_history['gemini'].append({
                "role": "user",
                "content": message
            })
            self.conversation_history['gemini'].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat(self, model_name: str, message: str) -> str:
        """
        Universal chat method
        
        Args:
            model_name: 'claude', 'gpt', or 'gemini'
            message: User's message
        
        Returns:
            AI model's response
        """
        if model_name.lower() == 'claude':
            return self.chat_with_claude(message)
        elif model_name.lower() == 'gpt':
            return self.chat_with_gpt(message)
        elif model_name.lower() == 'gemini':
            return self.chat_with_gemini(message)
        else:
            return "Error: Unknown model. Choose 'claude', 'gpt', or 'gemini'"
    
    def clear_history(self, model_name: str = None):
        """Clear conversation history for one or all models"""
        if model_name:
            self.conversation_history[model_name] = []
        else:
            for key in self.conversation_history:
                self.conversation_history[key] = []
    
    def get_history(self, model_name: str) -> List[Dict]:
        """Get conversation history for a specific model"""
        return self.conversation_history.get(model_name, [])


def main():
    """Demo usage of the MultiModelChatbot"""
    print("=== Multi-Model AI Chatbot ===\n")
    
    # Initialize chatbot
    bot = MultiModelChatbot()
    
    # Setup APIs (replace with your actual API keys)
    print("Setting up API clients...")
    
    # Example: Load from environment variables
    claude_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GOOGLE_API_KEY')
    
    if claude_key:
        bot.setup_claude(claude_key)
        print("✓ Claude configured")
    
    if openai_key:
        bot.setup_gpt(openai_key)
        print("✓ GPT configured")
    
    if gemini_key:
        bot.setup_gemini(gemini_key)
        print("✓ Gemini configured")
    
    print("\n" + "="*50)
    print("Available models: claude, gpt, gemini")
    print("Commands: 'switch [model]', 'clear', 'history', 'quit'")
    print("="*50 + "\n")
    
    current_model = 'claude'
    
    while True:
        user_input = input(f"\n[{current_model.upper()}] You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        elif user_input.lower().startswith('switch '):
            new_model = user_input.split()[1].lower()
            if new_model in ['claude', 'gpt', 'gemini']:
                current_model = new_model
                print(f"Switched to {current_model.upper()}")
            else:
                print("Invalid model. Choose: claude, gpt, or gemini")
            continue
        
        elif user_input.lower() == 'clear':
            bot.clear_history(current_model)
            print(f"Cleared {current_model.upper()} history")
            continue
        
        elif user_input.lower() == 'history':
            history = bot.get_history(current_model)
            print(f"\n--- {current_model.upper()} History ---")
            for msg in history:
                role = msg['role'].capitalize()
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"{role}: {content}")
            continue
        
        # Get response from current model
        print(f"\n[{current_model.upper()}] AI: ", end="", flush=True)
        response = bot.chat(current_model, user_input)
        print(response)


if __name__ == "__main__":
    main()
