import os
from groq import Groq
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class GroqChatbot:
    def __init__(self, model="llama-3.3-70b-versatile"):
        """
        Initialize the Groq chatbot with API key and model
        
        Args:
            model (str): Groq model to use (default: "llama-3.3-70b-versatile")
        """
        self.client = Groq(api_key=self._get_api_key())
        self.model = model
        self.chat_history = []
        
    def _get_api_key(self):
        """Retrieve API key from environment variables"""
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Please set GROQ_API_KEY environment variable. "
                "Get your key from https://console.groq.com/keys"
            )
        return key
    
    def _format_message(self, role, content):
        """Format message for chat history"""
        return {"role": role, "content": content}
    
    def is_ready(self):
        return hasattr(self, 'client') and self.client is not None
    
    def chat(self, system_prompt=None):
        """
        Start interactive chat session
        
        Args:
            system_prompt (str): Optional initial system prompt
        """
        if system_prompt:
            self.chat_history.append(
                self._format_message("system", system_prompt)
            )
        
        print(Fore.YELLOW + "\nGroq Chatbot initialized. Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input(Fore.GREEN + "You: ")
                
                if user_input.lower() in ["quit", "exit"]:
                    break
                    
                self.chat_history.append(
                    self._format_message("user", user_input)
                )
                
                response = self.client.chat.completions.create(
                    messages=self.chat_history,
                    model=self.model,
                    temperature=0.7,
                    max_tokens=1024
                )
                
                assistant_response = response.choices[0].message.content
                self.chat_history.append(
                    self._format_message("assistant", assistant_response)
                )
                
                print(Fore.BLUE + "\nAssistant:", assistant_response + "\n")
                
            except KeyboardInterrupt:
                print(Fore.RED + "\nExiting chat...")
                break
            except Exception as e:
                print(Fore.RED + f"\nError: {str(e)}")
                break

if __name__ == "__main__":
    # Example usage with phishing detection context
    SYSTEM_PROMPT = """You are a cybersecurity assistant specializing in phishing detection. 
    Help users analyze and identify potential threats. Be concise and technical when needed.Dont answer any question that is not related to cyber security.Only answer in text format."""
    
    bot = GroqChatbot()
    bot.chat(SYSTEM_PROMPT)