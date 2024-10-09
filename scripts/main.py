import sys
import subprocess
import importlib.util
from modules import script_callbacks, shared

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def is_package_installed(package):
    return importlib.util.find_spec(package) is not None

packages_to_install = ["ollama"]

for package in packages_to_install:
    if not is_package_installed(package):
        print(f"{package} library not found. Installing...")
        install_package(package)
        print(f"{package} library installed successfully.")

import gradio as gr
import ollama

class OllamaConfig:
    def __init__(self):
        self.default_host = "http://localhost:11434"
        self.default_model = "mistral-nemo"
        self.available_models = ["mistral-nemo", "llama3.2", "gdisney/mistral-uncensored"]

    def load_from_shared_opts(self):
        if hasattr(shared.opts, 'ollama_default_host'):
            self.default_host = shared.opts.ollama_default_host
        if hasattr(shared.opts, 'ollama_default_model'):
            self.default_model = shared.opts.ollama_default_model
        if hasattr(shared.opts, 'ollama_available_models'):
            self.available_models = shared.opts.ollama_available_models.split(',')

config = OllamaConfig()

class OllamaExtension:
    def __init__(self):
        self.client = None
        self.current_host = None

    def initialize_client(self, host):
        self.client = ollama.Client(host=host)
        self.current_host = host

    def get_ollama_response_stream(self, message, history, model):
        messages = [
            {"role": "system", "content": "Please respond using naturally written language, as if writing a well-formed text response."}
        ]
        
        for h_message, h_response in history:
            messages.append({"role": "user", "content": h_message})
            messages.append({"role": "assistant", "content": h_response})
        
        messages.append({"role": "user", "content": message})
        
        try:
            stream = self.client.chat(model=model, messages=messages, stream=True)
            full_response = ""
            for chunk in stream:
                if 'message' in chunk:
                    content = chunk['message'].get('content', '')
                    full_response += content
                    yield full_response
        except Exception as e:
            yield f"Error: {str(e)}"

    def chat(self, message, history, model, host):
        if not self.client or self.current_host != host:
            self.initialize_client(host)
        
        history = history or []
        history.append((message, ""))
        for response in self.get_ollama_response_stream(message, history[:-1], model):
            history[-1] = (message, response)
            yield "", history

def on_ui_tabs():
    config.load_from_shared_opts()  # Load the latest configuration
    
    with gr.Blocks(analytics_enabled=False) as ollama_interface:
        ollama_ext = OllamaExtension()
        
        with gr.Row():
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(label="Chat History", height=600)
                msg = gr.Textbox(label="Your message", placeholder="Type your message here...")
                with gr.Row():
                    clear = gr.Button("Clear")
                    send = gr.Button("Send")
            
            with gr.Column(scale=1):
                model = gr.Dropdown(choices=config.available_models, label="Model", value=config.default_model)
                host = gr.Textbox(label="Ollama URL", value=config.default_host, placeholder="http://localhost:11434")

        send.click(ollama_ext.chat, inputs=[msg, chatbot, model, host], outputs=[msg, chatbot])
        clear.click(lambda: [], outputs=[chatbot], show_progress=False)

    return [(ollama_interface, "Ollama Chat", "ollama_chat_tab")]

def on_ui_settings():
    section = ("ollama", "Ollama")
    shared.opts.add_option("ollama_default_host", shared.OptionInfo(config.default_host, "Default Ollama host", section=section))
    shared.opts.add_option("ollama_default_model", shared.OptionInfo(config.default_model, "Default Ollama model", section=section))
    shared.opts.add_option("ollama_available_models", shared.OptionInfo(",".join(config.available_models), "Available Ollama models (comma-separated)", section=section))

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)