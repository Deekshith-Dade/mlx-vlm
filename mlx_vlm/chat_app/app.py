from typing import List
from textual import work
from textual.app import App, ComposeResult, RenderResult
from textual.containers import VerticalScroll
from textual.widget import AwaitMount, Widget
from textual.widgets import Header, Footer, Input, Static, Markdown

from mlx_vlm import load
from mlx_vlm.generate import generate_step, apply_chat_template, stream_generate
from mlx_vlm.prompt_utils import get_message_json 
from mlx_vlm.utils import load_image

class MessageInfo(Widget):
    def __init__(
        self, 
        prompt_tokens: int,
        generation_tokens: int,
        prompt_tps: float,
        generation_tps: float,
        peak_memory: float,
        *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.prompt_tps = prompt_tps
        self.generation_tps = generation_tps
        self.peak_memory = peak_memory
    
    def render(self) -> RenderResult:
        return f"Prompt Tokens: {self.prompt_tokens}\nGeneration Tokens: {self.generation_tokens}\nPrompt TPS: {self.prompt_tps}\nGeneration TPS: {self.generation_tps}\nPeak Memory: {self.peak_memory}"

class ChatMessage(Static):
    def __init__(self, role: str, message: str = "", verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.role = role
        self.markdown = Markdown(message)
        self.message_info = None

    async def add_message_info(self, message_info: MessageInfo) -> None:
        self.message_info = message_info
        if self.verbose:
            await self.mount(self.message_info)
            self.message_info.scroll_visible()

    def compose(self) -> ComposeResult:
        yield self.markdown
    
    

class ChatApp(App):
    CSS_PATH = "tcss/chat.tcss"

    AUTO_FOCUS = "#message-input"

    def __init__(self, agent_name: str = "AI", **kwargs):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.model_path = "mlx-community/gemma-3n-E2B-it-4bit"

        self.load_model = True
        with self.console.status("[bold green]Loading model..."):
            self.model, self.processor = load(self.model_path)
            self.loading_model = False
        
        self.history = []
        self.images = []
        self.audio = []
        self.max_tokens = None
        self.temperature = 0.1
        self.verbose = True
    
    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="chat-view"):
            yield ChatMessage("system", "Welcome to Textual Chat App! Type a message below.", classes="message")
        
        yield Input(placeholder="Type your message here...", id="message-input")
        yield Footer()


    def add_to_history(self, role: str, text: str) -> None:
        content = text
        self.history.append({"role": role, "content": content})

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value
        if message:
            new_message = ChatMessage("user", message, 
            verbose=False,
                                      classes="message message-user")

            event.input.value = ""
            chat_view = self.query_one("#chat-view")
            chat_view.mount(new_message)

            new_message.scroll_visible()
            

            self.post_agent_response(message)
        
    def _prepare_messages(self) -> str:
        messages = apply_chat_template(
            self.processor, self.model.config, self.history, num_images=len(self.images), num_audios=len(self.audio)
        )
        return messages
        
    async def _generate_response(self, messages) -> str:
        for response in stream_generate(
            self.model,
            self.processor,
            messages,
            self.images,
            self.audio,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # **kwargs
        ):
            yield response

    @work
    async def stream_markdown(self, messages: list) -> None:
        new_message = ChatMessage("assistant", "", 
                                  verbose=self.verbose,
                                  classes="message")
        chat_view = self.query_one("#chat-view")
        chat_view.mount(new_message)

        stream = Markdown.get_stream(new_message.markdown)

        try:
            async for response in self._generate_response(messages):
                await stream.write(response.text)
                new_message.scroll_visible()
        finally:
            await stream.stop()

        message_info = MessageInfo(
            prompt_tokens=response.prompt_tokens,
            generation_tokens=response.generation_tokens,
            prompt_tps=response.prompt_tps,
            generation_tps=response.generation_tps,
            peak_memory=response.peak_memory
        )
        await new_message.add_message_info(message_info)
        self.add_to_history("assistant", response.text)
        
    def post_agent_response(self, prompt: str) -> None:
        self.add_to_history("user", prompt)
        messages = self._prepare_messages()
        self.stream_markdown(messages)
    

    