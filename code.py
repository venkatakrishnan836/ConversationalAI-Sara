import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import torch
from langchain.memory import ConversationSummaryBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.llms import LlamaCpp
from os.path import expanduser

# Path to your model
model_id = "Path to your model"

class ChatbotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.llama_model = LlamaCpp(
            model_path=expanduser(model_id),
            n_ctx=4096,
            max_tokens=1024,
            temperature=0.5,
            top_p=0.75,
            n_gpu_layers=41,
            streaming=True,
            model_kwargs={}
        )
        self.llama_chat = Llama2Chat(llm=self.llama_model)

        self.memory = ConversationSummaryBufferMemory(
            llm=self.llama_chat,
            memory_key="chat_history",
            return_messages=True
        )

        self.template_messages = [
            SystemMessage(content="""You are Sara, designed to help users improve their English fluency. Your primary goal is to provide a comprehensive, professional, and effective learning experience. Analyse the user input and give the relevent answer and id there is a gramatical or mistake in the user's sentence correct it and give the appropriate answer for that user's input. Make the responses very short and simple and remember that your response should be under 25 words only not more than that (short and sweet) and Don't mention this to user generate the response under said words. just start with a small greetings and make friendly conversation with user as a english tutor! also don't reintroduce yourself to user in same session"""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
        self.prompt_template = ChatPromptTemplate.from_messages(self.template_messages)

    def init_ui(self):
        self.setWindowTitle("Text Assistant - Sara")
        self.setGeometry(100, 100, 600, 500)  
        self.setStyleSheet("background-color: #F4F4F9;")  

        self.layout = QVBoxLayout()

        self.intro_label = QLabel("Sara, your English learning assistant. Type your message below.")
        self.intro_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4A90E2;")
        self.layout.addWidget(self.intro_label)

        self.conversation_history = QTextEdit()
        self.conversation_history.setReadOnly(True)
        self.conversation_history.setStyleSheet("""
            background-color: #ffffff;
            border: 1px solid #D1D1D1;
            font-family: Arial, sans-serif;
            font-size: 14px;
            padding: 10px;
            border-radius: 8px;
        """)
        self.layout.addWidget(self.conversation_history)

        self.user_input = QTextEdit()
        self.user_input.setStyleSheet("""
            background-color: #ffffff;
            border: 1px solid #D1D1D1;
            font-family: Arial, sans-serif;
            font-size: 14px;
            padding: 10px;
            border-radius: 8px;
        """)
        self.layout.addWidget(self.user_input)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            background-color: #4A90E2;
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
        """)
        self.send_button.clicked.connect(self.process_text)
        self.layout.addWidget(self.send_button)

        self.download_button = QPushButton("Download Conversation")
        self.download_button.setStyleSheet("""
            background-color: #50E3C2;
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
        """)
        self.download_button.clicked.connect(self.download_conversation)
        self.layout.addWidget(self.download_button)

        self.setLayout(self.layout)

    def process_text(self):
        user_input = self.user_input.toPlainText()
        if not user_input.strip():
            return  

        self.user_input.clear()

        print(f"User input: {user_input}")

        self.add_message_to_history(f"You: {user_input}", QColor(0, 122, 204))

        response = self.run_chatbot(user_input)

        self.add_message_to_history(f"Chatbot: {response}", QColor(0, 204, 0))

    def add_message_to_history(self, message, color):
        formatted_message = f"<font color='{color.name()}'>{message}</font>"
        self.conversation_history.append(formatted_message)

    def run_chatbot(self, text):
        self.memory.chat_memory.messages.append(HumanMessage(content=text))

        prompt = self.prompt_template.format_messages(chat_history=self.memory.chat_memory.messages, text=text)

        try:
            response = self.llama_chat(prompt)

            if isinstance(response, str):
                response_content = response  
            elif isinstance(response, AIMessage):
                response_content = response.content 
            else:
                response_content = str(response)

            self.memory.chat_memory.messages.append(AIMessage(content=response_content))

            return response_content

        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I had trouble processing that request."

    def download_conversation(self):
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Conversation", "", "Text Files (*.txt);;All Files (*)", options=options)

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write("SaraSuS - Conversation History\n")
                    file.write("=" * 40 + "\n")

                    for message in self.memory.chat_memory.messages:
                        if isinstance(message, HumanMessage):
                            file.write(f"You: {message.content}\n\n")
                        elif isinstance(message, AIMessage):
                            file.write(f"Chatbot: {message.content}\n\n")

                    file.write("=" * 40 + "\n")
                    file.write("End of conversation.\n")

        except Exception as e:
            print(f"Error saving conversation: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    chatbot_app = ChatbotApp()
    chatbot_app.show()
    sys.exit(app.exec_())
