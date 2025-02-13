import streamlit as st
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

class CapturingStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.generated_text = ""  # Store generated text

    def on_finalized_text(self, text, stream_end=False):
        self.generated_text += text  # Append streamed text


# Configure Streamlit for a wide layout
st.set_page_config(page_title="LLM Exam App", layout="wide")

# -------------------------------
# Load your LLM and tokenizer
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "unsloth/Qwen2.5-14B-Instruct-1M-unsloth-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -------------------------------
# Generation parameters and base prompt text
# -------------------------------
BASE_PROMPT = "You are a very smart and knowledgable assistant who is tasked with generating questions for an exam based on the text you are provided that contains different previous year question papers of the subject, you must generaete questions similar to and testing similar things to the input text. You are supposed to generate the output in markdown(.md) format.\n"
MAX_NEW_TOKENS = 20000  # Adjust as needed
TEMPERATURE = 0.7
TOP_P = 0.9

# -------------------------------
# Divide the page into 3 columns
# -------------------------------
col_left, col_center, col_right = st.columns(3)

# ----- Left Panel: Upload Documents -----
with col_left:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose one or more PDF files", type=["pdf"], accept_multiple_files=True)
    pdf_text = ""
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")
        st.session_state["pdf_text"] = pdf_text
        st.success("Uploaded documents processed successfully.")
    else:
        st.session_state.setdefault("pdf_text", "")

# ----- Center Panel: Chat Interface -----
with col_center:
    st.header("Chat")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Display previous messages (if any)
    for chat in st.session_state["chat_history"]:
        st.markdown(f"**{chat['role']}:** {chat['message']}")
    
    # Chat input widget
    chat_input = st.text_input("Enter your message here", key="chat_input")
    user_string = chat_input 
    if st.button("Send Chat"):
        if chat_input:
            user_string = chat_input 
            # Append user message
            st.session_state["chat_history"].append({"role": "User", "message": chat_input})
            # For demo purposes, the bot echoes the text back.
            bot_reply = "Echo: " + chat_input
            st.session_state["chat_history"].append({"role": "Bot", "message": bot_reply})
            st.rerun()

# ----- Right Panel: Generate Exam Questions -----
with col_right:
    st.header("Exam Questions")
    if st.button("Generate Questions"):
        st.write("Generating response...")
        # Create a combined prompt using the base text plus (a slice of) document content
        combined_prompt = BASE_PROMPT + user_string
        if st.session_state.get("pdf_text"):
            # Use only the first 1000 characters of the extracted text
            combined_prompt += st.session_state["pdf_text"][:1000] + "\n"
        else:
            combined_prompt += "No document content available.\n"
        combined_prompt += "\nQuestions:\n"

        try:
            # Tokenize the prompt and send to the model
            inputs = tokenizer(combined_prompt, return_tensors="pt").to(device)
            
            # Initialize a TextStreamer for clean output formatting
            streamer = CapturingStreamer(tokenizer)
            
            # Generate output using the model
            output_ids = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated tokens
            response_text = streamer.generated_text
            print(f"Debug: response_text = {response_text}")
            st.text_area("Generated Exam Questions (Markdown)", value=response_text, height=300)
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
