import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Assistant Eval", page_icon="🤖", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 AI Assistant Eval")
    choice = st.radio("Choose assistant", ["🌐 Frontier (Claude Sonnet)", "🦙 OSS (Qwen 2.5)"])
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        if "assistant" in st.session_state:
            st.session_state.assistant.reset()
        st.rerun()

# --- Load assistant ---
@st.cache_resource
def get_frontier():
    from assistants.frontier import FrontierAssistant
    return FrontierAssistant()

@st.cache_resource
def get_oss():
    from assistants.oss import OSSAssistant
    return OSSAssistant()

is_frontier = choice.startswith("🌐")

# Clear chat when switching assistants
if "last_choice" not in st.session_state:
    st.session_state.last_choice = choice
if st.session_state.last_choice != choice:
    st.session_state.messages = []
    st.session_state.last_choice = choice

try:
    assistant = get_frontier() if is_frontier else get_oss()
    assistant.reset()  # reset memory on each switch
    st.session_state.assistant = assistant
except Exception as e:
    st.error(f"Failed to load assistant: {e}")
    st.stop()

# --- Chat UI ---
st.header(f"Chat — {assistant.name}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = assistant.chat(prompt)
            except NotImplementedError as e:
                response = f"⚠️ {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})