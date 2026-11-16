
import streamlit as st

# --IMP-- Import our existing RAG pipeline
from gen_ai_rag.hybrid_search_streamlit import ask_question


# ============================================================
# STEP 1: CONFIGURE THE APPLICATION
# ============================================================

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# STEP 2: INITIALIZE CHAT HISTORY
# ============================================================

# session_state stores information between Streamlit reruns.
# This prevents our chat history from disappearing.

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# STEP 3: SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🤖 Knowledge Assistant")

    st.caption("Powered by Gemini + RAG")

    st.divider()

    st.subheader("About")
    st.write(
        "Ask questions about company policies. "
        "The assistant retrieves relevant sections "
        "before generating an answer."
    )

    # Clear the conversation history.
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# STEP 4: MAIN APPLICATION HEADER
# ============================================================

st.title("Knowledge Assistant")

st.caption(
    "Ask questions about company policies using AI."
)

st.divider()


# ============================================================
# STEP 5: DISPLAY PREVIOUS CHAT MESSAGES
# ============================================================

# Loop through stored messages and display them.

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display retrieved sources for assistant messages.
        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:
                with st.expander("Retrieved Sources"):
                    for i, source in enumerate(sources, start=1):
                        st.markdown(f"**Source {i}**")
                        st.write(source)

            # Display retrieval debugging information.
            debug = message.get("debug", {})

            if debug:
                with st.expander("Retrieval Debug Information"):
                    st.json(debug)


# ============================================================
# STEP 6: GET USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask about company policies..."
)


# ============================================================
# STEP 7: PROCESS USER QUESTION
# ============================================================

if question:

    # Save and display the user's question.
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response area.
    with st.chat_message("assistant"):

        with st.spinner("Searching company policies..."):

            try:
                # --IMP-- Call our existing RAG pipeline.
                result = ask_question(question)

                answer = result["answer"]
                sources = result["sources"]
                source_scores = result["source_scores"]
                debug = result["debug"]

                # Display generated answer.
                st.markdown(answer)

                # Display retrieved source sections.
                #This will display each retrieved section with its title.
                if sources:
                   with st.expander("Retrieved Policy Sources"):
                    for i, source in enumerate(sources):
                        title = source.strip().split("\n")[0]
                        score = source_scores[i]

                        st.markdown(f"### {title}")
                        st.caption(f"CrossEncoder relevance score: {score:.3f}")
                        st.write(source)

                        st.divider()

                # Display debug information.
                if debug:
                    with st.expander("Retrieval Debug Information"):
                        st.json(debug)

                # Save assistant response and supporting data.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "debug": debug
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
