import streamlit as st
import asyncio
from dotenv import load_dotenv
from research_manager import ResearchManager

load_dotenv(override=True)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide"
)

# Custom Styling & Header
st.title("🔬 Autonomous Deep Research Agent")
st.markdown("Powered by **Google Gemini** and structured agent workflows.")

# Example queries for quick testing
EXAMPLES = [
    "Most popular AI Agent frameworks in 2026",
    "Latest breakthroughs in Quantum Computing",
    "State of humanoid robotics and embodied AI"
]

selected_example = st.selectbox("Or choose an example query:", ["Select an example..."] + EXAMPLES)
default_query = selected_example if selected_example != "Select an example..." else ""

# Query Input Form
with st.form("research_form"):
    query_textbox = st.text_input(
        "Research Topic / Question",
        value=default_query,
        placeholder="Type a research question..."
    )
    run_button = st.form_submit_button("Investigate 🚀", type="primary")

# Execution Logic
if run_button:
    if not query_textbox.strip():
        st.warning("Please enter or select a research question.")
    else:
        # Create live status box for progress updates
        status_box = st.status("Initializing research pipeline...", expanded=True)
        report_placeholder = st.empty()

        async def execute_research():
            final_report = ""
            manager = ResearchManager()
            try:
                async for update in manager.run(query_textbox):
                    # Check if the yielded string is the final markdown report (typically long or starts with markdown formatting)
                    if update.startswith("#") or len(update) > 400:
                        final_report = update
                        status_box.update(label="✅ Research complete!", state="complete", expanded=False)
                    else:
                        status_box.write(update)
            except Exception as e:
                status_box.update(label="❌ Error during research execution", state="error", expanded=False)
                st.error(f"An error occurred: {e}")
            return final_report

        # Run the async generator pipeline inside Streamlit's event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        report_output = loop.run_until_complete(execute_research())

        # Display final report if successfully generated
        if report_output:
            st.divider()
            st.subheader("📄 Generated Research Report")
            st.markdown(report_output, unsafe_allow_html=True)