import streamlit as st
from langgraphapp import app

st.set_page_config(page_title="✈️ Travel Assistant", layout="centered")
st.title("🧭 Smart Travel Assistant")

st.markdown("Ask me anything related to flights, like:")
st.markdown("- *What is the price to fly from Chennai to Hyderabad?*")
st.markdown("- *Can you explain baggage rules for Indigo?*")

query = st.text_area("🔍 Enter your query here")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Analyzing your question..."):
            try:
                result = app.invoke({"input": query})
                
                # Debug display
                st.code(result, language="json")

                output = result.get("output", "No response generated.")
                route = result.get("route", "").upper()

                st.success("✅ Here's the response:")
                st.markdown(f"**📤 Answer:**\n{output}")

                if route == "RAG":
                    st.info("📚 This was answered using the RAG knowledge base.")
                elif route == "ML":
                    st.info("🧠 This was predicted using the ML model.")
                else:
                    st.warning("⚠️ Couldn't determine the route taken.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a question before submitting.")
