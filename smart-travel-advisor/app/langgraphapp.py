# langgraphapp.py

from langgraph.graph import StateGraph
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import TypedDict
import pandas as pd
import joblib
import datetime
import re
import os

import os
from dotenv import load_dotenv

 


# ============================
# ✅ Load Components
# ============================

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore_path = r"C:\Users\naray\OneDrive\Pictures\Desktop\01. My Learning\new\smart-travel-advisor\models\rag_vector_store\travel_docs_index"
vectorstore = FAISS.load_local(folder_path=vectorstore_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

ml_model_path = r"C:\Users\naray\OneDrive\Pictures\Desktop\01. My Learning\new\smart-travel-advisor\models\flight_price_model_xgb.joblib"
model = joblib.load(ml_model_path)

encoders = {
    "Airline": joblib.load(r"C:\Users\naray\OneDrive\Pictures\Desktop\01. My Learning\new\smart-travel-advisor\models\encoders\Airline_encoder.pkl"),
    "Source": joblib.load(r"C:\Users\naray\OneDrive\Pictures\Desktop\01. My Learning\new\smart-travel-advisor\models\encoders\Source_encoder.pkl"),
    "Destination": joblib.load(r"C:\Users\naray\OneDrive\Pictures\Desktop\01. My Learning\new\smart-travel-advisor\models\encoders\Destination_encoder.pkl"),
    "Total_Stops": joblib.load(r"C:\Users\naray\OneDrive\Pictures\Desktop\01. My Learning\new\smart-travel-advisor\models\encoders\Total_Stops_encoder.pkl")
}

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)   


# ============================
# ✅ Schema
# ============================

class TravelState(TypedDict):
    input: str
    output: str
    route: str

# ============================
# ✅ Router Node
# ============================

def router_runnable(state: dict) -> str:
    query = state["input"]
    prompt = f"""
You are an intent classifier for a travel assistant.

Classify the user query into exactly one of the following:
- ML — if it asks for flight prices, travel cost, or any numerical predictions.
- RAG — if it asks for places to visit, information about destinations, travel rules, or general knowledge.

Query: "{query}"

Your answer (just ML or RAG):
"""
    response = llm.invoke(prompt)
    result = response.content.strip().split()[0].upper()
    print("🧭 Routed to:", result)
    return result

# ============================
# ✅ ML Node
# ============================

def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return encoder.transform(["Unknown"])[0]

def parse_query(query):
    query = query.lower()
    cities = ['delhi', 'mumbai', 'bangalore', 'kolkata', 'chennai', 'hyderabad', 'goa', 'pune', 'jaipur']
    airlines = ['indigo', 'air india', 'spicejet', 'goair', 'vistara']

    source = next((c.title() for c in cities if f"from {c}" in query), "Delhi")
    dest = next((c.title() for c in cities if f"to {c}" in query), "Mumbai")
    airline = next((a.title() for a in airlines if a in query), "IndiGo")
    total_stops = "non-stop"

    match = re.search(r'on (\w+ \d{1,2})', query)
    if match:
        date = datetime.datetime.strptime(match.group(1) + " 2025", "%B %d %Y")
    else:
        date = datetime.datetime.today()

    dep_hour, dep_min = 10, 30
    arr_hour, arr_min = 13, 10
    duration = (arr_hour - dep_hour) * 60 + (arr_min - dep_min)

    return pd.DataFrame([{
        'Airline': airline,
        'Source': source,
        'Destination': dest,
        'Total_Stops': total_stops,
        'Journey_day': date.day,
        'Journey_month': date.month,
        'Dep_hour': dep_hour,
        'Dep_min': dep_min,
        'Arrival_hour': arr_hour,
        'Arrival_min': arr_min,
        'Duration_mins': duration
    }])

def ml_node(state):
    try:
        query = state["input"]
        features = parse_query(query)

        for col in ['Airline', 'Source', 'Destination', 'Total_Stops']:
            features[col] = features[col].apply(lambda x: safe_transform(encoders[col], x))

        prediction = model.predict(features)[0]
        result = {
            "input": query,
            "output": f"✈️ Predicted flight fare: ₹{prediction:.2f}",
            "route": "ML"
        }
        print("✅ ML Output:", result)
        return result

    except Exception as e:
        return {
            "input": state["input"],
            "output": f"❌ ML Node Error: {str(e)}",
            "route": "error"
        }

# ============================
# ✅ RAG Node
# ============================

def rag_node(state: TravelState) -> TravelState:
    query = state["input"]
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    rag_prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion:\n{query}"
    response = llm.invoke(rag_prompt)

    result = {
        "input": query,
        "output": f"📚 RAG Answer: {response.content}",
        "route": "RAG"
    }
    print("✅ RAG Output:", result)
    return result

# ============================
# ✅ LangGraph App
# ============================

def pass_input(state: TravelState) -> TravelState:
    return state

builder = StateGraph(TravelState)

builder.add_node("start", pass_input)
builder.add_node("ML", ml_node)
builder.add_node("RAG", rag_node)

builder.set_entry_point("start")

builder.add_conditional_edges(
    "start",
    router_runnable,
    path_map={"ML": "ML", "RAG": "RAG"}
)

builder.set_finish_point("ML")
builder.set_finish_point("RAG")

app = builder.compile()















  