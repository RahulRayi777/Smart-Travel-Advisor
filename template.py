import os

# Define the folder and file structure
structure = {
    "smart-travel-advisor": {
        "data": {
            "raw_docs": {},
            "cleaned_docs": {},
            "flight_price_data.csv": ""
        },
        "models": {
            "rag_vector_store": {},
            "flight_price_model.joblib": "",
            "explainer_shap.pkl": ""
        },
        "notebooks": {
            "1_EDA_and_Cleaning.ipynb": "",
            "2_Train_ML_Model.ipynb": "",
            "3_RAG_Pipeline.ipynb": "",
            "4_Query_Router_Test.ipynb": ""
        },
        "app": {
            "main.py": "# Streamlit entry point",
            "rag_pipeline.py": "# RAG logic goes here",
            "ml_predictor.py": "# ML prediction logic goes here",
            "query_router.py": "# Intent detection and routing"
        },
        "Dockerfile": "# Docker instructions",
        "requirements.txt": "streamlit\nfaiss-cpu\nopenai\nchromadb\nsentence-transformers\nscikit-learn\nxgboost\nshap\njoblib\npandas\nmatplotlib",
        "README.md": "# Smart Travel Advisor\n\nHybrid Chatbot using RAG + ML"
    }
}


# Function to create the structure
def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        elif content is None:
            open(path, 'a').close()  # Create empty file
        else:
            with open(path, 'w') as f:
                f.write(content)


if __name__ == "__main__":
    create_structure('.', structure)
    print("✅ Project structure created: smart-travel-advisor/")
