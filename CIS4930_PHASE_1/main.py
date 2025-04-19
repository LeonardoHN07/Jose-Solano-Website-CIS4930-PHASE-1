import spacy
import streamlit as st
import os

MODEL_PATH = "custom_trained_model"  # Path to your existing model

ENTITY_COLORS = {
    "GPE": "lightblue",
    "LOC": "lightgreen",
    "ORG": "lightcoral",
    "FAC": "lightyellow",
    "DATE": "lightpink",
    "MONEY": "lightgray",
    "EVENT": "lightgoldenrodyellow",
    "PERSON": "lavender",
    "PRODUCT": "gold",
    "TIME": "lightsteelblue",
}

ENTITY_LABELS = {
    "GPE": "Geopolitical Entity",
    "LOC": "Location",
    "ORG": "Organization",
    "FAC": "Facility",
    "DATE": "Date",
    "MONEY": "Money",
    "EVENT": "Event",
    "PERSON": "Person",
    "PRODUCT": "Product",
    "TIME": "Time",
}

class NERModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        try:
            self.custom_nlp = spacy.load(model_path)
            self.default_nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            st.stop()

    def highlight_entities_html(self, text, entity_vars):
        doc_custom = self.custom_nlp(text)
        doc_default = self.default_nlp(text)
        all_ents = sorted(doc_custom.ents + doc_default.ents, key=lambda e: e.start_char)

        html = ""
        last_end = 0
        for ent in all_ents:
            html += text[last_end:ent.start_char]
            if ent.label_ in entity_vars and entity_vars[ent.label_]:
                color = ENTITY_COLORS.get(ent.label_, "#DDDDDD")
                html += f"<mark style='background-color: {color}'>{ent.text} <span style='font-size:small;color:gray;'>({ent.label_})</span></mark>"
            else:
                html += ent.text
            last_end = ent.end_char
        html += text[last_end:]
        return html

# --- Streamlit UI ---
model = NERModel()

st.set_page_config(page_title="Travel & Tourism NER Highlighter", layout="wide")
st.title("🧳 Travel & Tourism NER Highlighter")
st.markdown("Using pre-trained model to highlight travel domain entities")

text_input = st.text_area("Enter text:", height=200)

st.subheader("Select entities to highlight:")
selected_labels = {}
cols = st.columns(2)
for i, (label, full) in enumerate(ENTITY_LABELS.items()):
    with cols[i % 2]:
        selected_labels[label] = st.checkbox(f"{full}", value=True)

if text_input:
    html_output = model.highlight_entities_html(text_input, selected_labels)
    st.subheader("Highlighted Output:")
    st.markdown(f"<div style='font-family:Arial; font-size:16px'>{html_output}</div>", unsafe_allow_html=True)
else:
    st.warning("Please enter some text to analyze.")

