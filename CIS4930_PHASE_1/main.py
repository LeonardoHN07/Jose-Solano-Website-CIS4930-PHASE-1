import spacy
from spacy.training import Example
import streamlit as st
from trainingModel import TRAINING_DATA

MODEL_PATH = "custom_trained_model"

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
        self.custom_nlp = self.load_model()
        self.default_nlp = spacy.load("en_core_web_sm")

    def load_model(self):
        try:
            return spacy.load(self.model_path)
        except:
            return spacy.load("en_core_web_sm")

    def train_model(self):
        if "ner" not in self.custom_nlp.pipe_names:
            ner = self.custom_nlp.add_pipe("ner")
        else:
            ner = self.custom_nlp.get_pipe("ner")

        for text, annotations in TRAINING_DATA:
            for start, end, label in annotations["entities"]:
                ner.add_label(label)

        examples = [
            Example.from_dict(self.custom_nlp.make_doc(text), annotations)
            for text, annotations in TRAINING_DATA
        ]

        unaffected_pipes = [pipe for pipe in self.custom_nlp.pipe_names if pipe != "ner"]
        with self.custom_nlp.disable_pipes(*unaffected_pipes):
            optimizer = self.custom_nlp.resume_training()
            for epoch in range(10):
                for example in examples:
                    self.custom_nlp.update([example], drop=0.5, sgd=optimizer)

        self.custom_nlp.to_disk(self.model_path)
        self.custom_nlp = self.load_model()

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
st.markdown("Enter text to highlight **custom Named Entities** from the travel domain.")

text_input = st.text_area("Enter text:", height=200)

st.subheader("Select entities to highlight:")
selected_labels = {}
cols = st.columns(2)
i = 0
for label, full in ENTITY_LABELS.items():
    with cols[i % 2]:
        selected_labels[label] = st.checkbox(f"{full}", value=True)
    i += 1

if st.button("Train & Highlight Entities"):
    model.train_model()
    html_output = model.highlight_entities_html(text_input, selected_labels)
    st.subheader("Highlighted Output:")
    st.markdown(f"<div style='font-family:Arial; font-size:16px'>{html_output}</div>", unsafe_allow_html=True)

