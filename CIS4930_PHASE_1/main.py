import spacy
from spacy.training import Example
import streamlit as st
#from trainingModel import TRAINING_DATA
import os
import random

TRAINING_DATA = \
[
   ("Tourists can now access different sources of information, and they can generate their own content and share their views and experiences. ", {"entities": [(0, 8, "PERSON")]}), # tourist
   ("In addition, travel is one of the dominant topics on social media, for example on Facebook and Twitter (Neidhardt et al., 2017; Travelmail Reporter,2013).",{"entities": [(95, 102, "ORGANIZATION")]} ),
   ("Consider the local communities whose rich cultures are rooted in nature, beautiful landscapes, coastal resorts, majestic mountain ranges, and the exquisite diversity of wildlife.", {"entities": [(73, 93, "LOCATION")]}),
   ("Consider the local communities whose rich cultures are rooted in nature, beautiful landscapes, coastal resorts, majestic mountain ranges, and the exquisite diversity of wildlife.",{"entities": [(95, 110, "LOCATION")]}),
   ("Although travelers from Mexico are physically on U.S. soil when they visit the United States, the goods and services they consume while in America are U.S. exports.",{"entities": [(9, 18, "PERSON")]} ),
   ("Gumbalimba Park is a family-friendly attraction that offers its visitors to Roatan the opportunity to mingle with friendly white-face Capuchin monkeys and free-flying exotic birds (Including macaws, parrots, and hummingbirds). ", {"entities": [(37,47,"EVENT")]}),
   ("Gumbalimba Park is a family-friendly attraction that offers its visitors to Roatan the opportunity to mingle with friendly white-face Capuchin monkeys and free-flying exotic birds (Including macaws, parrots, and hummingbirds). ", {"entities": [(64,72, "PERSON")]}),
   ("Visitors can spot indigenous lizards and iguanas all around the park.", {"entities": [(0,8,"PERSON")]}),
   ("in which includes travel spending by Mexican visitors to the U.S, education-related expenses by Mexicans in the U.S.", {"entities": [(37,44,"PERSON")]} ),
   ("The Los Glaciares National Park is an area of exceptional natural beauty, with rugged, towering mountains and numerous glacial lakes, including Lake Argentino, which is 160 km long.", {"entities": [(144,158, "LOCATION")]}),
   ("This study aims to investigate the potential tourism value of Debre Aron Monastery, which is one of the most overlooked religious sites in northern Ethiopia.", {"entities": [(120,135, "LOCATION")]}),
   ("Whether youre storing up to 77 beer cans or freshly caught fish, the Tundra 65 is more than capable of keeping its cargo safe and cold.", {"entities": [(31,35, "PRODUCT")]}),
   ("This seven-piece set includes one large, three medium, and two small packing cubes, plus a shoe bag that’ll help you organize your items and maximize the space in your luggage.", {"entities": [(168,176,"PRODUCT")]}),
   ("plus a shoe bag that’ll help you organize your items and maximize the space in your luggage.", {"entities": [(7,15,"PRODUCT")]}),
   ("This Cabeau Evolution S2 Travel Pillow will help keep you comfy with its plush, dual-density memory foam material.",{"entities": [(32,38, "PRODUCT")]} ),
   ("For instance, around one in three travellers in the US, Saudi Arabia, and the UAE said they were likely to fly in first or business class, compared to the global average of just one in five28.",{"entities": [(34,44, "PERSON")]}), # travellers # british spelling...
   ("International travel is the more popular choice for luxury travellers who book premium flight cabins, as nearly 82% of business and first-class flight bookings on Trip.com in 2022 were for international routes. ", {"entities": [(189,202,"EVENT")]}), # International
   ("A Skyscanner survey also reported a strong demand for premium cabins, particularly in markets with higher GDP per capita. ", {"entities": [(54,68,"PRODUCT")]}), # Premium cabins
   ("According to ForwardKeys data, passengers in premium flight cabins were around 60% below 2019 levels in the first quarter of 2022. ", {"entities": [(31,41,"PERSON")]}), # Passengers
   ("International travel is the more popular choice for luxury travellers who book premium flight cabins, as nearly 82% of business and first-class flight bookings on Trip.com in 2022 were for international routes. ", {"entities": [(87,93,"PRODUCT")]}), #flight

]

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
        try:
            self.default_nlp = spacy.load("en_core_web_sm")
        except:
            st.error("en_core_web_sm model not found. Please install it with: python -m spacy download en_core_web_sm")
            st.stop()

    def load_model(self):
        try:
            return spacy.load(self.model_path)
        except:
            # Create a blank model if custom model doesn't exist
            nlp = spacy.blank("en")
            if "ner" not in nlp.pipe_names:
                nlp.add_pipe("ner")
            return nlp

    def train_model(self):
        # Create a blank English model if it doesn't exist
        if not hasattr(self, 'custom_nlp') or self.custom_nlp is None:
            self.custom_nlp = spacy.blank("en")

        # Add NER pipe if it doesn't exist
        if "ner" not in self.custom_nlp.pipe_names:
            ner = self.custom_nlp.add_pipe("ner")
        else:
            ner = self.custom_nlp.get_pipe("ner")

        # Add all labels from the training data
        labels = set()
        for _, annotations in TRAINING_DATA:
            for _, _, label in annotations["entities"]:
                labels.add(label)
        for label in labels:
            ner.add_label(label)

        # Disable other pipes during training
        other_pipes = [pipe for pipe in self.custom_nlp.pipe_names if pipe != "ner"]
        with self.custom_nlp.disable_pipes(*other_pipes):
            # Initialize the model with random weights
            optimizer = self.custom_nlp.initialize()

            # Convert training data to spaCy's Example format
            examples = []
            for text, annotations in TRAINING_DATA:
                doc = self.custom_nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)

            # Train for 30 iterations
            for itn in range(30):
                random.shuffle(examples)
                losses = {}
                for batch in spacy.util.minibatch(examples, size=8):
                    self.custom_nlp.update(batch, drop=0.5, losses=losses, sgd=optimizer)
                print(f"Iteration {itn}, Losses: {losses}")  # For debugging

        # Save the model
        os.makedirs(self.model_path, exist_ok=True)
        self.custom_nlp.to_disk(self.model_path)
        self.custom_nlp = spacy.load(self.model_path)

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
    if text_input:
        html_output = model.highlight_entities_html(text_input, selected_labels)
        st.subheader("Highlighted Output:")
        st.markdown(f"<div style='font-family:Arial; font-size:16px'>{html_output}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

