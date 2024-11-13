import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load BART model and tokenizer
@st.cache_resource  # Cache the model and tokenizer to avoid reloading
def load_model():
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

main_query = "Ask me to explain the scenario, based on the scenario ask questions as plaintiff should have each and every little detail of the case. Ask questions till you are satisfied to prepare the plaintiff notice. Once you have the answers needed, prepare a plaintiff statement."

if "questions" not in st.session_state:
    st.session_state.questions = ["Please explain the scenario in detail:"]
    st.session_state.responses = []
    st.session_state.complete = False

st.title("Dynamic Plaintiff Notice Preparation")

# Generate the next question based on user's response
def generate_next_question(context):
    inputs = tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
    next_question = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return next_question

if not st.session_state.complete:
    current_question = st.session_state.questions[-1]
    user_answer = st.text_input(current_question, key=current_question)

    if user_answer:
        st.session_state.responses.append(user_answer)
        context = " ".join([f"{q}: {a}" for q, a in zip(st.session_state.questions, st.session_state.responses)])
        next_question = generate_next_question(context)
        
        if "I'm satisfied with the information" in next_question:
            st.session_state.complete = True
        else:
            st.session_state.questions.append(next_question)

if st.session_state.complete:
    st.subheader("Plaintiff Notice:")
    plaintiff_statement = f"PLAINTIFF NOTICE\n\nRegarding the scenario: {main_query}\n\n"
    for question, answer in zip(st.session_state.questions, st.session_state.responses):
        plaintiff_statement += f"{question}\n{answer}\n\n"
    plaintiff_statement += "This notice has been prepared based on the information provided above."
    st.write(plaintiff_statement)
else:
    st.write("Please provide responses to prepare a detailed plaintiff notice.")
