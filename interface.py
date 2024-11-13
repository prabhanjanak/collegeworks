import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load BART model and tokenizer for conditional generation
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Preloaded main query
main_query = "Ask me to explain the scenario, based on the scenario ask questions as plaintiff should have each and every little detail of the case. Ask questions till you are satisfied to prepare the plaintiff notice. Once you have the answers needed, prepare a plaintiff statement."

# Initialize session state to keep track of the conversation
if "questions" not in st.session_state:
    st.session_state.questions = ["Please explain the scenario in detail:"]
    st.session_state.responses = []
    st.session_state.complete = False

st.title("Dynamic Plaintiff Notice Preparation")

# Function to generate next question based on the user's responses
def generate_next_question(context):
    # Encode the context and generate the next question
    inputs = tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
    next_question = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return next_question

# Display the current question and collect user's answer
if not st.session_state.complete:
    # Show the last question in the session state
    current_question = st.session_state.questions[-1]
    user_answer = st.text_input(current_question, key=current_question)

    # If an answer is provided, process it and generate the next question
    if user_answer:
        # Append the user's answer to responses
        st.session_state.responses.append(user_answer)
        
        # Generate context for the next question
        context = " ".join([f"{q}: {a}" for q, a in zip(st.session_state.questions, st.session_state.responses)])
        
        # Generate the next question based on the accumulated context
        next_question = generate_next_question(context)
        
        # Append the next question or stop if satisfied
        if "I'm satisfied with the information" in next_question:
            st.session_state.complete = True
        else:
            st.session_state.questions.append(next_question)
        
# Generate the plaintiff statement once all details are gathered
if st.session_state.complete:
    st.subheader("Plaintiff Notice:")
    plaintiff_statement = f"PLAINTIFF NOTICE\n\nRegarding the scenario: {main_query}\n\n"
    for question, answer in zip(st.session_state.questions, st.session_state.responses):
        plaintiff_statement += f"{question}\n{answer}\n\n"
    plaintiff_statement += "This notice has been prepared based on the information provided above."

    st.write(plaintiff_statement)
else:
    st.write("Please provide responses to prepare a detailed plaintiff notice.")
