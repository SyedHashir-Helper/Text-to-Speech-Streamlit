import streamlit as st
import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# 
# Load your model here
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")
    return model, tokenizer, device

# Function to perform text-to-speech conversion
def text_to_speech(model, tokenizer, device, prompt, description):
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, model.config.sampling_rate

# Streamlit App
def main():
    st.title("Text-to-Speech Generator")

    # Load the TTS model
    model, tokenizer, device = load_model()

    # Text input
    prompt_input = st.text_input("Enter prompt:", value="Hey, how are you doing today?")
    description_input = st.text_area("Enter description:", 
                                     value="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.")

    # Button to generate speech
    if st.button("Generate Speech"):
        if prompt_input and description_input:
            st.write("Generating speech...")
            audio_output, sampling_rate = text_to_speech(model, tokenizer, device, prompt_input, description_input)
            
            # Save the audio output to a file
            audio_file = "parler_tts_out.wav"
            sf.write(audio_file, audio_output, sampling_rate)

            # Display the generated speech
            st.audio(audio_file, format="audio/wav")
        else:
            st.write("Please enter both prompt and description.")

if __name__ == "__main__":
    main()
