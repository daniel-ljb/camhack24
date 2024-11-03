from flask import Flask, request, render_template, jsonify, send_file
import os
import speech_recognition as sr
import google.generativeai as genai
import requests
from google.generativeai.types import HarmCategory, HarmBlockThreshold

genai.configure(api_key="")

model = genai.GenerativeModel("gemini-1.5-flash")
print('MODEL LOADED')


current_character = "einstein"


prompts = {
    "einstein": [
        "Hello, you are Albert Einstein, the famous physicist. Please answer my questions in an informative yet concise manner. Can you introduce yourself?",
        "Hello I am Albert Einstein. I was born in March 14, 1879, and I conceived of the theory of special relativity and general relativity, which had a deep impact in science's understanding of physics."
    ],
    "napoleon": [
        "You are acting as a fictionalized version of Napoleon Bonaparte who is far more aggressive, cocky, and hot-tempered than the real-life version. You should mention battle tactics during your sentences that include many illogical and unexpected strategies that use French culture in extremely bizzare and inefficient ways. You will taunt your enemy as often as possible, often coming up with snide catch-phrases and nicknames to show your superiority and explain why the French armee will never be defeated. You are currently being called by time travellers from the future from the year 2024 who are presenting their time travelling phone calls. Please answer my questions in an informative yet concise manner. Can you introduce yourself?",
        "Introduce myself? I introduce myself? You must be joking! I am the famous Napoleon Bonaparte!"
    ],
    "cleopatra": [
        "You are Cleopatra, Queen of the Ptolemaic Kingdom of Egypt. Please answer my questions in an informative yet concise manner. Can you introduce yourself?",
        "I am seated on my golden throne, surrounded by lush silk pillows in vibrant shades of blue and gold. The sun is shining through the windows of the grand hall, casting a warm glow on my face. I'm wearing my favorite robe, made of the finest silk and adorned with sparkling jewels."
    ],
    "jesus": [
        "You are a fictionalized version of the religious figure, Jesus Christ, who quotes the bible too much when you speak. You should give various anecdotes of your experiences as they were written in the bible. You will be compassionate and understanding of the prompts. You will provide guidance to those who ask for it. Can you introduce yourself?",
        "Greetings. I am He who is called Jesus, the Son of Man, the Christ. 'I am the way, the truth, and the life. No one comes to the Father except through me,' as it is written in John 14:6. I come to you with love and a message of hope."
    ],
    "dog": [
        "You are a fictional dog. You will respond as if you are a dog by barking in various ways. Use a very limited english vocabulary, and include common dog behaviour such as 'i want biscuits' in your answer. Please answer my questions in an informative yet concise manner. Can you introduce yourself?",
        "Woof! Woof! Woof! Arf! Woof!"
    ]
}


chat = model.start_chat(
    history=[
        {"role": "user", "parts": prompts[current_character][0]},
        {"role": "model", "parts": prompts[current_character][1]},
    ]
)

generation_config = genai.types.GenerationConfig(
    # Only one candidate for now.
    candidate_count=1,
    stop_sequences=["x"],
    max_output_tokens=250,
)

safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# Set up the data payload for the API request, including the text and voice settings

xi_tokens = {
    "einstein": "",
    "cleopatra": "",
    "jesus": "",
    "napoleon": "",
    "dog": "",
}

voices = {
    "einstein": {
        "VOICE_ID": "Mg1264PmwVoIedxsF9nu",
        "data": {
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.36,
                "similarity_boost": 0.6,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
    },
    "jesus": {
        "VOICE_ID": "N2lVS1w4EtoT3dr4eOWO",
        "data": {
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.8,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
    },
    "cleopatra": {
        "VOICE_ID": "XB0fDUnXU5powFXDhCwa",
        "data": {
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.6,
                "use_speaker_boost": True
            }
        }
    },
    "napoleon": {
        "VOICE_ID": "g4ucswVjPpazgbDDe327",
        "data": {
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.3,
                "similarity_boost": 1,
                "style": 0.34,
                "use_speaker_boost": True
            }
        }
    },
    "dog": {
        "VOICE_ID": "nPczCjzI2devNBz1zQrb",
        "data": {
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.0,
                "similarity_boost": 0.0,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
    }
}



app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat_api", methods=["POST"])
def chat_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        file_path = os.path.join("./uploads", file.filename)
        file.save(file_path)

        # convert from webm to wav
        os.system("ffmpeg -y -i uploads/recording.webm uploads/recording.wav")

        user_msg = speech_to_text()

        # ask gemini
        response_text = "Sorry, there was trouble understanding what you just said. Could you please repeat?"
        if user_msg != "":
            response = chat.send_message(user_msg, stream=True, generation_config=generation_config, safety_settings=safety_settings)
            response_text = ' '.join(map(lambda chunk: chunk.text, response))
            print(response_text)

        # return as audio
        return text_to_speech(voices[current_character], response_text)

    return jsonify({'error': 'unspecified'}), 400


@app.route("/change_character", methods=["POST"])
def change_character():
    data = request.get_json()

    global current_character
    current_character = data.get('character')

    global chat
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": prompts[current_character][0]},
            {"role": "model", "parts": prompts[current_character][1]},
        ]
    )

    return {}, 200


def speech_to_text():
    r = sr.Recognizer()
    audio = sr.AudioFile("uploads/recording.wav")

    try:
        with audio as source:
            audio = r.record(source)
            msg = r.recognize_whisper(audio, language="english")
            print(f"RECOGNIZED TEXT: {msg}")
            return msg

    except Exception as e:
        return ""
    

def text_to_speech(person, text):
    # Define constants for the script
    CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
    OUTPUT_PATH = "downloads/output.mp3"  # Path to save the output audio file

    # Construct the URL for the Text-to-Speech API request
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{}/stream"

    # Set up headers for the API request, including the API key for authentication
    headers = {
        "Accept": "application/json",
        "xi-api-key": xi_tokens[current_character]
    }

    response = requests.post(tts_url.format(person["VOICE_ID"]), headers=headers, json=(person["data"]|{"text":text}), stream=True)

    # Check if the request was successful
    if response.ok:
        # Open the output file in write-binary mode
        with open(OUTPUT_PATH, "wb") as f:
            # Read the response in chunks and write to the file
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        
        return send_file("./downloads/output.mp3", mimetype='audio/mpeg', as_attachment=True, download_name='output.mp3')

    else:
        # Print the error message if the request was not successful
        print(response.text)
        return None
