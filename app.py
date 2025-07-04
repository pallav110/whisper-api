from fastapi import FastAPI, UploadFile, File
import whisper
import shutil

app = FastAPI()
model = whisper.load_model("large")  # You can choose 'tiny', 'base', 'small', 'medium', 'large'

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_file = "temp_audio.mp3"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model.transcribe(temp_file)
    return {"text": result["text"]}
