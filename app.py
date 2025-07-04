import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
import whisper
import shutil

app = FastAPI()
model = whisper.load_model("large")  # or your choice

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_file = "temp_audio.mp3"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model.transcribe(temp_file)
    return {"text": result["text"]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
