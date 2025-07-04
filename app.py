import os
import uvicorn
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper

app = FastAPI()

# Load model once at startup - can be changed to a smaller model if needed
model = whisper.load_model("tiny")

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict:
    try:
        # Create a temporary file for uploaded audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp:
            # Write uploaded file to temp file
            shutil.copyfileobj(file.file, temp)
            temp.flush()  # Ensure data is written

            # Transcribe audio
            result = model.transcribe(temp.name)
        
        return {"text": result.get("text", "")}

    except Exception as e:
        # Return 500 error with message
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    finally:
        await file.close()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
