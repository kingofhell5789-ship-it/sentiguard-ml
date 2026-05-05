import os
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip
import cv2
import librosa
import numpy as np
from transformers import pipeline
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class ReliabilityGate:
    """Manages dynamic weighting based on audio reliability."""
    
    def __init__(self, noise_threshold_low=0.01, noise_threshold_high=0.5):
        self.noise_threshold_low = noise_threshold_low
        self.noise_threshold_high = noise_threshold_high
        self.interference_detected = False
        self.weights = {"text": 0.4, "audio": 0.3, "visual": 0.3}
        
    def check_audio_quality(self, rms_energy, zcr):
        """
        Check if audio quality is within acceptable range.
        
        Args:
            rms_energy: Root Mean Square energy of audio
            zcr: Zero Crossing Rate
            
        Returns:
            bool: True if interference detected, False otherwise
        """
        if rms_energy < self.noise_threshold_low or rms_energy > self.noise_threshold_high:
            self.interference_detected = True
            logger.warning(f"AUDIO INTERFERENCE DETECTED: RMS={rms_energy:.4f}, ZCR={zcr:.4f}")
            return True
        return False
    
    def get_weights(self):
        """Get dynamic weights based on interference status."""
        if self.interference_detected:
            self.weights = {"text": 0.7, "audio": 0.05, "visual": 0.25}
            logger.info("Weight shift applied: Prioritizing text (0.7) due to audio interference")
        else:
            self.weights = {"text": 0.4, "audio": 0.3, "visual": 0.3}
            logger.info("Normal weights applied: Balanced multimodal fusion")
        return self.weights
    
    def fuse_sentiments(self, text_score, audio_score, visual_score):
        """Fuse sentiment scores using dynamic weights."""
        weights = self.get_weights()
        final_score = (
            text_score * weights["text"] +
            audio_score * weights["audio"] +
            visual_score * weights["visual"]
        )
        return final_score

def extract_audio(video_path, output_audio_path="temp_audio.wav"):
    """Extract audio from video file."""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(output_audio_path, verbose=False, logger=None)
            video.close()
            return output_audio_path
        video.close()
        return None
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return None

def extract_frame(video_path):
    """Extract middle frame from video."""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        return None
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        return None

def analyze_audio(audio_path):
    """
    Analyze audio using librosa.
    Returns: sentiment score (0-1 based on audio characteristics)
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Calculate RMS Energy
        rms_energy = np.sqrt(np.mean(y**2))
        
        # Calculate Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        logger.info(f"Audio Analysis: RMS Energy={rms_energy:.4f}, ZCR={zcr:.4f}")
        
        # Simple sentiment approximation from audio energy
        # Higher energy + higher ZCR might indicate more emotional speech
        audio_sentiment = (rms_energy + zcr) / 2
        audio_sentiment = min(1.0, max(0.0, audio_sentiment))
        
        return {
            "score": audio_sentiment,
            "rms_energy": rms_energy,
            "zcr": zcr
        }
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        return {"score": 0.5, "rms_energy": 0.0, "zcr": 0.0}

def analyze_text(text):
    """Analyze text sentiment using transformer model."""
    try:
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for analysis")
            return {"score": 0.5, "label": "NEUTRAL", "confidence": 0.0}
        
        result = sentiment_pipeline(text[:512])[0]  # Limit to 512 chars
        
        score = result["score"]
        label = result["label"]
        
        # Convert to 0-1 scale where 1 is positive
        if label == "POSITIVE":
            text_sentiment = score
        else:
            text_sentiment = 1 - score
        
        logger.info(f"Text Analysis: Label={label}, Score={text_sentiment:.4f}")
        
        return {
            "score": text_sentiment,
            "label": label,
            "confidence": score
        }
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return {"score": 0.5, "label": "NEUTRAL", "confidence": 0.0}

def analyze_visual(frame):
    """
    Analyze visual sentiment from frame.
    For production, integrate DeepFace or FER2013.
    Currently returns mock analysis.
    """
    try:
        if frame is None:
            return {"score": 0.5, "emotion": "NEUTRAL"}
        
        # Mock visual emotion detection
        # In production, use: from deepface import DeepFace
        # result = DeepFace.analyze(frame, actions=['emotion'])
        
        logger.info("Visual Analysis: Using mock emotion detection")
        
        return {
            "score": 0.5,
            "emotion": "NEUTRAL"
        }
    except Exception as e:
        logger.error(f"Error analyzing visual: {str(e)}")
        return {"score": 0.5, "emotion": "NEUTRAL"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Main API endpoint for multimodal sentiment analysis.
    Accepts MP4 video file and returns fused sentiment with reliability metrics.
    """
    temp_video_path = f"temp_{file.filename}"
    temp_audio_path = "temp_audio.wav"
    
    try:
        # Save uploaded file
        with open(temp_video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Processing video: {file.filename}")
        
        # Initialize Reliability Gate
        gate = ReliabilityGate()
        
        # Extract audio
        audio_path = extract_audio(temp_video_path, temp_audio_path)
        
        # Extract frame
        frame = extract_frame(temp_video_path)
        
        # Analyze modalities
        audio_result = analyze_audio(audio_path) if audio_path else {"score": 0.5, "rms_energy": 0.0, "zcr": 0.0}
        
        # Check audio quality and trigger Reliability Gate
        gate.check_audio_quality(audio_result["rms_energy"], audio_result["zcr"])
        
        # Mock text from transcription (in production, use speech-to-text)
        mock_text = "This is a test sentiment analysis."
        text_result = analyze_text(mock_text)
        
        visual_result = analyze_visual(frame)
        
        # Fuse sentiments with dynamic weighting
        final_sentiment = gate.fuse_sentiments(
            text_result["score"],
            audio_result["score"],
            visual_result["score"]
        )
        
        response = {
            "filename": file.filename,
            "modalities": {
                "text": {
                    "score": float(text_result["score"]),
                    "label": text_result["label"]
                },
                "audio": {
                    "score": float(audio_result["score"]),
                    "rms_energy": float(audio_result["rms_energy"]),
                    "zcr": float(audio_result["zcr"])
                },
                "visual": {
                    "score": float(visual_result["score"]),
                    "emotion": visual_result["emotion"]
                }
            },
            "reliability": {
                "interference_detected": gate.interference_detected,
                "weights_applied": gate.get_weights()
            },
            "final_sentiment": {
                "score": float(final_sentiment),
                "label": "POSITIVE" if final_sentiment > 0.5 else "NEGATIVE"
            }
        }
        
        logger.info(f"Analysis complete. Final sentiment score: {final_sentiment:.4f}")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # Cleanup temp files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)