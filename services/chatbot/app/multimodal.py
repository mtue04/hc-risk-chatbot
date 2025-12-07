"""
Multimodal Processing Module for the HomeCredit Risk Chatbot.

This module enables the chatbot to process:
1. Voice input → Transcription (Google Speech-to-Text)
2. Image input → Document data extraction (Google Vision API or Gemini Vision)

Requires Google Cloud credentials with Speech-to-Text and Vision API enabled.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

# Google Cloud configuration
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class MultimodalProcessor:
    """
    Process multimodal inputs (voice, images) for the chatbot.
    
    Uses Google Cloud services for transcription and OCR.
    Falls back to Gemini Vision for image understanding if Vision API unavailable.
    """
    
    def __init__(self):
        self._speech_client = None
        self._vision_client = None
        self._gemini_model = None
        
    def _init_speech_client(self):
        """Lazy initialization of Speech-to-Text client."""
        if self._speech_client is None:
            try:
                from google.cloud import speech
                self._speech_client = speech.SpeechClient()
                logger.info("Google Speech-to-Text client initialized")
            except ImportError:
                logger.warning("google-cloud-speech not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Speech client: {e}")
        return self._speech_client
    
    def _init_vision_client(self):
        """Lazy initialization of Vision API client."""
        if self._vision_client is None:
            try:
                from google.cloud import vision
                self._vision_client = vision.ImageAnnotatorClient()
                logger.info("Google Vision API client initialized")
            except ImportError:
                logger.warning("google-cloud-vision not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Vision client: {e}")
        return self._vision_client
    
    def _init_gemini_vision(self):
        """Lazy initialization of Gemini for vision tasks."""
        if self._gemini_model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                logger.info("Gemini Vision model initialized")
            except ImportError:
                logger.warning("google-generativeai not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Vision: {e}")
        return self._gemini_model
    
    async def extract_text_from_file(
        self, 
        file_bytes: bytes, 
        filename: str
    ) -> dict[str, Any]:
        """
        Extract text content from various file formats.
        
        Supports: PDF, TXT, DOCX, DOC
        """
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        
        try:
            if ext == "txt":
                text = file_bytes.decode("utf-8", errors="ignore")
                return {"text": text, "file_type": "txt"}
                
            elif ext == "pdf":
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
                    return {"text": text, "file_type": "pdf", "pages": len(pdf_reader.pages)}
                except ImportError:
                    return {"error": "PyPDF2 not installed", "file_type": "pdf"}
                    
            elif ext in ["docx", "doc"]:
                try:
                    from docx import Document
                    doc = Document(io.BytesIO(file_bytes))
                    text = "\n".join([para.text for para in doc.paragraphs])
                    return {"text": text, "file_type": "docx"}
                except ImportError:
                    return {"error": "python-docx not installed", "file_type": "docx"}
                    
            elif ext in ["png", "jpg", "jpeg"]:
                return await self.extract_document_data(file_bytes, document_type="auto")
                
            else:
                return {"error": f"Unsupported file type: {ext}"}
                
        except Exception as e:
            logger.error(f"File extraction error: {e}")
            return {"error": str(e)}
    
    async def transcribe_audio(
        self, 
        audio_bytes: bytes, 
        language_code: str = "vi-VN"
    ) -> dict[str, Any]:
        """
        Transcribe audio to text using Google Speech-to-Text.
        
        Args:
            audio_bytes: Raw audio data (WAV, MP3, OGG, or WEBM)
            language_code: Language code (default: Vietnamese)
            
        Returns:
            Dict with transcription text and confidence score
        """
        client = self._init_speech_client()
        
        if client is None:
            return {
                "error": "Speech-to-Text service not available",
                "suggestion": "Install google-cloud-speech and set GOOGLE_APPLICATION_CREDENTIALS"
            }
        
        try:
            from google.cloud import speech
            
            audio = speech.RecognitionAudio(content=audio_bytes)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model="latest_long",  # Better for conversational speech
            )
            
            response = client.recognize(config=config, audio=audio)
            
            if not response.results:
                return {
                    "text": "",
                    "confidence": 0,
                    "message": "No speech detected in audio"
                }
            
            # Get the best result
            best_result = response.results[0].alternatives[0]
            
            logger.info(
                "audio_transcribed",
                text_length=len(best_result.transcript),
                confidence=best_result.confidence
            )
            
            return {
                "text": best_result.transcript,
                "confidence": best_result.confidence,
                "language": language_code,
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "error": f"Transcription failed: {str(e)}",
                "text": ""
            }
    
    async def extract_document_data(
        self, 
        image_bytes: bytes,
        document_type: str = "auto"
    ) -> dict[str, Any]:
        """
        Extract structured data from document images.
        
        Supports:
        - ID cards (CCCD/CMND)
        - Bank statements
        - Income documents
        - Loan applications
        
        Args:
            image_bytes: Image data (PNG, JPEG)
            document_type: Type hint for extraction ("id_card", "financial", "auto")
            
        Returns:
            Dict with extracted fields and confidence scores
        """
        # Try Vision API first for OCR
        vision_client = self._init_vision_client()
        
        extracted_text = ""
        if vision_client:
            try:
                from google.cloud import vision
                
                image = vision.Image(content=image_bytes)
                response = vision_client.document_text_detection(image=image)
                
                if response.full_text_annotation:
                    extracted_text = response.full_text_annotation.text
                    logger.info(f"OCR extracted {len(extracted_text)} characters")
                    
            except Exception as e:
                logger.warning(f"Vision API OCR failed: {e}")
        
        # Use Gemini Vision for intelligent extraction
        gemini = self._init_gemini_vision()
        
        if gemini is None:
            if extracted_text:
                return {
                    "raw_text": extracted_text,
                    "message": "OCR completed but intelligent extraction unavailable"
                }
            return {
                "error": "No vision processing available",
                "suggestion": "Install google-generativeai or google-cloud-vision"
            }
        
        try:
            import google.generativeai as genai
            from PIL import Image
            
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Create extraction prompt based on document type
            if document_type == "id_card":
                prompt = """Analyze this ID card image and extract:
                - Full name
                - ID number
                - Date of birth
                - Address
                - Issue date
                
                Return as JSON format with field names in English.
                If a field is not visible, set it to null."""
            elif document_type == "financial":
                prompt = """Analyze this financial document and extract:
                - Document type (salary slip, bank statement, tax document)
                - Name
                - Monthly/yearly income
                - Currency
                - Date/period
                
                Return as JSON format. Convert amounts to numbers."""
            else:
                prompt = """Analyze this document image and extract all relevant information.
                If it's an ID card, extract personal details.
                If it's a financial document, extract income/amount information.
                Return as JSON format with clear field names."""
            
            if extracted_text:
                prompt += f"\n\nOCR extracted text for reference:\n{extracted_text}"
            
            response = gemini.generate_content([prompt, img])
            
            logger.info("document_data_extracted", document_type=document_type)
            
            # Try to parse as JSON
            import json
            try:
                # Clean up markdown formatting if present
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                extracted_data = json.loads(text.strip())
                return {
                    "data": extracted_data,
                    "document_type": document_type,
                    "raw_text": extracted_text if extracted_text else None,
                }
            except json.JSONDecodeError:
                return {
                    "raw_response": response.text,
                    "raw_text": extracted_text if extracted_text else None,
                    "document_type": document_type,
                }
                
        except Exception as e:
            logger.error(f"Document extraction error: {e}")
            return {
                "error": f"Extraction failed: {str(e)}",
                "raw_text": extracted_text if extracted_text else None,
            }
    
    async def analyze_image_with_gemini(
        self, 
        image_bytes: bytes, 
        query: str
    ) -> str:
        """
        Use Gemini Vision for general image understanding.
        
        Args:
            image_bytes: Image data
            query: Question about the image
            
        Returns:
            Text response from Gemini
        """
        gemini = self._init_gemini_vision()
        
        if gemini is None:
            return "Image analysis not available. Please install google-generativeai."
        
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            
            response = gemini.generate_content([query, img])
            return response.text
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"Image analysis failed: {str(e)}"


# Singleton instance
multimodal_processor = MultimodalProcessor()


def get_multimodal_processor() -> MultimodalProcessor:
    """Get the multimodal processor instance."""
    return multimodal_processor
