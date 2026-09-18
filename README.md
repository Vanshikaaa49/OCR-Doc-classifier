# OCR-Doc-classifier
A Python-based OCR tool that extracts text from images and scanned documents using [Tesseract/EasyOCR/etc.], with preprocessing for noise reduction and layout detection.

OCR Text Extraction Tool

A Python-based OCR (Optical Character Recognition) tool that extracts text from images and scanned documents. It preprocesses images for better accuracy and supports multiple output formats.

# Features  
📄 Extract text from images (JPG, PNG, TIFF) and scanned PDFs
🧹 Image preprocessing — grayscale conversion, noise removal, thresholding
🌐 Multi-language text recognition
📦 Batch processing for multiple files at once
💾 Export results as plain text, JSON, or searchable PDF
Tech Stack
Language: Python 3.x
OCR Engine: Tesseract OCR (via pytesseract)
Image Processing: OpenCV, Pillow
PDF Handling: pdf2image / PyMuPDF
