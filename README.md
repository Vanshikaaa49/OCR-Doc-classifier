# OCR-Doc-classifier
A Python-based OCR tool that extracts text from images and scanned documents using [Tesseract/EasyOCR/etc.], with preprocessing for noise reduction and layout detection.

OCR-Doc-Classifier

A custom OCR pipeline built from scratch that reads document category labels (email, resume, scientific_publication) directly from document images, using a CRNN (CNN + Bidirectional LSTM) architecture trained with CTC loss.

This is not a wrapper around Tesseract/EasyOCR — the model, training loop, and decoding logic were built and debugged from the ground up.

Result

81% exact-match accuracy on a held-out validation set, after several rounds of debugging and tuning.

Dataset
165 labeled document images, loaded from a zipped CSV (image_path, text_label columns)
3 classes: email, resume, scientific_publication
Split 80/20 train/validation
Pipeline
Image preprocessing — grayscale conversion, resize to 128×32, pixel normalization
Character-level vocabulary — 16 unique characters extracted from the labels, mapped to indices
Label encoding + padding — text labels converted to fixed-length integer sequences, padded with a dedicated blank token
Data augmentation — random pixel shifts + Gaussian noise applied to training images, doubling the effective training set (132 → 264 samples)
Model — CRNN: 2×(Conv2D + MaxPooling) for feature extraction, reshaped into a sequence, fed through 2 stacked Bidirectional LSTM layers, output via a Dense softmax layer over vocab size + 1 (blank)
Loss — CTC (ctc_batch_cost), which allows sequence prediction without exact character-to-pixel alignment
Training — Adam optimizer, EarlyStopping on validation loss, multiple random seeds tried to reduce run-to-run variance
Evaluation — exact-match accuracy on the validation set, decoded using standard CTC greedy decoding (argmax + blank/repeat collapsing)
Real Issues Hit and Fixed
CTC blank-token misalignment — initially indexed the blank token at position 0, which conflicted with Keras' CTC implementation expecting it last; caused a graph execution error during training
Padding bug — labels were briefly padded with their own first character instead of the blank token, corrupting encoded sequences
Overfitting — training loss dropped sharply while validation loss diverged; addressed with EarlyStopping and augmentation
Timestep-to-label-length ratio experiment — tried widening input images (128→256) to double model timesteps (32→64) to reduce character-dropping on longer labels; this did not meaningfully improve accuracy, indicating the bottleneck was dataset size, not model capacity
Stale EarlyStopping state across training runs — reusing one EarlyStopping instance across multiple seed-based training loops caused corrupted "best weights" restoration; fixed by instantiating it fresh inside each loop
Shape mismatches after Colab session resets — retraining on wide (256-width) images while testing against original (128-width) validation data silently produced 0% accuracy; required careful variable/shape tracking to resolve
Tech Stack
Python 3.x
TensorFlow / Keras
OpenCV, NumPy
Pandas, scikit-learn
