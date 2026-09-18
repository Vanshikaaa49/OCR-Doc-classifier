# OCR-Doc-classifier
A Python-based OCR tool that extracts text from images and scanned documents using [Tesseract/EasyOCR/etc.], with preprocessing for noise reduction and layout detection.

INTRO

A custom OCR pipeline built from scratch that reads document category labels (email, resume, scientific_publication) directly from document images, using a CRNN (CNN + Bidirectional LSTM) architecture trained with CTC loss.

This is not a wrapper around Tesseract/EasyOCR — the model, training loop, and decoding logic were built and debugged from the ground up.

RESULT

81% exact-match accuracy on a held-out validation set, after several rounds of debugging and tuning.

DATASET

1. 165 labeled document images, loaded from a zipped CSV (image_path, text_label columns)
2. 3 classes: email, resume, scientific_publication
3. Split 80/20 train/validation
   
PIPELINE

1. Image preprocessing — grayscale conversion, resize to 128×32, pixel normalization
2. Character-level vocabulary — 16 unique characters extracted from the labels, mapped to indices
3. Label encoding + padding — text labels converted to fixed-length integer sequences, padded with a dedicated blank token
4. Data augmentation — random pixel shifts + Gaussian noise applied to training images, doubling the effective training set (132 → 264 samples)
5. Model — CRNN: 2×(Conv2D + MaxPooling) for feature extraction, reshaped into a sequence, fed through 2 stacked Bidirectional LSTM layers, output via a Dense softmax layer over vocab size + 1 (blank)
6. Loss — CTC (ctc_batch_cost), which allows sequence prediction without exact character-to-pixel alignment
7. Training — Adam optimizer, EarlyStopping on validation loss, multiple random seeds tried to reduce run-to-run variance
8. Evaluation — exact-match accuracy on the validation set, decoded using standard CTC greedy decoding (argmax + blank/repeat collapsing)

REAL ISSUES HIT AND FIXED 

1. CTC blank-token misalignment — initially indexed the blank token at position 0, which conflicted with Keras' CTC implementation expecting it last; caused a graph execution error during training
2. Padding bug — labels were briefly padded with their own first character instead of the blank token, corrupting encoded sequences
3. Overfitting — training loss dropped sharply while validation loss diverged; addressed with EarlyStopping and augmentation
4. Timestep-to-label-length ratio experiment — tried widening input images (128→256) to double model timesteps (32→64) to reduce character-dropping on longer labels; this did not meaningfully improve accuracy, indicating the bottleneck was dataset size, not model capacity
5. Stale EarlyStopping state across training runs — reusing one EarlyStopping instance across multiple seed-based training loops caused corrupted "best weights" restoration; fixed by instantiating it fresh inside each loop
6. Shape mismatches after Colab session resets — retraining on wide (256-width) images while testing against original (128-width) validation data silently produced 0% accuracy; required careful variable/shape tracking to resolve

TECH STACK 

Python 3.x
TensorFlow / Keras
OpenCV, NumPy
Pandas, scikit-learn

LIMITATIONS 

 1. Very small dataset (165 images) caps achievable accuracy and makes results somewhat sensitive to random seed
 2. Only 3 document classes; not tested on a broader label set
 3. No held-out test set separate from validation — accuracy is reported on the validation split used during training

CONTRIBUTING

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.
