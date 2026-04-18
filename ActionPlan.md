# Action Plan
Disclaimer: I generated this plan with an AI 

## 1. Dataset Selection and Audit (Winnie)
Objective: validate that the dataset is suitable for our protocol.

Steps:
- Choose a dataset (e.g., EmoV-DB dataset or equivalent)
- Verify the presence of:
    - audio (wav)
    - aligned transcriptions
    - emotional labels
- Analyze:
    - emotion distribution (balanced or not)
    - average audio duration
    - signal quality (approximate SNR)
- Decide:
    - use dataset as-is
    - or require preprocessing (cleaning, filtering)

## 2. Data Preprocessing (Winnie)
Objective: obtain a clean and standardized dataset.

Steps:
- Resampling (e.g., 22 kHz or 16 kHz)
- Amplitude normalization
- Cleaning:
    - removal of long silences
    - trimming
- Text-audio alignment (if needed)
- Text encoding:
    - phonemes (ideal) or characters
- Emotion encoding:
    - mapping → integers or embeddings

## 3. Construction of Experimental Conditions (Winnie)
Objective: formalize our factorial design.

### Case 1 — already noisy dataset
split:
- clean subset (if possible)
- noisy subset

### Case 2 — clean dataset
create noisy version:
- add white noise (controlled SNR)
- optionally real-world noise

### Result:
- dataset_clean
- dataset_noisy

## 4. Data Splitting (Winnie)
Objective: avoid data leakage.

- Train / Validation / Test (e.g., 80/10/10)
- Stratification:
    - balanced emotions in each split

## 5. Model Implementation (Coco)
Objective: isolate the emotional variable.

### Model A — baseline
`input = text`

### Model B — emotion-aware
`input = text + emotion embedding`

### Constraints:
- same architecture
- same hyperparameters
- same vocoder

Otherwise comparison is invalid

## 6. Training (Coco)
Objective: produce comparable models.

Steps:
- fix random seeds (reproducibility)
- logging:
    - loss
    - validation metrics
- train on:
    - dataset_clean
    - dataset_noisy

We will potentially obtain 4 models:
- baseline_clean
- baseline_noisy
- emotion_clean
- emotion_noisy

## 7. Sample Generation (Coco)
Objective: create the evaluation dataset.

- select a fixed set of sentences
- generate for each condition:
    - same texts
    - same emotions (for Model B)

Output:
`generated audio per condition`

## 8. Objective Evaluation (Jim + Tim + Lub)
Objective: automatically measure performance.

### Audio quality:
- PESQ
- SNR

### Intelligibility:
- WER (via ASR)

### Emotional fidelity:
- SER model → accuracy / F1 (emotion2vec)
- confusion matrix analysis

## 9. Subjective Evaluation (MOS)
Objective: human validation.

- panel of evaluators
- protocol:
    - audio randomization
    - model anonymization
- scoring:
    - 1 to 5 (quality / naturalness / expressiveness)

## 10. Statistical Analysis (Tim)
Objective: validate our hypotheses.

- compute means / variances
- ANOVA:
    - model factor
    - noise factor
    - interaction
- visualizations:
    - boxplots
    - barplots

## 11. Qualitative Analysis (All)
Objective: fine-grained interpretation.

- listen for:
    - typical errors
    - artifacts
    - poorly rendered emotions
- analyze:
    - emotion confusion
    - impact of noise

## 12. Results Writing (All)
Objective: convert into academic content.

- present:
    - numerical results
    - graphs
    - statistical tests
- interpret:
    - hypothesis validated or not
    - quality vs expressiveness trade-off

## 13. Conclusion and Limitations (All)

Limitations:
- dataset
- artificial noise
- metrics

Future work:
- diffusion models
- multi-speaker setups
- improved embeddings