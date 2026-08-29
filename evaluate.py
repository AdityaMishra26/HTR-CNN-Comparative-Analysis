# ============================================================
# COMPLETE HTR EVALUATION
# Exact Match + CER + WER + Precision + Recall + F1
# ============================================================

import torch
from tqdm.auto import tqdm
from collections import Counter

model.eval()

all_true_texts = []
all_pred_texts = []

with torch.no_grad():

    for batch in tqdm(val_loader, desc="Evaluating validation set"):

        images = batch["images"].to(device)

        # Model prediction
        outputs = model(images)

        # Get predicted class indices
        predicted_indices = outputs.argmax(dim=2)

        # Decode every sample in the batch
        for i in range(images.size(0)):

            # ------------------------------------------------
            # GROUND TRUTH
            # ------------------------------------------------
            true_length = batch["label_lengths"][i].item()

            true_indices = (
                batch["labels"][i][:true_length]
                .cpu()
                .tolist()
            )

            # Decode ground truth
            true_text = "".join(
                characters[idx - 1]
                for idx in true_indices
                if idx > 0
            )

            # ------------------------------------------------
            # PREDICTION WITH CTC DECODING
            # ------------------------------------------------
            pred_indices = (
                predicted_indices[i]
                .cpu()
                .tolist()
            )

            decoded = []
            previous = -1

            for idx in pred_indices:

                # CTC blank = 0
                if idx != 0 and idx != previous:
                    decoded.append(characters[idx - 1])

                previous = idx

            pred_text = "".join(decoded)

            # Save results
            all_true_texts.append(true_text)
            all_pred_texts.append(pred_text)


# ============================================================
# 1. EXACT MATCH ACCURACY
# ============================================================

correct_predictions = sum(
    true == pred
    for true, pred in zip(all_true_texts, all_pred_texts)
)

total_samples = len(all_true_texts)

exact_match_accuracy = (
    correct_predictions / total_samples * 100
)


# ============================================================
# 2. CER / WER
# ============================================================

def levenshtein_distance(reference, hypothesis):

    rows = len(reference) + 1
    cols = len(hypothesis) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):

            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    return dp[-1][-1]


total_char_errors = 0
total_chars = 0

total_word_errors = 0
total_words = 0

for true_text, pred_text in zip(all_true_texts, all_pred_texts):

    # CER
    total_char_errors += levenshtein_distance(
        true_text,
        pred_text
    )

    total_chars += len(true_text)

    # WER
    true_words = true_text.split()
    pred_words = pred_text.split()

    total_word_errors += levenshtein_distance(
        true_words,
        pred_words
    )

    total_words += len(true_words)


cer = total_char_errors / total_chars
wer = total_word_errors / total_words


# ============================================================
# 3. CHARACTER-LEVEL PRECISION / RECALL / F1
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0

for true_text, pred_text in zip(all_true_texts, all_pred_texts):

    true_counter = Counter(true_text)
    pred_counter = Counter(pred_text)

    # Matching characters
    tp = sum(
        min(true_counter[char], pred_counter[char])
        for char in true_counter
    )

    # Extra predicted characters
    fp = len(pred_text) - tp

    # Missing ground-truth characters
    fn = len(true_text) - tp

    total_tp += tp
    total_fp += fp
    total_fn += fn


precision = (
    total_tp / (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0
)

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# 4. FINAL RESULTS
# ============================================================

print("=" * 65)
print("       COMPLETE HTR MODEL EVALUATION RESULTS")
print("=" * 65)

print(f"Total Validation Samples     : {total_samples}")
print(f"Exact Correct Predictions    : {correct_predictions}")

print("-" * 65)

print(f"Exact Match Accuracy         : {exact_match_accuracy:.2f}%")
print(f"Character Error Rate (CER)   : {cer:.4f} ({cer*100:.2f}%)")
print(f"Word Error Rate (WER)        : {wer:.4f} ({wer*100:.2f}%)")

print("-" * 65)

print(f"Character Accuracy (1-CER)   : {(1-cer)*100:.2f}%")
print(f"Word Accuracy (1-WER)        : {(1-wer)*100:.2f}%")

print("-" * 65)

print("CHARACTER-LEVEL CLASSIFICATION METRICS")
print(f"Precision                    : {precision*100:.2f}%")
print(f"Recall                       : {recall*100:.2f}%")
print(f"F1-Score                     : {f1_score*100:.2f}%")

print("=" * 65)