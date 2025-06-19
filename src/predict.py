# src/predict.py

import os  # To help build file paths correctly
import torch  # PyTorch library, for tensors and model operations
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)  # Hugging Face stuff for models


# --- Configuration ---
_SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Gets the directory where this script is
MODEL_PATH = os.path.join(_SCRIPT_DIR, "fine_tuned_model")

print(f"DEBUG (predict.py): Model path set to: {MODEL_PATH}")  # For checking the path

# --- Device Setup ---
# Check if a GPU is available, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Trying to get the name of the GPU, just for information
    try:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"INFO (predict.py): GPU is available ({gpu_name}), using CUDA.")
    except Exception as e:
        print(
            f"INFO (predict.py): GPU is available, using CUDA. (Could not get GPU name: {e})"
        )
else:
    device = torch.device("cpu")
    print(
        "INFO (predict.py): GPU not available, using CPU. Predictions might be slower."
    )

# --- Load Model and Tokenizer ---
# Load these once when the script (or module) is first loaded rather than loading them every time to predict.
model = None
tokenizer = None
id2label_mapping = {0: "negative", 1: "neutral", 2: "positive"}  # Default mapping

try:
    print(f"INFO (predict.py): Loading model from {MODEL_PATH}...")
    # Load the pre-trained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)  # Move the model to the GPU (or CPU if no GPU)
    model.eval()  # Set the model to evaluation mode (important for layers like Dropout)
    print("INFO (predict.py): Model loaded successfully and set to evaluation mode.")

    print(f"INFO (predict.py): Loading tokenizer from {MODEL_PATH}...")
    # Load the tokenizer that matches the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("INFO (predict.py): Tokenizer loaded successfully.")

    # Get the label mapping from the model's configuration
    # This was saved during fine-tuning
    if hasattr(model.config, "id2label") and model.config.id2label:
        id2label_mapping = model.config.id2label
        # Convert string keys from config.json to int if necessary
        id2label_mapping = {int(k): v for k, v in id2label_mapping.items()}
        print(
            f"INFO (predict.py): Loaded id2label mapping from model config: {id2label_mapping}"
        )
    else:
        print(
            "WARN (predict.py): id2label not found in model config, using default mapping."
        )

except FileNotFoundError:
    print(f"--- CRITICAL ERROR (predict.py) ---")
    print(f"Model or Tokenizer files NOT FOUND at the specified path: {MODEL_PATH}")
    print(
        f"Please ensure the '{os.path.basename(MODEL_PATH)}' directory exists at '{_SCRIPT_DIR}' and contains all necessary model files (pytorch_model.bin/model.safetensors, config.json, tokenizer files, etc.)."
    )
    # Keep model and tokenizer as None, so predict_sentiments can handle it
except Exception as e:
    print(f"--- ERROR (predict.py) ---")
    print(f"An unexpected error occurred loading model or tokenizer: {e}")
    # Keep model and tokenizer as None


# --- Preprocessing Function ---
def preprocess_tweet(text):
    """Replaces @user mentions and http links with placeholders."""
    preprocessed_text = []
    if text is None:
        return ""  # Handle None input
    # Split text into parts by space
    for t in text.split(" "):
        if len(t) > 0:  # Avoid processing empty parts from multiple spaces
            t = "@user" if t.startswith("@") else t  # Replace mentions
            t = "http" if t.startswith("http") else t  # Replace links
        preprocessed_text.append(t)
    return " ".join(preprocessed_text)  # Put the parts back together


# --- Prediction Function (UPDATED to return probabilities) ---
def predict_sentiments(comment_list: list):
    """
    Predicts sentiments for a list of comment strings.
    Returns a list of dictionaries, each containing the predicted label
    and the probabilities (scores) for each class.
    e.g., [{'label': 'positive', 'scores': {'negative': 0.1, 'neutral': 0.2, 'positive': 0.7}}, ...]
    """
    # Check if model and tokenizer are ready
    if model is None or tokenizer is None:
        print(
            "ERROR (predict.py - predict_sentiments): Model or Tokenizer not loaded. Cannot predict."
        )
        # Return an error structure
        return [{"label": "Error: Model not loaded", "scores": {}}] * len(comment_list)

    if not comment_list:  # Handle empty input list
        return []

    inference_batch_size = 64  # You can adjust this number based on performance/memory
    print(
        f"INFO (predict.py): Predicting sentiments for {len(comment_list)} comments in batches of {inference_batch_size}..."
    )

    all_results_list = []  # Collect results for all batches

    # --- Loop through the comment list in batches ---
    try:
        total_comments = len(comment_list)
        # This loop goes from 0 to total_comments, jumping by inference_batch_size each time
        for i in range(0, total_comments, inference_batch_size):
            # Get the current slice of comments for this batch
            batch_comments = comment_list[i : i + inference_batch_size]

            # Just printing progress for long lists
            current_batch_num = i // inference_batch_size + 1
            total_batches = (
                total_comments + inference_batch_size - 1
            ) // inference_batch_size
            print(
                f"DEBUG (predict.py): Processing batch {current_batch_num}/{total_batches}..."
            )

            # --- Process ONLY the current batch ---
            # 1. Preprocess this specific batch
            processed_batch = [preprocess_tweet(comment) for comment in batch_comments]

            # 2. Tokenize this batch
            # Tokenizer handles padding within this smaller batch
            inputs = tokenizer(
                processed_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=(
                    tokenizer.model_max_length
                    if hasattr(tokenizer, "model_max_length")
                    and tokenizer.model_max_length
                    else 512
                ),
            )

            # 3. Move this batch's inputs to the device (GPU/CPU)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 4. Make prediction for this batch - no need for gradients
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # Raw scores from the model for this batch

            # 5. Calculate probabilities and get predicted class IDs for this batch
            probabilities_batch = torch.softmax(logits, dim=-1)
            predicted_class_ids_batch = torch.argmax(probabilities_batch, dim=-1)

            # 6. Move results back to CPU, convert to lists for easier looping
            probs_list_batch = probabilities_batch.cpu().numpy().tolist()
            ids_list_batch = predicted_class_ids_batch.cpu().numpy().tolist()

            # 7. Format results for each comment in THIS batch
            batch_results = []
            for j in range(len(ids_list_batch)):
                pred_id = ids_list_batch[j]
                pred_label = id2label_mapping.get(
                    pred_id, "Unknown"
                )  # Map ID to label name
                # Create the scores dictionary for this comment
                pred_scores = {
                    label_name: probs_list_batch[j][label_id]
                    for label_id, label_name in id2label_mapping.items()
                    if 0
                    <= label_id
                    < probabilities_batch.shape[-1]  # Safety check for index
                }
                # Add the result for this comment
                batch_results.append({"label": pred_label, "scores": pred_scores})

            # Add the results from this completed batch to our main list
            all_results_list.extend(batch_results)
            # --- Finished processing current batch ---

        print(
            f"INFO (predict.py): Finished processing all {len(all_results_list)} comments."
        )

    except Exception as e:
        # Catch errors that might happen during the loop
        print(f"--- ERROR (predict.py - predict_sentiments loop) ---")
        print(
            f"An error occurred during batch prediction (around comment index {i}): {e}"
        )
        import traceback

        traceback.print_exc()  # Print full error details to console
        # Try to return results for processed batches + error messages for the rest
        num_processed = len(all_results_list)
        num_remaining = len(comment_list) - num_processed
        # Add error indicators for comments that couldn't be processed
        all_results_list.extend(
            [{"label": "Error: Batch failed", "scores": {}}] * num_remaining
        )

    # Return the list containing results for all comments
    return all_results_list


# --- Main block for testing this script directly (UPDATED to show scores) ---
if __name__ == "__main__":
    print("\n--- Testing predict.py Script Directly ---")
    if model and tokenizer:
        sample_comments_for_testing = [
            "This is an amazing movie, I loved it!",
            "I'm not sure how I feel about this, it was okay.",
            "Worst experience ever, would not recommend.",
            "The food was alright, but the service was slow.",
            "What a fantastic day! #blessed",
            "I hate waiting in long lines.",
            "@user Check out http this is cool.",
            "Just a normal sentence, nothing special here.",
            "",
            "This new update is absolutely terrible and full of bugs.",
        ]

        print("\nInput Comments for Direct Test:")
        for i, c in enumerate(sample_comments_for_testing):
            print(f"{i+1}. '{c}'")

        # Get predictions (now a list of dictionaries)
        prediction_results = predict_sentiments(sample_comments_for_testing)

        print("\nPredicted Sentiments and Scores (Direct Test):")
        # Loop through the results list
        for i, (comment, result) in enumerate(
            zip(sample_comments_for_testing, prediction_results)
        ):
            print(f"{i+1}. Comment: '{comment}'")
            # Format scores nicely for printing
            scores_dict = result.get("scores", {})
            formatted_scores = ", ".join(
                [f"{name}: {score:.3f}" for name, score in scores_dict.items()]
            )
            print(f"   -> Predicted Label: {result.get('label', 'N/A')}")
            # Also print the raw scores dictionary
            print(f"   -> Scores: {{{formatted_scores}}}")
        print("--- Direct Test Finished ---")
    else:
        print("ERROR (predict.py - main test): Model and/or tokenizer not loaded.")
        print(
            f"Please check the MODEL_PATH ('{MODEL_PATH}') and ensure model files are present."
        )
