"""
This module performs sentiment analysis using BERT with custom datasets.
It includes model training, evaluation, and hyperparameter tuning.
"""

# %% Imports
import torch  # pylint: disable=E0401
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments  # pylint: disable=E0401
from torch.utils.data import Dataset  # pylint: disable=E0401
import numpy as np  # pylint: disable=E0401
import pandas as pd  # pylint: disable=E0401
from sklearn.model_selection import train_test_split  # pylint: disable=E0401
from sklearn.utils.class_weight import compute_class_weight  # pylint: disable=E0401
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)  # pylint: disable=E0401
import matplotlib.pyplot as plt  # pylint: disable=E0401

# Prepare the tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a custom dataset class
class SentimentDataset(Dataset):
    """
    Custom dataset class for sentiment analysis.
    Encodes texts and assigns labels for sentiment classification.
    """
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# Load the dataset
deberta_df = pd.read_csv('filtered_data.csv')
texts_list = deberta_df['reviews.text'].astype(str).fillna('').tolist()

deberta_df['sentiment'] = deberta_df['reviews.rating'].map(
    lambda x: 'negative' if x in [1, 2] else ('neutral' if x == 3 else 'positive')
)
sentiment_labels = deberta_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).tolist()

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts_list, sentiment_labels, test_size=0.1, stratify=sentiment_labels
)
train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

# Ensure GPU usage
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(sentiment_labels), y=sentiment_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Load BERT model with classification head
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(DEVICE)

# Define a custom trainer with weighted loss function
class WeightedLossTrainer(Trainer):
    """
    Custom Trainer class to apply weighted loss during training.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        labels = labels.to(DEVICE)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

best_params = {'batch_size': 16, 'learning_rate': 3e-5}

# Reinitialize training arguments using the best parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=best_params['batch_size'],
    per_device_eval_batch_size=best_params['batch_size'],
    warmup_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",
    learning_rate=best_params['learning_rate']
)

# Train the model using the best parameters found
trainer = WeightedLossTrainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
preds_output_val = trainer.predict(val_dataset)
val_predictions = np.argmax(preds_output_val.predictions, axis=1)

# Evaluate model based on precision of predicting negative sentiment
classification_report_val = classification_report(val_labels, val_predictions, target_names=['negative', 'neutral', 'positive'])
negative_precision = precision_score(val_labels, val_predictions, average=None)[0]
print("Classification Report:\n", classification_report_val)
print("Negative Precision:", negative_precision)

# Save the model in the same directory as your Jupyter notebook
bert_model.save_pretrained('./bert_pretrained')
tokenizer.save_pretrained('./bert_pretrained')

def evaluate_model(trainer_model, eval_dataset, eval_labels, eval_texts_list):
    """
    Evaluates the model performance based on accuracy, precision, recall, and F1 score.
    Displays the classification report and confusion matrix.
    """
    preds_output_eval = trainer_model.predict(eval_dataset)
    preds_eval = np.argmax(preds_output_eval.predictions, axis=1)

    accuracy = accuracy_score(eval_labels, preds_eval)
    precision = precision_score(eval_labels, preds_eval, average='weighted')
    recall = recall_score(eval_labels, preds_eval, average='weighted')
    f1 = f1_score(eval_labels, preds_eval, average='weighted')

    print("Accuracy:", accuracy)
    print("Weighted Precision:", precision)
    print("Weighted Recall:", recall)
    print("Weighted F1 Score:", f1)

    target_names = ['negative', 'neutral', 'positive']
    report_eval = classification_report(eval_labels, preds_eval, target_names=target_names)
    print("\nClassification Report:\n", report_eval)

    cm = confusion_matrix(eval_labels, preds_eval)
    plot_confusion_matrix(cm, target_names)

    print_sample_predictions(eval_texts_list, eval_labels, preds_eval)

def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix for a given set of predictions and actual labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # pylint: disable=no-member
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def print_sample_predictions(eval_texts_list, eval_labels, eval_preds, num_samples=20):
    """
    Prints sample predictions for each class, showing the text, actual, and predicted labels.
    """
    data = pd.DataFrame({'Text': eval_texts_list, 'Actual': eval_labels, 'Predicted': eval_preds})
    target_names = ['negative', 'neutral', 'positive']
    for i, name in enumerate(target_names):
        print(f"\nSample {num_samples} Predictions for Rating '{name}':")
        samples = data[data['Actual'] == i].sample(n=min(num_samples, len(data[data['Actual'] == i])))
        for _, row in samples.iterrows():
            print(f"Text: {row['Text'][:100]}... | Actual: {target_names[row['Actual']]} | Predicted: {target_names[row['Predicted']]}")

# Evaluate the model
evaluate_model(trainer, val_dataset, val_labels, val_texts)

def predict_sentiment(input_texts, batch_size=16):
    """
    Uses the trained BERT model to predict sentiment for a given list of texts.
    """
    predictions = []
    bert_model.eval()
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, axis=1).cpu().numpy()
        predictions.extend(batch_preds)
    return predictions

# Get predictions for the entire dataset
final_predictions = predict_sentiment(texts_list)

# Map numeric predictions to sentiment labels
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
deberta_df['predicted_sentiment'] = [label_map[pred] for pred in final_predictions]

# Save the updated DataFrame with predictions to a new CSV file
deberta_df.to_csv('filtered_data_with_predictions.csv', index=False)

print("Predictions have been added to 'filtered_data_with_predictions.csv'.")
