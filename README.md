# Spam Detector

This project is a simple spam message classifier built with classical machine learning.
It trains several text classification models on an English spam dataset, compares their metrics, and saves the best model as a `.joblib` file.

The project also includes a multilingual command line interface.
It can detect the language of the input message, translate non English text into English, and then run spam prediction with the trained model.

## What the project does

- Loads an email spam dataset from Hugging Face
- Cleans the data and splits it into train and test sets
- Trains three models: Naive Bayes, Logistic Regression, and Linear SVM
- Compares models with accuracy, precision, recall, and F1 score
- Saves the best model into the `models/` directory
- Detects the user language in the CLI app
- Translates supported non English messages into English before prediction

## Project structure

```text
spam-detector/
├── src/
│   ├── train.py
│   └── predict.py
├── notebooks/
├── models/
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to train the model

Run the training script from the project root:

```bash
python3 src/train.py
```

What happens during training:

- The script downloads the dataset
- It trains three different models
- It prints evaluation metrics on the test split
- It saves the best model into `models/`

After training, you should see a message with the best model name and the path where it was saved.

## How to test the model

There are two practical ways to test it.

### 1. Check evaluation metrics after training

The easiest way to test model quality is to run:

```bash
python3 src/train.py
```

This prints the metrics for each model on the held out test set.
Use these results to compare model performance.

### 2. Run interactive prediction

After training, start the CLI app:

```bash
python3 src/predict.py
```

Then enter messages manually and check whether the prediction looks reasonable.

Useful commands inside the CLI:

- `/lang` to reset language detection
- `/exit` to quit the app

## Multilingual language detection and translation

The classifier itself is trained on English text only.
To make the CLI usable for other languages, `src/predict.py` adds two extra steps before prediction:

1. It detects the input language with `lingua-language-detector`
2. If the message is not in English, it translates the text into English with `deep-translator`

Then the translated text is passed to the spam classifier.

This means:

- English messages are classified directly
- Non English messages are translated first
- Accuracy may be lower for non English text because the original model was trained only on English data

Supported languages in the current CLI include:

- English
- Russian
- Spanish
- French
- German
- Portuguese
- Chinese
- Hindi
- Arabic
- Bengali
- Urdu
- Indonesian
- Japanese
- Turkish
- Korean

## Notes

- Run commands from the project root directory
- The `models/` directory is expected to contain a saved `.joblib` model before `predict.py` can work
- If no saved model is found, `predict.py` will raise `FileNotFoundError`

## Example workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/train.py
python3 src/predict.py
```
