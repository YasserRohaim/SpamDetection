# Spam Classification using LSTM

This project is a spam classification system built using the Enron email dataset. The system uses a Long Short-Term Memory (LSTM) neural network, implemented with TensorFlow, to classify emails as spam or ham. Additionally, a FastAPI service is provided to run model inference on new email data.

## Project Overview

- **Dataset:** The project uses the Enron email dataset, which contains a large collection of emails categorized as spam or ham.
- **Model:** The model is a Sequential LSTM network that processes the preprocessed email text data to classify it into spam or ham categories.
- **API:** A FastAPI service is provided to allow users to input an email and receive a prediction.

## Project Structure

- **`train_model.ipynb`:** Jupyter notebook for preprocessing the dataset, training the LSTM model, and saving the tokenizer, model, and label encoder.
- **`inference.py`:** Python script implementing a FastAPI service for running inference on new emails using the trained LSTM model.
- **`model.keras`:** The trained LSTM model saved in Keras format.
- **`tokenizer.pickle`:** Tokenizer fitted on the training data, saved for use during inference.
- **`label_encoder.pickle`:** Label encoder used to map the target labels (spam/ham) to integers and vice versa.
- **`email.csv`:** A CSV file containing the email data used for testing the API.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/spam-classification-lstm.git
    cd spam-classification-lstm
    ```

2. **Install the dependencies:**
    ```bash
    pip install -r requirements
    ```

3. **Run the FastAPI service:**
    ```bash
    uvicorn inference:app --reload
    ```

4. **Test the API:**
    - You can test the API by navigating to `http://127.0.0.1:8000/` in your browser and adding the email text in the URL as a query parameter, like so:
    ```
    http://127.0.0.1:8000/?new_email=Your email text here
    ```

## Usage

### Training the Model
- Use the `train_model.ipynb` notebook to preprocess the dataset and train the LSTM model. The notebook walks through loading the dataset, tokenizing and padding the email text, and training the LSTM model.

### Running Inference
- The `inference.py` script provides a FastAPI endpoint to run inference on new email data. The model predicts whether the email is spam or ham.

### Example
To predict if the email "Congratulations! You've won a $1000 gift card." is spam, navigate to:
http://127.0.0.1:8000/?new_email=Congratulations! You've won a $1000 gift card.

The API will return a JSON response indicating whether the email is classified as "spam" or "ham".

## Contributing

Feel free to fork the repository and submit pull requests. Contributions are welcome to improve the model accuracy, add new features, or optimize the API.

## License

This project is licensed under the MIT License.
