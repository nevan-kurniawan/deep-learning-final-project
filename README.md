# MoE-LoRA Texture Classification

This project implements a Mixture of Experts (MoE) with Low-Rank Adaptation (LoRA) model for texture classification. It includes a Streamlit-based web application for easy interaction with the trained model.

## Running the App Locally

Follow these steps to set up and run the texture classification application on your local machine.

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning)

### Setup Instructions

1.  **Clone the Repository** (if you haven't already):

    ```bash
    git clone https://github.com/nevan-kurniawan/deep-learning-final-project.git
    cd deep-learning-final-project
    ```

2.  **Create and Activate a Virtual Environment** (Recommended):

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Install the required Python packages for the application:

    ```bash
    pip install -r app/requirements.txt
    ```

    _Note: This requirements file is specific to the application. If you want to train the model or run notebooks, you might need the root `requirements.txt` as well._


### Running the Application

1.  **Start Streamlit**:
    Run the following command from the project root directory:

    ```bash
    streamlit run app/app.py
    ```

2.  **Access the App**:
    The application should open automatically in your default web browser. If not, open the URL displayed in the terminal (usually `http://localhost:8501`).

3.  **Use the Classifier**:
    - Upload an image (JPG or PNG) using the file uploader.
    - Click the **Classify** button.
    - The model will predict the texture class and display the confidence score.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**:
  Make sure you run the streamlit command from the project root (`deep-learning-final-project`), not from inside the `app` folder. The app adds the parent directory to `sys.path`, but running from the root is the standard way.
