🚗 Used Car Price Predictor: LLM Fine-Tuning with QLoRA

An end-to-end Machine Learning project that fine-tunes Meta's Llama-3.2-3B to predict the price of used cars based on their textual descriptions.

This project covers the complete ML lifecycle: data cleaning, prompt formatting, QLoRA fine-tuning, model evaluation, and deployment to the Hugging Face Hub.

📌 Project Overview

Predicting the price of a used car is a complex regression task influenced by numerous factors such as brand, mileage, year, fuel type, engine, and transmission.

This project transforms raw car specifications into natural language prompts and fine-tunes a Large Language Model (LLM) to generate an estimated price.

✨ Key Features

Custom Data Pipeline: Uses Pydantic (items.py) to clean, validate, and structure raw car data.

Prompt Engineering: Converts structured car data into a Prompt/Completion format suitable for Supervised Fine-Tuning (SFT).

4-Bit Quantization: Utilizes QLoRA to train a 3B parameter model efficiently on limited GPU resources, tested on Google Colab T4.

Robust Evaluation: Implements a custom evaluation pipeline to calculate Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE), alongside visual scatter plots.

Hugging Face Integration: Supports dataset and model pushing to the Hugging Face Hub.

🛠️ Tech Stack

Language: Python

ML Frameworks: PyTorch, Hugging Face transformers, peft, trl, datasets, bitsandbytes

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib

Experiment Tracking: Weights & Biases (W&B)

Environment: Google Colab (NVIDIA T4 GPU)

📊 Dataset

The project uses a custom dataset (used-car-price-cleand-prompts) containing thousands of used car listings.

Each data point is structured as a Prompt/Completion pair.

Prompt Example

What does this cost to the nearest dollar? Brand: Toyota Model: Camry SE Year: 2021 milage: 25000.0 Fuel: Gasoline Engine: 2.5L I4 Transmission: A/T Exterior: White Interior: Black Accident: none reported Clean title: yes Price is $ 

Completion Example

29900.00 

🧠 Model Details

Base Model: meta-llama/Llama-3.2-3B

Fine-Tuning Technique: QLoRA (4-bit quantization)

LoRA Configuration

r: 32

lora_alpha: 64

lora_dropout: 0.1

Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Training Hyperparameters

Epochs: 1

Batch Size: 32

Learning Rate: 1e-4

Optimizer: paged_adamw_32bit

📈 Evaluation Results

The model was evaluated on a held-out test set of 1,594 used cars.

Mean Absolute Error (MAE): $6,270.23

Mean Absolute Percentage Error (MAPE): 17.12%

On this test set, the model's predictions had an average absolute error of approximately $6,270, corresponding to a MAPE of 17.12%.

The model showed lower errors on standard vehicles under $100k, while larger absolute errors were observed on some luxury and sports cars, which may be affected by price outliers in the dataset.

Actual vs Predicted Prices

￼

📂 Project Structure

├── items.py # Pydantic CarItem class, data processing, and Hub integration ├── qlora.ipynb # Main Jupyter Notebook for training and evaluation ├── requirements.txt # Python dependencies ├── results/ │ └── actual-vs-predicted.png └── README.md # Project documentation 

🚀 How to Run

1. Clone the Repository

git clone https://github.com/your-username/used-car-price-llm.git cd used-car-price-llm 

2. Install Dependencies

pip install -r requirements.txt 

3. Set Up Secrets

Create a .env file or set the following environment variables in your notebook/Colab:

HF_TOKEN=your_huggingface_write_token WANDB_API_KEY=your_wandb_api_key 

4. Data Preparation

Run items.py to clean the data and push the prompt-formatted dataset to the Hugging Face Hub.

5. Fine-Tuning

Open qlora.ipynb and run the cells sequentially. The notebook will:

Load the base model in 4-bit.

Apply LoRA adapters.

Train the model.

Evaluate MAE/MAPE.

Push the fine-tuned model to the Hugging Face Hub.

🔮 Future Improvements

The following improvements are planned to further investigate and improve model performance:

Outlier Removal: Filter


extreme prices, such as prices below $2,000 or above $300,000, to reduce potential noise.

Extended Training: Increase training from 1 epoch to 3 epochs.

Unsloth Integration: Experiment with Unsloth to reduce training time and VRAM usage.

Higher LoRA Rank: Increase r from 32 to 64 to investigate whether a higher rank improves performance.

🙏 Acknowledgments

Meta AI for the Llama-3.2-3B base model.

The Hugging Face team for the transformers, peft, and trl libraries.

Ed Donner for the foundational course and inspiration for this project structure.

Feel free to reach out if you have any questions or suggestions!
