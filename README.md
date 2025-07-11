# Templock: Improved Time-LLM Model for Electricity Price Forecasting

This repository contains the implementation of **Templock**, an improved model based on Time-LLMs (Large Language Models) for **electricity price forecasting** and **economic benefit analysis** in power markets. The model integrates a **Time Series Encoding Module (TEM)** and a **Multi-Patch Method (Pblock)** to handle the nonlinearity and non-stationarity of electricity price data.

## Overview

Electricity price forecasting is crucial for the efficient operation of power grids and market participants. The Templock model enhances the prediction accuracy by improving the extraction of temporal features from historical time series data. It leverages the semantic understanding capabilities of Large Language Models (LLMs) and integrates advanced time series algorithms to improve performance.

### Key Contributions:

* **Time Series Encoding Module (TEM)**: Enhances time feature representation using advanced algorithms like TimesNet and non-stationary transformers.
* **Multi-Patch Method (Pblock)**: Divides time series data into multiple patches to capture multi-granularity temporal features.
* **Domain-Specific Prompts**: Utilizes LLM's understanding capabilities with tailored prompt strategies for more accurate forecasting.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Templock.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Templock
   ```

3. Install required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure to have an NVIDIA GPU (e.g., A100) for faster computation during training, especially for large datasets.

## Data Availability

The following publicly available datasets are used for training and evaluation:

1. **PJM Electricity Market Load Dataset (October 1, 2024 - October 31, 2024)**
   This dataset includes hourly electricity load data from the PJM market in the United States.
   Available at: [PJM Dataset](https://dataminer2.pjm.com/list)

2. **Load Demand Dataset from a Factory in the High-tech Zone, Hunan Province, China (September 6, 2017 - July 6, 2018)**
   Provides real load demand data from a factory in China.
   Available at: [Factory Dataset](https://pan.baidu.com/s/1CxsQpifjTcPTCd74bMbyqw)

3. **Australian Electricity Price and Load Dataset (January 1, 2006 - January 1, 2011)**
   Includes hourly electricity price and load data from the Australian National Electricity Market (NEM).
   Available at: [Australian Dataset](https://pan.baidu.com/s/1LjLHqhu2YJjgjLEOcqI1XQ)

4. **European Transmission System Load Demand Dataset (January 1, 2006 - December 31, 2015)**
   This dataset contains hourly electricity load demand data from multiple European countries.
   Available at: [European Dataset](https://pan.baidu.com/s/14gnwMlWYimms3-DJKiyhUQ)

## Model Overview

The model is based on **Large Language Models (LLMs)**, with modifications to enhance its performance in time series forecasting tasks like electricity price prediction. The key components of the model are:

1. **Time Feature Encoding**: Temporal features (e.g., time of day, month, weekday) are encoded using sine and cosine functions to capture periodicity.
2. **Multi-Patch Method (Pblock)**: Divides time series into patches to capture different granularities of time series data.
3. **LLM-Based Prediction Module**: Utilizes pre-trained language models (e.g., GPT-2) to process and predict time series data.
4. **Dynamic Optimization and Projection**: Implements optimization techniques to refine predictions based on real-time data.

### Prediction Workflow:

1. The model is trained on historical electricity data and learns temporal patterns through encoding and time series decomposition.
2. The trained model can then predict future electricity prices, and the predictions can be used to optimize economic benefits for market participants.

## Running the Code

### Training the Model:

To train the model, execute the following command:

```bash
python train.py --dataset <dataset_name> --epochs 100 --batch_size 64
```

Replace `<dataset_name>` with the dataset you wish to train on (e.g., "PJM", "Australia", etc.).

### Testing the Model:

To test the model on a pre-trained model, use the following command:

```bash
python test.py --model_path <path_to_pretrained_model> --test_dataset <dataset_name>
```

## Evaluation Metrics

The performance of the model is evaluated using the following metrics:

* **Mean Squared Error (MSE)**
* **Mean Absolute Error (MAE)**
* **Mean Absolute Percentage Error (MAPE)**
* **Root Mean Squared Error (RMSE)**

These metrics are calculated to evaluate the model's prediction accuracy for different datasets and forecasting horizons.


