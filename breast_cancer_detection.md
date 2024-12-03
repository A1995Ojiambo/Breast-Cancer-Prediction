# Breast Cancer Detection

This repository contains a Machine Learning project focused on early detection of breast cancer. The goal is to classify tumors as benign or malignant using predictive modeling, thereby improving diagnostic accuracy and patient outcomes.

## Project Overview

- **Objective**: Develop a machine learning model to classify tumors as benign or malignant based on medical attributes such as tumor size, texture, and cell features.
- **Dataset**: [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Methods Used**: Logistic Regression, Random Forest, feature scaling, and hyperparameter optimization.
- **Results**: Achieved a best model accuracy of **99.1%**, showcasing the potential of machine learning in healthcare.

## Project Structure

```
├── data
│   └── breast_cancer_kaggle.csv      # Raw dataset
├── notebooks
│   └── data_preprocessing.ipynb     # Data preprocessing steps
│   └── model_building.ipynb         # Model training and evaluation
├── src
│   └── utils.py                     # Helper functions for data handling
│   └── models.py                    # Model definitions
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── LICENSE                          # License for the project
```

## Key Steps

1. **Data Preprocessing**
   - Imported and explored the dataset.
   - Handled missing values and performed feature engineering.
   - Scaled the features using `StandardScaler`.

2. **Model Building**
   - Trained Logistic Regression and Random Forest models.
   - Evaluated model performance using accuracy, precision, recall, and F1 score.
   - Cross-validated the models to ensure generalizability.

3. **Hyperparameter Optimization**
   - Used Randomized Search to fine-tune Logistic Regression parameters.
   - Achieved the best configuration with `solver='saga'`, `penalty='l2'`, and `C=1.5`.

4. **Final Model**
   - Implemented the optimized Logistic Regression model.
   - Achieved cross-validation accuracy of **97.16%** with a standard deviation of **2.18%**.

5. **Prediction**
   - Tested the model on a single observation to classify tumor malignancy.

## Dependencies

To run the project, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-detection.git
   cd breast-cancer-detection
   ```
2. Preprocess the data and train the models by running the notebooks in the `notebooks` directory.
3. Use `src/models.py` to predict tumor classification on new data.

## Results

| Model                | Accuracy | Precision | Recall  | F1 Score |
|----------------------|----------|-----------|---------|----------|
| Logistic Regression  | 99.1%    | 100%      | 97.6%   | 98.8%    |
| Random Forest        | 97.3%    | 97.6%     | 95.2%   | 96.4%    |

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com) for providing the dataset.
- Open-source libraries such as NumPy, Pandas, Scikit-learn, and Matplotlib.
