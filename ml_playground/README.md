# Machine Learning model playground

Machine Learning model playground is a package for commonly used ML models, where each model has its own class with a fit and predict method.


Registry structure:

ml_playground/
├── pyproject.toml
├── README.md
├── .gitignore
│
├── src/
│   └── ml_playground/
│       ├── __init__.py
│       ├── base.py                             # Defines the base class that each model must inherit
│       ├── registry.py                         # Registry of each model and its class name 
│       └── models/                             # List of each machine learning model class  
│           ├── __init__.py
│           └── linear_regression.py
│
└── tests/                                      # Lists test cases for each model type 
    └── linear_regression_test.py
