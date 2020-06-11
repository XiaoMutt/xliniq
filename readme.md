## Abstract
Being able to discover similar clinical studies is critical for therapeutic product development. However, due to the complexity of medical research and terminology, a query may not contain all the necessary keywords, affecting the final search results. Here, a keyword-recommending algorithm using collaborative filtering is developed and examined. The recommended keywords can be used in the downstream document matching algorithm, providing a potentially more accurate way of retrieving similar documents.

## Contents
- Data Extraction, transformation and load: 
    - Medical Subject Headings (MeSH) term extraction from clinical trial records (documents)
    - Transform MeSH terms and documents to TF-IDF
    - Load TF-IDF to utility matrix

- Model Development
    - Naive mathematical model
    - ReLu activation and parameter regularization
    - Mini-batch gradient descent algorithm
    
- Model Training Validation, and Test
    - Learning rate search
    - Hyperparamter grid search
    - Model performance metrics: mean squared errors and average precision at top K

- Model Additional Usage and Extension
    - Use as a data recovery method
    - Use kernel for feature mapping

## Full Report
Please see xliniq.pdf.