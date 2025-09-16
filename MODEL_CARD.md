# Model Card — Bank Marketing Classifier

## Intended Use
Binary classification: predict term-deposit subscription using UCI Bank Marketing data.

## Training Data
- Source: UCI ML Repository (`bank-additional-full.csv`)
- Split: 64/16/20 stratified into train/validation/test sets.

## Algorithms
- LightGBM with one-hot encoded categorical features and standardized numerical features.
- Cross-validation folds: 5 (Stratified).
- Optional isotonic calibration.

## Metrics (Validation)
- ROC AUC: <to be filled>
- F1: <to be filled>
- Log Loss: <to be filled>

## Ethical Considerations
- Potential bias in socio-economic features; evaluate fairness metrics before deployment.

## Limitations
- Performance depends on feature drift (economic conditions).
- Not suitable for high-stakes decisions without human oversight.
