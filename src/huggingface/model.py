"""
The idea is so make a model without the need to create `inference.py`.
Therefore:
* same model as PytorchModel with HF installed
* extra hyperparameters for the "tasks"
* generic inference script on s3 "hard-coded" into source dir and entry point
"""
