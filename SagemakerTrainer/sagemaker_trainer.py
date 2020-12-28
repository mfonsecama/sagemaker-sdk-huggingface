class SageMakerTrainer:
    def __init__(self, training_args, task, aws_config):
        """Initialize SageMakerTrainer class with all necessary arguments."""

    def upload_data_to_s3(self):
        """uploads datasets files (arrow, json, etc.) to s3."""

    def train(self):
        self.estimtator = Sagemaker.start_training_job()

    def get_estimator(self):
        """returns the trained Sagemaker estimator"""
        return self.estimator

    def save_model(self):
        """Downloads the trained model and saves it on the local machine"""
        pass

    def get_log(self):
        pass

    def get_metrics(self):
        pass

