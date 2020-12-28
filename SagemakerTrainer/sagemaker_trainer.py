class SageMakerTrainer:
    def __init__(self, training_args, task, aws_config):
        """Initialize SageMakerTrainer class with all necessary arguments."""

    def _configure_aws_session(self):
        """creates as sagemaker session with the provided aws_config and check a Sagemaker execution role exists, otherwise creates one"""
        self.sess = Sagemaker.Session()
        if iam.get_role(RoleName=role_name)["Role"]["Arn"]:
            self.role = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        else:
            self.role = self._creates_required_iam_role()

    def _creates_required_iam_role(self):
        """Creates IAM execution role for Sagemaker with provided policy document"""
        return boto3.create_iam_role(role_policy)

    def _get_train_script_for_task(self):
        """Gathers the correct training script for the given task"""
        pass

    def _create_sme_experiments(self):
        """creates an AWS Sagemaker experiment"""
        pass

    def upload_data_to_s3(self):
        """uploads datasets files (arrow, json, etc.) to s3."""
        pass

    def train(self):
        """creates an AWS Sagemaker training job"""
        if not self.training_input:
            self.training_input = self.upload_data_to_s3()
        self.train_script = self._get_train_script_for_task()

        self.estimtator = Sagemaker.start_training_job()
        self.estimtator.fit(self.train_input)

    def get_estimator(self):
        """returns the trained Sagemaker estimator"""
        return self.estimator

    def save_model(self):
        """Downloads the trained model and saves it on the local machine"""
        pass

    def get_log(self):
        """return Sagemaker training job logs"""
        pass

    def get_metrics(self):
        """returns DataFrame with evluation results"""
        pass

    def plot_result(self):
        """plots result metrics as chart"""
        pass
