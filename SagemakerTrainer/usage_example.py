    # define training args
    training_args = TrainingArguments(
        output_dir="model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32
    )
    # define sagemaker arguments
    sagemaker_args = SagemakerArguments(
        aws_profile ='JohnDoeHF', # aws profile name (local)
        iam_role    = 'SageMakerExecution', # permssions to start job
        task_name   = 'question-answering', # pipeline tasks
        instance_type = 'gpu-small', # or ml.p3.8xlarge
        instance_count = 1 # instance number used
    )
    # create Trainer instance
    sagemaker_trainer = SagemakerTrainer(
        model=model,
        args=training_args,
        sagemaker_args=sagemaker_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    # train model
    sagemaker_trainer.train()