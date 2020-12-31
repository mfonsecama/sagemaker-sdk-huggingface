Hello Guys,

i am currently working on how we could edit/extend the fine-tuning scripts from `examples/` to work out-of-the-box within sagemaker. For that i adjusted the [`run_glue.py` script](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py).

To test it I created a [custom huggingface extension for sagemaker](https://github.com/philschmid/sagemaker-sdk-huggingface) where I created a sagemaker compatible docker container and a huggingface estimator.
The container was build with the `transformers==4.1.1` and `datasets==1.1.3`. That is also the reason why I only adjusted the `run_glue.py` and not any other files. The `run_glue.py` can i dynamically pass into the Sagemaker Training Job, but when i would adjust any other files yet i would have to rebuild the container... . For all the functions, which would move to a different directory I added a comment `# Should be moved to path_to_file/filename.py`.

As an Example how you could use this to create a Sagemaker training job using the extension i build you would create an `HuggingFace()` Estimator and then call `.fit()`. The example i used is demonstrated below or you can find it in this [github repostiroy](https://github.com/philschmid/sagemaker-sdk-huggingface/blob/main/examples/06_transformers_existing_training_scripts/sagemaker-notebook.ipynb)

```python
from huggingface.estimator import HuggingFace


huggingface_estimator = HuggingFace(entry_point='run_glue.py',
                            source_dir='../../transformers/examples/text-classification',
                            sagemaker_session=sess,
                            base_job_name='huggingface-sdk-extension',
                            instance_type='ml.p3.2xlarge',
                            instance_count=1,
                            role=role,
                            framework_version={'transformers':'4.1.1','datasets':'1.1.3'},
                            py_version='py3',
                            hyperparameters = {
                                'model_name_or_path': 'distilbert-base-cased',
                                'task_name':'MRPC',
                                'do_train': True,
                                'do_eval': True,
                                'max_seq_length':'128',
                                'per_device_train_batch_size':32,
                                'learning_rate':2e-5,
                                'num_train_epochs': 3.0
                            })

huggingface_estimator.fit()
```

**_Note:_ Sagemaker Requirements**  
In Sagemaker you can define Hyperparameters, which are getting passed into the training script within the `HuggingFace(hyperparameters={})` dictonary. This parameter will be then passed into the training script as named arguments. So the hyperparameters from the example are going to look like this when the training script is executed.
`--do_eval True --do_train True --learning_rate 2e-05 --max_seq_length 128 --model_name_or_path distilbert-base-cased --num_train_epochs 3.0 --output_dir Not defined sagemaker --per_device_train_batch_size 32 --task_name MRPC`.

### How I proceeded

1. I created a function `is_run_on_sagemaker()` to determine if the script is running in a Sagemaker Runtime environment. This function should be move to the `transformers/src/transformers/file_utils.py` file.

2. I had to adjust the `sys.argv` because:

   1. `TrainingArguments` are expecting the parameter `output_dir`, but in a Sagemaker Runtime the output_dir is defined from the enviroment variable `SM_OUTPUT_DATA_DIR`.
   2. `TrainingArguments` are expecting for boolean parameters not a `True` as value. If `--train_do` exist its `True` otherwise its `False`. In Sagemaker you cannot pass keys only so i removed all `True`s from the `sys.argv` at the beginning. A better solution could that we adjust the HfArgumentParser to accept `'True'` for boolean arguments.

3. Therefore i created an `parse_sagemaker_args()` function which:

   - first adds the `--output_dir` with the correct value for Sagemaker
   - Secound parses alle existing environment variables to check if the datasets are passed into training job. When you run a fine-tuning script in sagemaker you can pass data into `.fit()` which is on S3 and will be downloaded before the training starts. I added two options you can either add the the direct S3 uri to a file (e.g. `s3://my-data-bucket/path/to/my/training/data.csv`) or you can add a path (e.g. `s3://my-data-bucket/path/to/data`) and pass the file as hyperparameters `train_file`.
   - Third I had to remove all `True`s from the `sys.argv` for the boolean parameters.

4. Adjusted all file saving and model saving section and added conditions if the script is run on Sagemaker.

#### Testing

I tested it using the jupyter notebook I provided at the top. The log of the training script is attached:

```bash
2020-12-31 08:22:11 Starting - Starting the training job...
2020-12-31 08:22:34 Starting - Launching requested ML instancesProfilerReport-1609402930: InProgress
......
2020-12-31 08:23:35 Starting - Preparing the instances for training......
2020-12-31 08:24:36 Downloading - Downloading input data
2020-12-31 08:24:36 Training - Downloading the training image.....................
2020-12-31 08:28:12 Training - Training image download completed. Training in progress..bash: cannot set terminal process group (-1): Inappropriate ioctl for device
bash: no job control in this shell
2020-12-31 08:28:12,243 sagemaker-training-toolkit INFO     Imported framework sagemaker_pytorch_container.training
2020-12-31 08:28:12,266 sagemaker_pytorch_container.training INFO     Block until all host DNS lookups succeed.
2020-12-31 08:28:12,498 sagemaker_pytorch_container.training INFO     Invoking user training script.
2020-12-31 08:28:12,878 sagemaker-training-toolkit INFO     Installing dependencies from requirements.txt:
/opt/conda/bin/python -m pip install -r requirements.txt
Requirement already satisfied: datasets>=1.1.3 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 1)) (1.1.3)
Requirement already satisfied: protobuf in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 3)) (3.14.0)
Requirement already satisfied: multiprocess in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (0.70.11.1)
Requirement already satisfied: pandas in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (1.1.5)
Requirement already satisfied: tqdm<4.50.0,>=4.27 in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (4.49.0)
Requirement already satisfied: dataclasses in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (0.8)
Requirement already satisfied: requests>=2.19.0 in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (2.25.1)
Requirement already satisfied: xxhash in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (2.0.0)
Requirement already satisfied: pyarrow>=0.17.1 in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (2.0.0)
Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (1.19.1)
Requirement already satisfied: dill in /opt/conda/lib/python3.6/site-packages (from datasets>=1.1.3->-r requirements.txt (line 1)) (0.3.3)
Collecting sentencepiece!=0.1.92
  Downloading sentencepiece-0.1.94-cp36-cp36m-manylinux2014_x86_64.whl (1.1 MB)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests>=2.19.0->datasets>=1.1.3->-r requirements.txt (line 1)) (1.25.11)
Requirement already satisfied: chardet<5,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests>=2.19.0->datasets>=1.1.3->-r requirements.txt (line 1)) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests>=2.19.0->datasets>=1.1.3->-r requirements.txt (line 1)) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests>=2.19.0->datasets>=1.1.3->-r requirements.txt (line 1)) (2020.12.5)
Requirement already satisfied: six>=1.9 in /opt/conda/lib/python3.6/site-packages (from protobuf->-r requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/lib/python3.6/site-packages (from pandas->datasets>=1.1.3->-r requirements.txt (line 1)) (2.8.1)
Requirement already satisfied: pytz>=2017.2 in /opt/conda/lib/python3.6/site-packages (from pandas->datasets>=1.1.3->-r requirements.txt (line 1)) (2020.4)
Installing collected packages: sentencepiece
Successfully installed sentencepiece-0.1.94

2020-12-31 08:28:15,036 sagemaker-training-toolkit INFO     Invoking user script

Training Env:

{
    "additional_framework_parameters": {},
    "channel_input_dirs": {},
    "current_host": "algo-1",
    "framework_module": "sagemaker_pytorch_container.training:main",
    "hosts": [
        "algo-1"
    ],
    "hyperparameters": {
        "task_name": "MRPC",
        "do_train": true,
        "num_train_epochs": 3.0,
        "do_eval": true,
        "max_seq_length": "128",
        "per_device_train_batch_size": 32,
        "learning_rate": 2e-05,
        "model_name_or_path": "distilbert-base-cased"
    },
    "input_config_dir": "/opt/ml/input/config",
    "input_data_config": {},
    "input_dir": "/opt/ml/input",
    "is_master": true,
    "job_name": "huggingface-sdk-extension-2020-12-31-08-22-10-312",
    "log_level": 20,
    "master_hostname": "algo-1",
    "model_dir": "/opt/ml/model",
    "module_dir": "s3://sagemaker-eu-central-1-558105141721/huggingface-sdk-extension-2020-12-31-08-22-10-312/source/sourcedir.tar.gz",
    "module_name": "run_glue",
    "network_interface_name": "eth0",
    "num_cpus": 8,
    "num_gpus": 1,
    "output_data_dir": "/opt/ml/output/data",
    "output_dir": "/opt/ml/output",
    "output_intermediate_dir": "/opt/ml/output/intermediate",
    "resource_config": {
        "current_host": "algo-1",
        "hosts": [
            "algo-1"
        ],
        "network_interface_name": "eth0"
    },
    "user_entry_point": "run_glue.py"
}

Environment variables:

SM_HOSTS=["algo-1"]
SM_NETWORK_INTERFACE_NAME=eth0
SM_HPS={"do_eval":true,"do_train":true,"learning_rate":2e-05,"max_seq_length":"128","model_name_or_path":"distilbert-base-cased","num_train_epochs":3.0,"per_device_train_batch_size":32,"task_name":"MRPC"}
SM_USER_ENTRY_POINT=run_glue.py
SM_FRAMEWORK_PARAMS={}
SM_RESOURCE_CONFIG={"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"}
SM_INPUT_DATA_CONFIG={}
SM_OUTPUT_DATA_DIR=/opt/ml/output/data
SM_CHANNELS=[]
SM_CURRENT_HOST=algo-1
SM_MODULE_NAME=run_glue
SM_LOG_LEVEL=20
SM_FRAMEWORK_MODULE=sagemaker_pytorch_container.training:main
SM_INPUT_DIR=/opt/ml/input
SM_INPUT_CONFIG_DIR=/opt/ml/input/config
SM_OUTPUT_DIR=/opt/ml/output
SM_NUM_CPUS=8
SM_NUM_GPUS=1
SM_MODEL_DIR=/opt/ml/model
SM_MODULE_DIR=s3://sagemaker-eu-central-1-558105141721/huggingface-sdk-extension-2020-12-31-08-22-10-312/source/sourcedir.tar.gz
SM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{},"current_host":"algo-1","framework_module":"sagemaker_pytorch_container.training:main","hosts":["algo-1"],"hyperparameters":{"do_eval":true,"do_train":true,"learning_rate":2e-05,"max_seq_length":"128","model_name_or_path":"distilbert-base-cased","num_train_epochs":3.0,"per_device_train_batch_size":32,"task_name":"MRPC"},"input_config_dir":"/opt/ml/input/config","input_data_config":{},"input_dir":"/opt/ml/input","is_master":true,"job_name":"huggingface-sdk-extension-2020-12-31-08-22-10-312","log_level":20,"master_hostname":"algo-1","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-eu-central-1-558105141721/huggingface-sdk-extension-2020-12-31-08-22-10-312/source/sourcedir.tar.gz","module_name":"run_glue","network_interface_name":"eth0","num_cpus":8,"num_gpus":1,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"},"user_entry_point":"run_glue.py"}
SM_USER_ARGS=["--do_eval","True","--do_train","True","--learning_rate","2e-05","--max_seq_length","128","--model_name_or_path","distilbert-base-cased","--num_train_epochs","3.0","--per_device_train_batch_size","32","--task_name","MRPC"]
SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
SM_HP_TASK_NAME=MRPC
SM_HP_DO_TRAIN=true
SM_HP_NUM_TRAIN_EPOCHS=3.0
SM_HP_DO_EVAL=true
SM_HP_MAX_SEQ_LENGTH=128
SM_HP_PER_DEVICE_TRAIN_BATCH_SIZE=32
SM_HP_LEARNING_RATE=2e-05
SM_HP_MODEL_NAME_OR_PATH=distilbert-base-cased
PYTHONPATH=/opt/ml/code:/opt/conda/bin:/opt/conda/lib/python36.zip:/opt/conda/lib/python3.6:/opt/conda/lib/python3.6/lib-dynload:/opt/conda/lib/python3.6/site-packages

Invoking script with the following command:

/opt/conda/bin/python run_glue.py --do_eval True --do_train True --learning_rate 2e-05 --max_seq_length 128 --model_name_or_path distilbert-base-cased --num_train_epochs 3.0 --per_device_train_batch_size 32 --task_name MRPC


['run_glue.py', '--do_eval', '--do_train', '--learning_rate', '2e-05', '--max_seq_length', '128', '--model_name_or_path', 'distilbert-base-cased', '--num_train_epochs', '3.0', '--per_device_train_batch_size', '32', '--task_name', 'MRPC', '--output_dir', '/opt/ml/output/data']
Downloading and preparing dataset glue/mrpc (download: 1.43 MiB, generated: 1.43 MiB, post-processed: Unknown size, total: 2.85 MiB) to /root/.cache/huggingface/datasets/glue/mrpc/1.0.0/7c99657241149a24692c402a5c3f34d4c9f1df5ac2e4c3759fadea38f6cb29c4...
Dataset glue downloaded and prepared to /root/.cache/huggingface/datasets/glue/mrpc/1.0.0/7c99657241149a24692c402a5c3f34d4c9f1df5ac2e4c3759fadea38f6cb29c4. Subsequent calls will reuse this data.
[2020-12-31 08:28:43.990 algo-1:31 INFO json_config.py:90] Creating hook from json_config at /opt/ml/input/config/debughookconfig.json.
[2020-12-31 08:28:43.991 algo-1:31 INFO hook.py:193] tensorboard_dir has not been set for the hook. SMDebug will not be exporting tensorboard summaries.
[2020-12-31 08:28:43.991 algo-1:31 INFO hook.py:238] Saving to /opt/ml/output/tensors
[2020-12-31 08:28:43.991 algo-1:31 INFO state_store.py:67] The checkpoint config file /opt/ml/input/config/checkpointconfig.json does not exist.
[2020-12-31 08:28:44.017 algo-1:31 INFO hook.py:398] Monitoring the collections: losses
[2020-12-31 08:28:44.017 algo-1:31 INFO hook.py:461] Hook is writing from the hook with pid: 31

[2020-12-31 08:28:45.513 algo-1:31 WARNING hook.py:978] var is not Tensor or list or tuple of Tensors, module_name:distilbert.transformer BaseModelOutput
[2020-12-31 08:28:45.514 algo-1:31 WARNING hook.py:978] var is not Tensor or list or tuple of Tensors, module_name:distilbert BaseModelOutput
[2020-12-31 08:28:45.523 algo-1:31 WARNING hook.py:978] var is not Tensor or list or tuple of Tensors, module_name:DistilBertForSequenceClassification SequenceClassifierOutput
{'epoch': 3.0}
12/31/2020 08:28:19 - WARNING - __main__ -   Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
12/31/2020 08:28:19 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir='/opt/ml/output/data', overwrite_output_dir=False, do_train=True, do_eval=True, do_predict=False, model_parallel=False, evaluation_strategy=<EvaluationStrategy.NO: 'no'>, prediction_loss_only=False, per_device_train_batch_size=32, per_device_eval_batch_size=8, per_gpu_train_batch_size=None, per_gpu_eval_batch_size=None, gradient_accumulation_steps=1, eval_accumulation_steps=None, learning_rate=2e-05, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=-1, warmup_steps=0, logging_dir='runs/Dec31_08-28-19_algo-1', logging_first_step=False, logging_steps=500, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level='O1', local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name='/opt/ml/output/data', disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, fp16_backend='auto', sharded_ddp=False)
#015Downloading:   0%|          | 0.00/8.68k [00:00<?, ?B/s]#015Downloading: 28.7kB [00:00, 16.1MB/s]
#015Downloading:   0%|          | 0.00/4.97k [00:00<?, ?B/s]#015Downloading: 28.7kB [00:00, 19.9MB/s]
#015Downloading: 0.00B [00:00, ?B/s]#015Downloading: 6.22kB [00:00, 3.90MB/s]
#015Downloading: 0.00B [00:00, ?B/s]#015Downloading: 19.7kB [00:00, 106kB/s]#015Downloading: 54.5kB [00:00, 122kB/s]#015Downloading: 124kB [00:00, 152kB/s] #015Downloading: 280kB [00:00, 201kB/s]#015Downloading: 576kB [00:00, 273kB/s]#015Downloading: 959kB [00:01, 369kB/s]#015Downloading: 1.05MB [00:01, 928kB/s]
#015Downloading: 0.00B [00:00, ?B/s]#015Downloading: 19.4kB [00:00, 103kB/s]#015Downloading: 54.3kB [00:00, 119kB/s]#015Downloading: 124kB [00:00, 150kB/s] #015Downloading: 298kB [00:00, 200kB/s]#015Downloading: 441kB [00:00, 582kB/s]
#0150 examples [00:00, ? examples/s]#0151705 examples [00:00, 17044.33 examples/s]#0153300 examples [00:00, 16698.53 examples/s]#015                                          #015#0150 examples [00:00, ? examples/s]#015                                #015#0150 examples [00:00, ? examples/s]#015                                #01512/31/2020 08:28:28 - INFO - filelock -   Lock 139800303634584 acquired on /root/.cache/huggingface/transformers/ebe1ea24d11aa664488b8de5b21e33989008ca78f207d4e30ec6350b693f073f.302bfd1b5e031cc1b17796e0b6e5b242ba2045d31d00f97589e12b458ebff27a.lock
[INFO|file_utils.py:1301] 2020-12-31 08:28:28,367 >> https://huggingface.co/distilbert-base-cased/resolve/main/config.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmplyt9e_gw
#015Downloading:   0%|          | 0.00/411 [00:00<?, ?B/s]#015Downloading: 100%|██████████| 411/411 [00:00<00:00, 496kB/s]
[INFO|file_utils.py:1305] 2020-12-31 08:28:28,649 >> storing https://huggingface.co/distilbert-base-cased/resolve/main/config.json in cache at /root/.cache/huggingface/transformers/ebe1ea24d11aa664488b8de5b21e33989008ca78f207d4e30ec6350b693f073f.302bfd1b5e031cc1b17796e0b6e5b242ba2045d31d00f97589e12b458ebff27a
[INFO|file_utils.py:1308] 2020-12-31 08:28:28,649 >> creating metadata file for /root/.cache/huggingface/transformers/ebe1ea24d11aa664488b8de5b21e33989008ca78f207d4e30ec6350b693f073f.302bfd1b5e031cc1b17796e0b6e5b242ba2045d31d00f97589e12b458ebff27a
2020-12-31 08:29:30,381 sagemaker-training-toolkit INFO     Reporting training SUCCESS
12/31/2020 08:28:28 - INFO - filelock -   Lock 139800303634584 released on /root/.cache/huggingface/transformers/ebe1ea24d11aa664488b8de5b21e33989008ca78f207d4e30ec6350b693f073f.302bfd1b5e031cc1b17796e0b6e5b242ba2045d31d00f97589e12b458ebff27a.lock
[INFO|configuration_utils.py:431] 2020-12-31 08:28:28,650 >> loading configuration file https://huggingface.co/distilbert-base-cased/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/ebe1ea24d11aa664488b8de5b21e33989008ca78f207d4e30ec6350b693f073f.302bfd1b5e031cc1b17796e0b6e5b242ba2045d31d00f97589e12b458ebff27a
[INFO|configuration_utils.py:467] 2020-12-31 08:28:28,651 >> Model config DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "finetuning_task": "mrpc",
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "output_past": true,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "tie_weights_": true,
  "vocab_size": 28996
}

[INFO|configuration_utils.py:431] 2020-12-31 08:28:28,933 >> loading configuration file https://huggingface.co/distilbert-base-cased/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/ebe1ea24d11aa664488b8de5b21e33989008ca78f207d4e30ec6350b693f073f.302bfd1b5e031cc1b17796e0b6e5b242ba2045d31d00f97589e12b458ebff27a
[INFO|configuration_utils.py:467] 2020-12-31 08:28:28,933 >> Model config DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "output_past": true,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "tie_weights_": true,
  "vocab_size": 28996
}

12/31/2020 08:28:29 - INFO - filelock -   Lock 139797608840104 acquired on /root/.cache/huggingface/transformers/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791.lock
[INFO|file_utils.py:1301] 2020-12-31 08:28:29,217 >> https://huggingface.co/bert-base-cased/resolve/main/vocab.txt not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpvm6yksc0
#015Downloading:   0%|          | 0.00/213k [00:00<?, ?B/s]#015Downloading:  17%|█▋        | 36.9k/213k [00:00<00:00, 212kB/s]#015Downloading:  94%|█████████▍| 201k/213k [00:00<00:00, 282kB/s] #015Downloading: 100%|██████████| 213k/213k [00:00<00:00, 604kB/s]
[INFO|file_utils.py:1305] 2020-12-31 08:28:29,855 >> storing https://huggingface.co/bert-base-cased/resolve/main/vocab.txt in cache at /root/.cache/huggingface/transformers/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
[INFO|file_utils.py:1308] 2020-12-31 08:28:29,855 >> creating metadata file for /root/.cache/huggingface/transformers/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
12/31/2020 08:28:29 - INFO - filelock -   Lock 139797608840104 released on /root/.cache/huggingface/transformers/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791.lock
12/31/2020 08:28:30 - INFO - filelock -   Lock 139797608841112 acquired on /root/.cache/huggingface/transformers/226a307193a9f4344264cdc76a12988448a25345ba172f2c7421f3b6810fddad.3dab63143af66769bbb35e3811f75f7e16b2320e12b7935e216bd6159ce6d9a6.lock
[INFO|file_utils.py:1301] 2020-12-31 08:28:30,143 >> https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmp5vnay570
#015Downloading:   0%|          | 0.00/436k [00:00<?, ?B/s]#015Downloading:   8%|▊         | 36.9k/436k [00:00<00:01, 214kB/s]#015Downloading:  46%|████▌     | 201k/436k [00:00<00:00, 284kB/s] #015Downloading: 100%|██████████| 436k/436k [00:00<00:00, 1.10MB/s]
[INFO|file_utils.py:1305] 2020-12-31 08:28:30,827 >> storing https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json in cache at /root/.cache/huggingface/transformers/226a307193a9f4344264cdc76a12988448a25345ba172f2c7421f3b6810fddad.3dab63143af66769bbb35e3811f75f7e16b2320e12b7935e216bd6159ce6d9a6
[INFO|file_utils.py:1308] 2020-12-31 08:28:30,827 >> creating metadata file for /root/.cache/huggingface/transformers/226a307193a9f4344264cdc76a12988448a25345ba172f2c7421f3b6810fddad.3dab63143af66769bbb35e3811f75f7e16b2320e12b7935e216bd6159ce6d9a6
12/31/2020 08:28:30 - INFO - filelock -   Lock 139797608841112 released on /root/.cache/huggingface/transformers/226a307193a9f4344264cdc76a12988448a25345ba172f2c7421f3b6810fddad.3dab63143af66769bbb35e3811f75f7e16b2320e12b7935e216bd6159ce6d9a6.lock
[INFO|tokenization_utils_base.py:1802] 2020-12-31 08:28:30,827 >> loading file https://huggingface.co/bert-base-cased/resolve/main/vocab.txt from cache at /root/.cache/huggingface/transformers/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
[INFO|tokenization_utils_base.py:1802] 2020-12-31 08:28:30,827 >> loading file https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json from cache at /root/.cache/huggingface/transformers/226a307193a9f4344264cdc76a12988448a25345ba172f2c7421f3b6810fddad.3dab63143af66769bbb35e3811f75f7e16b2320e12b7935e216bd6159ce6d9a6
12/31/2020 08:28:31 - INFO - filelock -   Lock 139800303634584 acquired on /root/.cache/huggingface/transformers/9c9f39769dba4c5fe379b4bc82973eb01297bd607954621434eb9f1bc85a23a0.06b428c87335c1bb22eae46fdab31c8286efa0aa09e898a7ac42ddf5c3f5dc19.lock
[INFO|file_utils.py:1301] 2020-12-31 08:28:31,151 >> https://huggingface.co/distilbert-base-cased/resolve/main/pytorch_model.bin not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpi2h8yubw
#015Downloading:   0%|          | 0.00/263M [00:00<?, ?B/s]#015Downloading:   2%|▏         | 4.13M/263M [00:00<00:06, 41.3MB/s]#015Downloading:   3%|▎         | 8.25M/263M [00:00<00:06, 41.2MB/s]#015Downloading:   5%|▍         | 12.8M/263M [00:00<00:05, 42.4MB/s]#015Downloading:   7%|▋         | 17.5M/263M [00:00<00:05, 43.8MB/s]#015Downloading:   9%|▊         | 22.4M/263M [00:00<00:05, 45.2MB/s]#015Downloading:  10%|█         | 27.3M/263M [00:00<00:05, 46.2MB/s]#015Downloading:  12%|█▏        | 32.2M/263M [00:00<00:04, 47.2MB/s]#015Downloading:  14%|█▍        | 37.3M/263M [00:00<00:04, 48.1MB/s]#015Downloading:  16%|█▌        | 42.3M/263M [00:00<00:04, 48.7MB/s]#015Downloading:  18%|█▊        | 47.3M/263M [00:01<00:04, 49.1MB/s]#015Downloading:  20%|█▉        | 52.3M/263M [00:01<00:04, 49.4MB/s]#015Downloading:  22%|██▏       | 57.6M/263M [00:01<00:04, 50.4MB/s]#015Downloading:  24%|██▍       | 63.7M/263M [00:01<00:03, 53.3MB/s]#015Downloading:  27%|██▋       | 69.9M/263M [00:01<00:03, 55.6MB/s]#015Downloading:  29%|██▉       | 76.1M/263M [00:01<00:03, 57.3MB/s]#015Downloading:  31%|███▏      | 82.3M/263M [00:01<00:03, 58.6MB/s]#015Downloading:  33%|███▎      | 88.2M/263M [00:01<00:02, 58.6MB/s]#015Downloading:  36%|███▌      | 94.5M/263M [00:01<00:02, 59.8MB/s]#015Downloading:  38%|███▊      | 101M/263M [00:01<00:02, 60.7MB/s] #015Downloading:  41%|████      | 107M/263M [00:02<00:02, 57.8MB/s]#015Downloading:  43%|████▎     | 113M/263M [00:02<00:02, 55.2MB/s]#015Downloading:  45%|████▍     | 118M/263M [00:02<00:02, 52.6MB/s]#015Downloading:  47%|████▋     | 124M/263M [00:02<00:02, 51.7MB/s]#015Downloading:  49%|████▉     | 129M/263M [00:02<00:02, 51.1MB/s]#015Downloading:  51%|█████     | 134M/263M [00:02<00:02, 50.8MB/s]#015Downloading:  53%|█████▎    | 139M/263M [00:02<00:02, 50.7MB/s]#015Downloading:  55%|█████▍    | 144M/263M [00:02<00:02, 49.6MB/s]#015Downloading:  57%|█████▋    | 149M/263M [00:02<00:02, 49.7MB/s]#015Downloading:  59%|█████▊    | 154M/263M [00:02<00:02, 49.9MB/s]#015Downloading:  60%|██████    | 159M/263M [00:03<00:02, 49.9MB/s]#015Downloading:  62%|██████▏   | 164M/263M [00:03<00:01, 49.6MB/s]#015Downloading:  64%|██████▍   | 169M/263M [00:03<00:01, 49.7MB/s]#015Downloading:  66%|██████▌   | 174M/263M [00:03<00:01, 49.8MB/s]#015Downloading:  68%|██████▊   | 179M/263M [00:03<00:01, 49.9MB/s]#015Downloading:  70%|██████▉   | 184M/263M [00:03<00:01, 49.9MB/s]#015Downloading:  72%|███████▏  | 189M/263M [00:03<00:01, 50.0MB/s]#015Downloading:  74%|███████▍  | 194M/263M [00:03<00:01, 50.0MB/s]#015Downloading:  76%|███████▌  | 199M/263M [00:03<00:01, 50.1MB/s]#015Downloading:  78%|███████▊  | 205M/263M [00:03<00:01, 51.3MB/s]#015Downloading:  80%|████████  | 211M/263M [00:04<00:00, 53.9MB/s]#015Downloading:  82%|████████▏ | 217M/263M [00:04<00:00, 56.1MB/s]#015Downloading:  85%|████████▍ | 223M/263M [00:04<00:00, 57.3MB/s]#015Downloading:  87%|████████▋ | 229M/263M [00:04<00:00, 58.6MB/s]#015Downloading:  89%|████████▉ | 235M/263M [00:04<00:00, 59.7MB/s]#015Downloading:  92%|█████████▏| 241M/263M [00:04<00:00, 58.4MB/s]#015Downloading:  94%|█████████▍| 247M/263M [00:04<00:00, 52.6MB/s]#015Downloading:  96%|█████████▌| 253M/263M [00:04<00:00, 51.7MB/s]#015Downloading:  98%|█████████▊| 258M/263M [00:04<00:00, 50.8MB/s]#015Downloading: 100%|█████████▉| 263M/263M [00:05<00:00, 50.9MB/s]#015Downloading: 100%|██████████| 263M/263M [00:05<00:00, 52.2MB/s]
[INFO|file_utils.py:1305] 2020-12-31 08:28:36,253 >> storing https://huggingface.co/distilbert-base-cased/resolve/main/pytorch_model.bin in cache at /root/.cache/huggingface/transformers/9c9f39769dba4c5fe379b4bc82973eb01297bd607954621434eb9f1bc85a23a0.06b428c87335c1bb22eae46fdab31c8286efa0aa09e898a7ac42ddf5c3f5dc19
[INFO|file_utils.py:1308] 2020-12-31 08:28:36,253 >> creating metadata file for /root/.cache/huggingface/transformers/9c9f39769dba4c5fe379b4bc82973eb01297bd607954621434eb9f1bc85a23a0.06b428c87335c1bb22eae46fdab31c8286efa0aa09e898a7ac42ddf5c3f5dc19
12/31/2020 08:28:36 - INFO - filelock -   Lock 139800303634584 released on /root/.cache/huggingface/transformers/9c9f39769dba4c5fe379b4bc82973eb01297bd607954621434eb9f1bc85a23a0.06b428c87335c1bb22eae46fdab31c8286efa0aa09e898a7ac42ddf5c3f5dc19.lock
[INFO|modeling_utils.py:1024] 2020-12-31 08:28:36,253 >> loading weights file https://huggingface.co/distilbert-base-cased/resolve/main/pytorch_model.bin from cache at /root/.cache/huggingface/transformers/9c9f39769dba4c5fe379b4bc82973eb01297bd607954621434eb9f1bc85a23a0.06b428c87335c1bb22eae46fdab31c8286efa0aa09e898a7ac42ddf5c3f5dc19
[WARNING|modeling_utils.py:1132] 2020-12-31 08:28:38,515 >> Some weights of the model checkpoint at distilbert-base-cased were not used when initializing DistilBertForSequenceClassification: ['vocab_transform.weight', 'vocab_transform.bias', 'vocab_layer_norm.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_projector.bias']
- This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[WARNING|modeling_utils.py:1143] 2020-12-31 08:28:38,515 >> Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-cased and are newly initialized: ['pre_classifier.weight', 'pre_classifier.bias', 'classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
#015  0%|          | 0/4 [00:00<?, ?ba/s]#015 25%|██▌       | 1/4 [00:00<00:00,  9.17ba/s]#015 75%|███████▌  | 3/4 [00:00<00:00, 10.17ba/s]#015100%|██████████| 4/4 [00:00<00:00, 13.12ba/s]
#015  0%|          | 0/1 [00:00<?, ?ba/s]#015100%|██████████| 1/1 [00:00<00:00, 29.95ba/s]
#015  0%|          | 0/2 [00:00<?, ?ba/s]#015100%|██████████| 2/2 [00:00<00:00, 14.81ba/s]#015100%|██████████| 2/2 [00:00<00:00, 14.77ba/s]
12/31/2020 08:28:39 - INFO - __main__ -   Sample 2619 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'idx': 2916, 'input_ids': [101, 1109, 10830, 1127, 1678, 1146, 1114, 24987, 1149, 13260, 1147, 1692, 1222, 7277, 2180, 5303, 117, 3455, 3081, 5097, 1104, 4961, 1149, 13260, 9966, 1222, 1140, 119, 102, 20661, 1127, 1678, 1146, 1114, 24987, 1149, 13260, 1147, 1692, 1222, 7277, 2180, 5303, 117, 3455, 170, 3081, 118, 3674, 21100, 2998, 1106, 1103, 2175, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label': 1, 'sentence1': 'The proceedings were taken up with prosecutors outlining their case against Amrozi , reading 33 pages of documents outlining allegations against him .', 'sentence2': 'Proceedings were taken up with prosecutors outlining their case against Amrozi , reading a 33-page accusation letter to the court .'}.
12/31/2020 08:28:39 - INFO - __main__ -   Sample 456 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'idx': 509, 'input_ids': [101, 20394, 11252, 1424, 3878, 1684, 1111, 1103, 4116, 118, 5534, 1433, 1132, 170, 6539, 4010, 1111, 9283, 1105, 6646, 1110, 1919, 1344, 3075, 1104, 1397, 3625, 112, 188, 5200, 1728, 1107, 1594, 118, 7820, 20394, 11252, 15449, 119, 102, 9018, 1116, 1107, 20394, 11252, 15449, 112, 188, 4116, 118, 5534, 1433, 1132, 170, 6539, 4010, 1111, 9283, 117, 1105, 6646, 1110, 1919, 1344, 3075, 1104, 3625, 112, 188, 5200, 1728, 1107, 1103, 1594, 118, 187, 15677, 3660, 1805, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label': 1, 'sentence1': "Chechen officials working for the Moscow-backed government are a frequent target for rebels and tension is running high ahead of next Sunday 's presidential election in war-torn Chechnya .", 'sentence2': "Officials in Chechnya 's Moscow-backed government are a frequent target for rebels , and tension is running high ahead of Sunday 's presidential election in the war-ravaged region ."}.
12/31/2020 08:28:39 - INFO - __main__ -   Sample 102 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'idx': 116, 'input_ids': [101, 6433, 111, 11767, 112, 188, 2260, 4482, 7448, 2174, 1116, 5799, 125, 119, 1969, 1827, 1106, 5103, 1495, 119, 1851, 117, 1229, 11896, 1116, 1810, 4426, 2174, 1116, 2204, 127, 119, 126, 1827, 1106, 122, 117, 20278, 119, 1851, 119, 102, 1109, 6433, 111, 11767, 112, 188, 2260, 10146, 1108, 1146, 122, 119, 3453, 1827, 117, 1137, 121, 119, 1407, 3029, 117, 1106, 5311, 1559, 119, 5599, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label': 0, 'sentence1': "Standard & Poor 's 500 stock index futures declined 4.40 points to 983.50 , while Nasdaq futures fell 6.5 points to 1,206.50 .", 'sentence2': "The Standard & Poor 's 500 Index was up 1.75 points , or 0.18 percent , to 977.68 ."}.
#015Downloading:   0%|          | 0.00/1.67k [00:00<?, ?B/s]#015Downloading: 4.39kB [00:00, 3.86MB/s]
[INFO|trainer.py:388] 2020-12-31 08:28:43,678 >> The following columns in the training set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: sentence2, idx, sentence1.
[INFO|trainer.py:388] 2020-12-31 08:28:43,678 >> The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: sentence2, idx, sentence1.
[INFO|trainer.py:703] 2020-12-31 08:28:43,680 >> ***** Running training *****
[INFO|trainer.py:704] 2020-12-31 08:28:43,680 >>   Num examples = 3668
[INFO|trainer.py:705] 2020-12-31 08:28:43,680 >>   Num Epochs = 3
[INFO|trainer.py:706] 2020-12-31 08:28:43,680 >>   Instantaneous batch size per device = 32
[INFO|trainer.py:707] 2020-12-31 08:28:43,680 >>   Total train batch size (w. parallel, distributed & accumulation) = 32
[INFO|trainer.py:708] 2020-12-31 08:28:43,680 >>   Gradient Accumulation steps = 1
[INFO|trainer.py:709] 2020-12-31 08:28:43,680 >>   Total optimization steps = 345
#015  0%|          | 0/345 [00:00<?, ?it/s]#015  0%|          | 1/345 [00:02<11:36,  2.03s/it]#015  1%|          | 2/345 [00:02<08:19,  1.46s/it]#015  1%|          | 3/345 [00:02<06:01,  1.06s/it]#015  1%|          | 4/345 [00:02<04:24,  1.29it/s]#015  1%|▏         | 5/345 [00:02<03:17,  1.72it/s]#015  2%|▏         | 6/345 [00:02<02:30,  2.26it/s]#015  2%|▏         | 7/345 [00:02<01:57,  2.88it/s]#015  2%|▏         | 8/345 [00:02<01:34,  3.57it/s]#015  3%|▎         | 9/345 [00:03<01:18,  4.29it/s]#015  3%|▎         | 10/345 [00:03<01:07,  4.99it/s]#015  3%|▎         | 11/345 [00:03<00:59,  5.64it/s]#015  3%|▎         | 12/345 [00:03<00:53,  6.22it/s]#015  4%|▍         | 13/345 [00:03<00:49,  6.71it/s]#015  4%|▍         | 14/345 [00:03<00:46,  7.09it/s]#015  4%|▍         | 15/345 [00:03<00:44,  7.40it/s]#015  5%|▍         | 16/345 [00:03<00:43,  7.57it/s]#015  5%|▍         | 17/345 [00:03<00:42,  7.75it/s]#015  5%|▌         | 18/345 [00:04<00:41,  7.85it/s]#015  6%|▌         | 19/345 [00:04<00:40,  7.96it/s]#015  6%|▌         | 20/345 [00:04<00:40,  8.02it/s]#015  6%|▌         | 21/345 [00:04<00:40,  8.07it/s]#015  6%|▋         | 22/345 [00:04<00:40,  8.03it/s]#015  7%|▋         | 23/345 [00:04<00:40,  8.04it/s]#015  7%|▋         | 24/345 [00:04<00:39,  8.07it/s]#015  7%|▋         | 25/345 [00:04<00:39,  8.07it/s]#015  8%|▊         | 26/345 [00:05<00:39,  8.11it/s]#015  8%|▊         | 27/345 [00:05<00:39,  8.11it/s]#015  8%|▊         | 28/345 [00:05<00:38,  8.14it/s]#015  8%|▊         | 29/345 [00:05<00:39,  8.10it/s]#015  9%|▊         | 30/345 [00:05<00:39,  8.06it/s]#015  9%|▉         | 31/345 [00:05<00:38,  8.10it/s]#015  9%|▉         | 32/345 [00:05<00:38,  8.13it/s]#015 10%|▉         | 33/345 [00:05<00:38,  8.12it/s]#015 10%|▉         | 34/345 [00:06<00:38,  8.14it/s]#015 10%|█         | 35/345 [00:06<00:38,  8.12it/s]#015 10%|█         | 36/345 [00:06<00:38,  8.10it/s]#015 11%|█         | 37/345 [00:06<00:38,  8.10it/s]#015 11%|█         | 38/345 [00:06<00:37,  8.13it/s]#015 11%|█▏        | 39/345 [00:06<00:37,  8.10it/s]#015 12%|█▏        | 40/345 [00:06<00:37,  8.08it/s]#015 12%|█▏        | 41/345 [00:06<00:37,  8.09it/s]#015 12%|█▏        | 42/345 [00:07<00:37,  8.08it/s]#015 12%|█▏        | 43/345 [00:07<00:37,  8.09it/s]#015 13%|█▎        | 44/345 [00:07<00:37,  8.09it/s]#015 13%|█▎        | 45/345 [00:07<00:37,  8.09it/s]#015 13%|█▎        | 46/345 [00:07<00:37,  8.08it/s]#015 14%|█▎        | 47/345 [00:07<00:36,  8.08it/s]#015 14%|█▍        | 48/345 [00:07<00:36,  8.08it/s]#015 14%|█▍        | 49/345 [00:07<00:36,  8.08it/s]#015 14%|█▍        | 50/345 [00:08<00:36,  8.07it/s]#015 15%|█▍        | 51/345 [00:08<00:36,  8.08it/s]#015 15%|█▌        | 52/345 [00:08<00:36,  8.09it/s]#015 15%|█▌        | 53/345 [00:08<00:36,  8.10it/s]#015 16%|█▌        | 54/345 [00:08<00:35,  8.10it/s]#015 16%|█▌        | 55/345 [00:08<00:35,  8.09it/s]#015 16%|█▌        | 56/345 [00:08<00:35,  8.09it/s]#015 17%|█▋        | 57/345 [00:08<00:35,  8.08it/s]#015 17%|█▋        | 58/345 [00:09<00:35,  8.08it/s]#015 17%|█▋        | 59/345 [00:09<00:35,  8.02it/s]#015 17%|█▋        | 60/345 [00:09<00:35,  8.04it/s]#015 18%|█▊        | 61/345 [00:09<00:35,  7.95it/s]#015 18%|█▊        | 62/345 [00:09<00:35,  7.93it/s]#015 18%|█▊        | 63/345 [00:09<00:35,  7.97it/s]#015 19%|█▊        | 64/345 [00:09<00:35,  8.00it/s]#015 19%|█▉        | 65/345 [00:09<00:35,  7.99it/s]#015 19%|█▉        | 66/345 [00:10<00:34,  8.02it/s]#015 19%|█▉        | 67/345 [00:10<00:34,  8.04it/s]#015 20%|█▉        | 68/345 [00:10<00:34,  8.06it/s]#015 20%|██        | 69/345 [00:10<00:34,  8.08it/s]#015 20%|██        | 70/345 [00:10<00:34,  8.08it/s]#015 21%|██        | 71/345 [00:10<00:33,  8.07it/s]#015 21%|██        | 72/345 [00:10<00:33,  8.07it/s]#015 21%|██        | 73/345 [00:10<00:33,  8.03it/s]#015 21%|██▏       | 74/345 [00:11<00:33,  8.01it/s]#015 22%|██▏       | 75/345 [00:11<00:33,  8.03it/s]#015 22%|██▏       | 76/345 [00:11<00:33,  8.04it/s]#015 22%|██▏       | 77/345 [00:11<00:33,  8.05it/s]#015 23%|██▎       | 78/345 [00:11<00:33,  8.06it/s]#015 23%|██▎       | 79/345 [00:11<00:33,  8.06it/s]#015 23%|██▎       | 80/345 [00:11<00:32,  8.07it/s]#015 23%|██▎       | 81/345 [00:11<00:32,  8.07it/s]#015 24%|██▍       | 82/345 [00:12<00:32,  8.07it/s]#015 24%|██▍       | 83/345 [00:12<00:32,  8.08it/s]#015 24%|██▍       | 84/345 [00:12<00:32,  8.08it/s]#015 25%|██▍       | 85/345 [00:12<00:32,  8.09it/s]#015 25%|██▍       | 86/345 [00:12<00:32,  8.08it/s]#015 25%|██▌       | 87/345 [00:12<00:31,  8.08it/s]#015 26%|██▌       | 88/345 [00:12<00:31,  8.08it/s]#015 26%|██▌       | 89/345 [00:12<00:31,  8.10it/s]#015 26%|██▌       | 90/345 [00:13<00:31,  8.09it/s]#015 26%|██▋       | 91/345 [00:13<00:31,  8.08it/s]#015 27%|██▋       | 92/345 [00:13<00:31,  8.08it/s]#015 27%|██▋       | 93/345 [00:13<00:31,  8.08it/s]#015 27%|██▋       | 94/345 [00:13<00:31,  8.08it/s]#015 28%|██▊       | 95/345 [00:13<00:30,  8.09it/s]#015 28%|██▊       | 96/345 [00:13<00:30,  8.08it/s]#015 28%|██▊       | 97/345 [00:13<00:30,  8.09it/s]#015 28%|██▊       | 98/345 [00:14<00:30,  8.09it/s]#015 29%|██▊       | 99/345 [00:14<00:30,  8.08it/s]#015 29%|██▉       | 100/345 [00:14<00:30,  8.08it/s]#015 29%|██▉       | 101/345 [00:14<00:30,  8.09it/s]#015 30%|██▉       | 102/345 [00:14<00:30,  7.97it/s]#015 30%|██▉       | 103/345 [00:14<00:30,  7.98it/s]#015 30%|███       | 104/345 [00:14<00:30,  7.99it/s]#015 30%|███       | 105/345 [00:14<00:30,  7.99it/s]#015 31%|███       | 106/345 [00:15<00:29,  7.99it/s]#015 31%|███       | 107/345 [00:15<00:29,  8.00it/s]#015 31%|███▏      | 108/345 [00:15<00:29,  8.01it/s]#015 32%|███▏      | 109/345 [00:15<00:29,  8.02it/s]#015 32%|███▏      | 110/345 [00:15<00:29,  8.01it/s]#015 32%|███▏      | 111/345 [00:15<00:29,  8.00it/s]#015 32%|███▏      | 112/345 [00:15<00:29,  8.00it/s]#015 33%|███▎      | 113/345 [00:15<00:28,  8.00it/s]#015 33%|███▎      | 114/345 [00:16<00:28,  8.00it/s]#015 34%|███▎      | 116/345 [00:16<00:27,  8.39it/s]#015 34%|███▍      | 117/345 [00:16<00:27,  8.30it/s]#015 34%|███▍      | 118/345 [00:16<00:27,  8.24it/s]#015 34%|███▍      | 119/345 [00:16<00:27,  8.19it/s]#015 35%|███▍      | 120/345 [00:16<00:27,  8.12it/s]#015 35%|███▌      | 121/345 [00:16<00:27,  8.11it/s]#015 35%|███▌      | 122/345 [00:16<00:27,  8.10it/s]#015 36%|███▌      | 123/345 [00:17<00:27,  8.09it/s]#015 36%|███▌      | 124/345 [00:17<00:27,  8.09it/s]#015 36%|███▌      | 125/345 [00:17<00:27,  8.09it/s]#015 37%|███▋      | 126/345 [00:17<00:27,  8.10it/s]#015 37%|███▋      | 127/345 [00:17<00:26,  8.09it/s]#015 37%|███▋      | 128/345 [00:17<00:26,  8.04it/s]#015 37%|███▋      | 129/345 [00:17<00:26,  8.04it/s]#015 38%|███▊      | 130/345 [00:17<00:26,  8.05it/s]#015 38%|███▊      | 131/345 [00:18<00:26,  8.06it/s]#015 38%|███▊      | 132/345 [00:18<00:26,  8.06it/s]#015 39%|███▊      | 133/345 [00:18<00:26,  8.06it/s]#015 39%|███▉      | 134/345 [00:18<00:26,  8.06it/s]#015 39%|███▉      | 135/345 [00:18<00:26,  8.06it/s]#015 39%|███▉      | 136/345 [00:18<00:25,  8.07it/s]#015 40%|███▉      | 137/345 [00:18<00:25,  8.06it/s]#015 40%|████      | 138/345 [00:18<00:25,  8.06it/s]#015 40%|████      | 139/345 [00:19<00:25,  8.05it/s]#015 41%|████      | 140/345 [00:19<00:25,  8.07it/s]#015 41%|████      | 141/345 [00:19<00:25,  8.08it/s]#015 41%|████      | 142/345 [00:19<00:25,  8.09it/s]#015 41%|████▏     | 143/345 [00:19<00:24,  8.09it/s]#015 42%|████▏     | 144/345 [00:19<00:24,  8.10it/s]#015 42%|████▏     | 145/345 [00:19<00:24,  8.10it/s]#015 42%|████▏     | 146/345 [00:19<00:24,  8.10it/s]#015 43%|████▎     | 147/345 [00:20<00:24,  8.10it/s]#015 43%|████▎     | 148/345 [00:20<00:24,  8.11it/s]#015 43%|████▎     | 149/345 [00:20<00:24,  8.12it/s]#015 43%|████▎     | 150/345 [00:20<00:24,  8.12it/s]#015 44%|████▍     | 151/345 [00:20<00:23,  8.12it/s]#015 44%|████▍     | 152/345 [00:20<00:23,  8.13it/s]#015 44%|████▍     | 153/345 [00:20<00:23,  8.11it/s]#015 45%|████▍     | 154/345 [00:20<00:23,  8.11it/s]#015 45%|████▍     | 155/345 [00:21<00:23,  8.03it/s]#015 45%|████▌     | 156/345 [00:21<00:23,  8.05it/s]#015 46%|████▌     | 157/345 [00:21<00:23,  8.07it/s]#015 46%|████▌     | 158/345 [00:21<00:23,  8.08it/s]#015 46%|████▌     | 159/345 [00:21<00:22,  8.09it/s]#015 46%|████▋     | 160/345 [00:21<00:22,  8.10it/s]#015 47%|████▋     | 161/345 [00:21<00:22,  8.11it/s]#015 47%|████▋     | 162/345 [00:21<00:22,  8.10it/s]#015 47%|████▋     | 163/345 [00:22<00:22,  7.95it/s]#015 48%|████▊     | 164/345 [00:22<00:23,  7.75it/s]#015 48%|████▊     | 165/345 [00:22<00:23,  7.68it/s]#015 48%|████▊     | 166/345 [00:22<00:23,  7.74it/s]#015 48%|████▊     | 167/345 [00:22<00:22,  7.81it/s]#015 49%|████▊     | 168/345 [00:22<00:22,  7.86it/s]#015 49%|████▉     | 169/345 [00:22<00:22,  7.89it/s]#015 49%|████▉     | 170/345 [00:22<00:22,  7.93it/s]#015 50%|████▉     | 171/345 [00:23<00:21,  7.93it/s]#015 50%|████▉     | 172/345 [00:23<00:21,  7.98it/s]#015 50%|█████     | 173/345 [00:23<00:21,  8.03it/s]#015 50%|█████     | 174/345 [00:23<00:21,  8.05it/s]#015 51%|█████     | 175/345 [00:23<00:21,  8.08it/s]#015 51%|█████     | 176/345 [00:23<00:20,  8.09it/s]#015 51%|█████▏    | 177/345 [00:23<00:20,  8.10it/s]#015 52%|█████▏    | 178/345 [00:23<00:20,  8.10it/s]#015 52%|█████▏    | 179/345 [00:24<00:20,  8.09it/s]#015 52%|█████▏    | 180/345 [00:24<00:20,  8.10it/s]#015 52%|█████▏    | 181/345 [00:24<00:20,  8.10it/s]#015 53%|█████▎    | 182/345 [00:24<00:20,  8.09it/s]#015 53%|█████▎    | 183/345 [00:24<00:20,  8.07it/s]#015 53%|█████▎    | 184/345 [00:24<00:19,  8.07it/s]#015 54%|█████▎    | 185/345 [00:24<00:19,  8.07it/s]#015 54%|█████▍    | 186/345 [00:24<00:19,  8.07it/s]#015 54%|█████▍    | 187/345 [00:25<00:19,  8.07it/s]#015 54%|█████▍    | 188/345 [00:25<00:19,  8.07it/s]#015 55%|█████▍    | 189/345 [00:25<00:19,  8.07it/s]#015 55%|█████▌    | 190/345 [00:25<00:19,  8.06it/s]#015 55%|█████▌    | 191/345 [00:25<00:19,  8.07it/s]#015 56%|█████▌    | 192/345 [00:25<00:18,  8.07it/s]#015 56%|█████▌    | 193/345 [00:25<00:18,  8.07it/s]#015 56%|█████▌    | 194/345 [00:25<00:18,  8.07it/s]#015 57%|█████▋    | 195/345 [00:26<00:18,  8.07it/s]#015 57%|█████▋    | 196/345 [00:26<00:18,  8.07it/s]#015 57%|█████▋    | 197/345 [00:26<00:18,  8.07it/s]#015 57%|█████▋    | 198/345 [00:26<00:18,  8.06it/s]#015 58%|█████▊    | 199/345 [00:26<00:18,  8.06it/s]#015 58%|█████▊    | 200/345 [00:26<00:17,  8.07it/s]#015 58%|█████▊    | 201/345 [00:26<00:17,  8.08it/s]#015 59%|█████▊    | 202/345 [00:26<00:17,  8.08it/s]#015 59%|█████▉    | 203/345 [00:27<00:17,  8.07it/s]#015 59%|█████▉    | 204/345 [00:27<00:17,  8.06it/s]#015 59%|█████▉    | 205/345 [00:27<00:17,  8.07it/s]#015 60%|█████▉    | 206/345 [00:27<00:17,  8.06it/s]#015 60%|██████    | 207/345 [00:27<00:17,  8.05it/s]#015 60%|██████    | 208/345 [00:27<00:17,  8.06it/s]#015 61%|██████    | 209/345 [00:27<00:16,  8.06it/s]#015 61%|██████    | 210/345 [00:27<00:16,  8.06it/s]#015 61%|██████    | 211/345 [00:28<00:16,  8.06it/s]#015 61%|██████▏   | 212/345 [00:28<00:16,  8.05it/s]#015 62%|██████▏   | 213/345 [00:28<00:16,  8.06it/s]#015 62%|██████▏   | 214/345 [00:28<00:16,  8.06it/s]#015 62%|██████▏   | 215/345 [00:28<00:16,  8.07it/s]#015 63%|██████▎   | 216/345 [00:28<00:15,  8.07it/s]#015 63%|██████▎   | 217/345 [00:28<00:15,  8.07it/s]#015 63%|██████▎   | 218/345 [00:28<00:15,  8.07it/s]#015 63%|██████▎   | 219/345 [00:29<00:15,  8.08it/s]#015 64%|██████▍   | 220/345 [00:29<00:15,  8.01it/s]#015 64%|██████▍   | 221/345 [00:29<00:15,  8.02it/s]#015 64%|██████▍   | 222/345 [00:29<00:15,  8.04it/s]#015 65%|██████▍   | 223/345 [00:29<00:15,  8.04it/s]#015 65%|██████▍   | 224/345 [00:29<00:15,  8.05it/s]#015 65%|██████▌   | 225/345 [00:29<00:14,  8.05it/s]#015 66%|██████▌   | 226/345 [00:29<00:14,  8.04it/s]#015 66%|██████▌   | 227/345 [00:30<00:14,  8.04it/s]#015 66%|██████▌   | 228/345 [00:30<00:14,  8.03it/s]#015 66%|██████▋   | 229/345 [00:30<00:14,  7.98it/s]#015 67%|██████▋   | 231/345 [00:30<00:13,  8.38it/s]#015 67%|██████▋   | 232/345 [00:30<00:13,  8.27it/s]#015 68%|██████▊   | 233/345 [00:30<00:13,  8.20it/s]#015 68%|██████▊   | 234/345 [00:30<00:13,  8.15it/s]#015 68%|██████▊   | 235/345 [00:30<00:13,  8.11it/s]#015 68%|██████▊   | 236/345 [00:31<00:13,  8.09it/s]#015 69%|██████▊   | 237/345 [00:31<00:13,  8.07it/s]#015 69%|██████▉   | 238/345 [00:31<00:13,  8.07it/s]#015 69%|██████▉   | 239/345 [00:31<00:13,  8.06it/s]#015 70%|██████▉   | 240/345 [00:31<00:13,  8.05it/s]#015 70%|██████▉   | 241/345 [00:31<00:12,  8.06it/s]#015 70%|███████   | 242/345 [00:31<00:12,  8.05it/s]#015 70%|███████   | 243/345 [00:31<00:12,  8.05it/s]#015 71%|███████   | 244/345 [00:32<00:12,  8.05it/s]#015 71%|███████   | 245/345 [00:32<00:12,  8.05it/s]#015 71%|███████▏  | 246/345 [00:32<00:12,  8.04it/s]#015 72%|███████▏  | 247/345 [00:32<00:12,  8.04it/s]#015 72%|███████▏  | 248/345 [00:32<00:12,  8.04it/s]#015 72%|███████▏  | 249/345 [00:32<00:11,  8.04it/s]#015 72%|███████▏  | 250/345 [00:32<00:11,  8.03it/s]#015 73%|███████▎  | 251/345 [00:32<00:11,  8.04it/s]#015 73%|███████▎  | 252/345 [00:33<00:11,  8.04it/s]#015 73%|███████▎  | 253/345 [00:33<00:11,  8.05it/s]#015 74%|███████▎  | 254/345 [00:33<00:11,  8.05it/s]#015 74%|███████▍  | 255/345 [00:33<00:11,  8.05it/s]#015 74%|███████▍  | 256/345 [00:33<00:11,  8.05it/s]#015 74%|███████▍  | 257/345 [00:33<00:10,  8.05it/s]#015 75%|███████▍  | 258/345 [00:33<00:10,  8.05it/s]#015 75%|███████▌  | 259/345 [00:33<00:10,  8.04it/s]#015 75%|███████▌  | 260/345 [00:34<00:10,  8.04it/s]#015 76%|███████▌  | 261/345 [00:34<00:10,  7.98it/s]#015 76%|███████▌  | 262/345 [00:34<00:10,  7.99it/s]#015 76%|███████▌  | 263/345 [00:34<00:10,  8.00it/s]#015 77%|███████▋  | 264/345 [00:34<00:10,  8.01it/s]#015 77%|███████▋  | 265/345 [00:34<00:10,  7.91it/s]#015 77%|███████▋  | 266/345 [00:34<00:10,  7.88it/s]#015 77%|███████▋  | 267/345 [00:34<00:09,  7.94it/s]#015 78%|███████▊  | 268/345 [00:35<00:09,  7.99it/s]#015 78%|███████▊  | 269/345 [00:35<00:09,  7.96it/s]#015 78%|███████▊  | 270/345 [00:35<00:09,  8.00it/s]#015 79%|███████▊  | 271/345 [00:35<00:09,  8.02it/s]#015 79%|███████▉  | 272/345 [00:35<00:09,  8.03it/s]#015 79%|███████▉  | 273/345 [00:35<00:08,  8.05it/s]#015 79%|███████▉  | 274/345 [00:35<00:08,  8.07it/s]#015 80%|███████▉  | 275/345 [00:35<00:08,  8.09it/s]#015 80%|████████  | 276/345 [00:36<00:08,  8.11it/s]#015 80%|████████  | 277/345 [00:36<00:08,  8.11it/s]#015 81%|████████  | 278/345 [00:36<00:08,  8.09it/s]#015 81%|████████  | 279/345 [00:36<00:08,  8.10it/s]#015 81%|████████  | 280/345 [00:36<00:08,  8.09it/s]#015 81%|████████▏ | 281/345 [00:36<00:07,  8.09it/s]#015 82%|████
████▏ | 282/345 [00:36<00:07,  8.09it/s]#015 82%|████████▏ | 283/345 [00:36<00:07,  8.10it/s]#015 82%|████████▏ | 284/345 [00:37<00:07,  8.11it/s]#015 83%|████████▎ | 285/345 [00:37<00:07,  8.11it/s]#015 83%|████████▎ | 286/345 [00:37<00:07,  8.11it/s]#015 83%|████████▎ | 287/345 [00:37<00:07,  8.12it/s]#015 83%|████████▎ | 288/345 [00:37<00:07,  8.11it/s]#015 84%|████████▍ | 289/345 [00:37<00:06,  8.11it/s]#015 84%|████████▍ | 290/345 [00:37<00:06,  8.12it/s]#015 84%|████████▍ | 291/345 [00:37<00:06,  8.11it/s]#015 85%|████████▍ | 292/345 [00:38<00:06,  8.11it/s]#015 85%|████████▍ | 293/345 [00:38<00:06,  8.12it/s]#015 85%|████████▌ | 294/345 [00:38<00:06,  8.10it/s]#015 86%|████████▌ | 295/345 [00:38<00:06,  8.10it/s]#015 86%|████████▌ | 296/345 [00:38<00:06,  8.10it/s]#015 86%|████████▌ | 297/345 [00:38<00:05,  8.11it/s]#015 86%|████████▋ | 298/345 [00:38<00:05,  8.12it/s]#015 87%|████████▋ | 299/345 [00:38<00:05,  8.11it/s]#015 87%|████████▋ | 300/345 [00:39<00:05,  8.11it/s]#015 87%|████████▋ | 301/345 [00:39<00:05,  8.11it/s]#015 88%|████████▊ | 302/345 [00:39<00:05,  8.09it/s]#015 88%|████████▊ | 303/345 [00:39<00:05,  7.98it/s]#015 88%|████████▊ | 304/345 [00:39<00:05,  8.01it/s]#015 88%|████████▊ | 305/345 [00:39<00:04,  8.04it/s]#015 89%|████████▊ | 306/345 [00:39<00:04,  7.92it/s]#015 89%|████████▉ | 307/345 [00:39<00:04,  7.97it/s]#015 89%|████████▉ | 308/345 [00:40<00:04,  8.00it/s]#015 90%|████████▉ | 309/345 [00:40<00:04,  8.03it/s]#015 90%|████████▉ | 310/345 [00:40<00:04,  8.04it/s]#015 90%|█████████ | 311/345 [00:40<00:04,  8.05it/s]#015 90%|█████████ | 312/345 [00:40<00:04,  8.05it/s]#015 91%|█████████ | 313/345 [00:40<00:04,  7.98it/s]#015 91%|█████████ | 314/345 [00:40<00:03,  8.01it/s]#015 91%|█████████▏| 315/345 [00:40<00:03,  8.02it/s]#015 92%|█████████▏| 316/345 [00:41<00:03,  8.04it/s]#015 92%|█████████▏| 317/345 [00:41<00:03,  8.05it/s]#015 92%|█████████▏| 318/345 [00:41<00:03,  8.00it/s]#015 92%|█████████▏| 319/345 [00:41<00:03,  8.03it/s]#015 93%|█████████▎| 320/345 [00:41<00:03,  8.04it/s]#015 93%|█████████▎| 321/345 [00:41<00:02,  8.06it/s]#015 93%|█████████▎| 322/345 [00:41<00:02,  8.07it/s]#015 94%|█████████▎| 323/345 [00:41<00:02,  8.05it/s]#015 94%|█████████▍| 324/345 [00:42<00:02,  8.06it/s]#015 94%|█████████▍| 325/345 [00:42<00:02,  8.08it/s]#015 94%|█████████▍| 326/345 [00:42<00:02,  8.07it/s]#015 95%|█████████▍| 327/345 [00:42<00:02,  8.03it/s]#015 95%|█████████▌| 328/345 [00:42<00:02,  8.05it/s]#015 95%|█████████▌| 329/345 [00:42<00:01,  8.07it/s]#015 96%|█████████▌| 330/345 [00:42<00:01,  8.09it/s]#015 96%|█████████▌| 331/345 [00:42<00:01,  8.09it/s]#015 96%|█████████▌| 332/345 [00:43<00:01,  8.09it/s]#015 97%|█████████▋| 333/345 [00:43<00:01,  8.09it/s]#015 97%|█████████▋| 334/345 [00:43<00:01,  8.10it/s]#015 97%|█████████▋| 335/345 [00:43<00:01,  8.05it/s]#015 97%|█████████▋| 336/345 [00:43<00:01,  8.03it/s]#015 98%|█████████▊| 337/345 [00:43<00:00,  8.03it/s]#015 98%|█████████▊| 338/345 [00:43<00:00,  8.04it/s]#015 98%|█████████▊| 339/345 [00:43<00:00,  8.04it/s]#015 99%|█████████▊| 340/345 [00:44<00:00,  8.04it/s]#015 99%|█████████▉| 341/345 [00:44<00:00,  8.04it/s]#015 99%|█████████▉| 342/345 [00:44<00:00,  8.02it/s]#015 99%|█████████▉| 343/345 [00:44<00:00,  8.01it/s]#015100%|█████████▉| 344/345 [00:44<00:00,  8.01it/s][INFO|trainer.py:862] 2020-12-31 08:29:28,297 >>

Training completed. Do not forget to share your model on huggingface.co/models =)


#015                                                 #015#015100%|██████████| 345/345 [00:44<00:00,  8.01it/s]#015100%|██████████| 345/345 [00:44<00:00,  7.73it/s]
[INFO|trainer.py:1226] 2020-12-31 08:29:28,298 >> Saving model checkpoint to /opt/ml/model
[INFO|configuration_utils.py:289] 2020-12-31 08:29:28,300 >> Configuration saved in /opt/ml/model/config.json
[INFO|modeling_utils.py:814] 2020-12-31 08:29:28,950 >> Model weights saved in /opt/ml/model/pytorch_model.bin
12/31/2020 08:29:28 - INFO - __main__ -   ***** Train results *****
12/31/2020 08:29:28 - INFO - __main__ -     global_step = 345
12/31/2020 08:29:28 - INFO - __main__ -     training_loss = 0.4789575106855752
12/31/2020 08:29:28 - INFO - __main__ -   *** Evaluate ***
[INFO|trainer.py:388] 2020-12-31 08:29:28,986 >> The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: sentence2, idx, sentence1.
[INFO|trainer.py:1412] 2020-12-31 08:29:28,987 >> ***** Running Evaluation *****
[INFO|trainer.py:1413] 2020-12-31 08:29:28,987 >>   Num examples = 408
[INFO|trainer.py:1414] 2020-12-31 08:29:28,987 >>   Batch size = 8
#015  0%|          | 0/51 [00:00<?, ?it/s]#015 18%|█▊        | 9/51 [00:00<00:00, 80.14it/s]#015 33%|███▎      | 17/51 [00:00<00:00, 77.98it/s]#015 49%|████▉     | 25/51 [00:00<00:00, 76.58it/s]#015 65%|██████▍   | 33/51 [00:00<00:00, 75.53it/s]#015 80%|████████  | 41/51 [00:00<00:00, 74.76it/s]#015 96%|█████████▌| 49/51 [00:00<00:00, 74.40it/s]12/31/2020 08:29:29 - INFO - /opt/conda/lib/python3.6/site-packages/datasets/metric.py -   Removing /root/.cache/huggingface/metrics/glue/mrpc/default_experiment-1-0.arrow
#015100%|██████████| 51/51 [00:00<00:00, 72.39it/s]
12/31/2020 08:29:29 - INFO - __main__ -   ***** Eval results mrpc *****
12/31/2020 08:29:29 - INFO - __main__ -     epoch = 3.0
12/31/2020 08:29:29 - INFO - __main__ -     eval_accuracy = 0.7892156862745098
12/31/2020 08:29:29 - INFO - __main__ -     eval_combined_score = 0.8183667083854819
12/31/2020 08:29:29 - INFO - __main__ -     eval_f1 = 0.847517730496454
12/31/2020 08:29:29 - INFO - __main__ -     eval_loss = 0.4569968283176422


2020-12-31 08:29:40 Uploading - Uploading generated training model
2020-12-31 08:30:16 Completed - Training job completed
Training seconds: 357
Billable seconds: 357
```

For local testing you can ran this script. It will add all the required Sagemaker environment variables to the script.

```bash
export TASK_NAME=mrpc
export SM_CHANNELS=["test","train"]
export SM_OUTPUT_DATA_DIR=/opt/ml/output/data
export SM_MODEL_DIR=/opt/ml/model
export M_CHANNEL_TEST=/opt/ml/input/data/test
export SM_CHANNEL_TRAIN=/opt/ml/input/data/train


  python ../../transformers/examples/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train True \
  --do_eval True \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
```

I would love to receive suggestions for improvement.
If it looks okay for you I would move the `is_run_on_sagemaker()` to the correct path and we could merge it.

P.S. i also added a fix for the `train_result.metrics` https://discuss.huggingface.co/t/attributeerror-trainoutput-object-has-no-attribute-metrics-when-finetune-custom-dataset/2970
