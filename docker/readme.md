building cpu docker

```bash
docker build -t pytorch-sagemaker . -f Dockerfile.cpu
```

```bash
    python src/main.py --buildspec pytorch/buildspec.yml \
                      --framework pytorch \
                      --image_types training \
                      --device_types gpu \
                      --py_versions py3
```

```bash
    python aws_deep_learning_container/src/main.py --buildspec aws_deep_learning_container/pytorch/buildspec.yml \
                      --framework pytorch \
                      --image_types training \
                      --device_types gpu \
                      --py_versions py3
```
