# ResNet-18 Knowledge Distillation

PyTorch implementation of knowledge distillation for compressing a ResNet-18 image classifier on CIFAR-10.

A smaller student network learns from a larger teacher using ground-truth labels and teacher-generated soft targets. The resulting student model is approximately 4x smaller while achieving 87.89% test accuracy.

## Results

| Metric            |   Teacher |   Student |
| :---------------- | --------: | --------: |
| Model size        |  42.69 MB |  10.69 MB |
| Parameters        |    11.18M |     2.80M |
| Inference latency | 2.9549 ms | 2.6423 ms |
| Test accuracy     |         — |    87.89% |
| Size reduction    |         — |     3.99x |
| Speedup           |         — |     1.12x |

Latency was measured on an NVIDIA T4 GPU using a Google Colab runtime.

## Method

The student uses the same ResNet-18 residual architecture as the teacher with reduced channel widths. This reduces the parameter count and model size while preserving the overall residual network structure.

During training, the student receives two sources of supervision:

* Cross-entropy loss against CIFAR-10 ground-truth labels
* KL-divergence loss between the temperature-scaled teacher and student output distributions

The total loss combines both objectives using a temperature of `4.0` and a distillation weight of `0.5`.

```text
                    CIFAR-10 Image
                           |
              +------------+------------+
              |                         |
              v                         v
      Teacher ResNet-18        Student ResNet-18
        11.18M params            2.80M params
              |                         |
              |         Logits          |
              +------------+------------+
                           |
                  Distillation Loss
                           |
              +------------+------------+
              |                         |
              v                         v
       Cross-Entropy Loss        KL-Divergence Loss
              |                         |
              +------------+------------+
                           |
                           v
                    Student Update
```

## Architecture

The teacher uses ResNet-18 with channel widths of `64, 128, 256, 512` across its four residual stages.

The student uses reduced widths of `32, 64, 128, 256`.

Both models produce predictions for the 10 CIFAR-10 classes.

| Model   | Parameters | Model Size |
| :------ | ---------: | ---------: |
| Teacher |     11.18M |   42.69 MB |
| Student |      2.80M |   10.69 MB |

## Training

| Setting             | Value    |
| :------------------ | :------- |
| Dataset             | CIFAR-10 |
| Batch size          | 128      |
| Epochs              | 10       |
| Optimizer           | SGD      |
| Learning rate       | 0.15     |
| Momentum            | 0.9      |
| Weight decay        | 5e-4     |
| Scheduler           | StepLR   |
| Scheduler step      | 7 epochs |
| Scheduler gamma     | 0.1      |
| Temperature         | 4.0      |
| Distillation weight | 0.5      |

## Evaluation

The training script evaluates the student on the CIFAR-10 test set after each epoch.

The final recorded student accuracy is **87.89%**.

Inference latency uses GPU synchronization, 10 warm-up runs, and 100 measured runs.

| Benchmark       |    Result |
| :-------------- | --------: |
| Student latency | 2.6423 ms |
| Teacher latency | 2.9549 ms |
| Speedup         |     1.12x |
| Student size    |  10.69 MB |
| Teacher size    |  42.69 MB |
| Size reduction  |     3.99x |

The benchmark was run on an NVIDIA T4 GPU through Google Colab.

## Project Structure

ResNet18-Distillation/
├── resnet18distillation.py
├── Resnet18Distillation.ipynb
├── requirements.txt
├── README.md
└── LICENSE

`resnet18distillation.py` contains the training, distillation, evaluation, model-size calculation, and latency benchmarking code.

`Resnet18Distillation.ipynb` provides a notebook version of the experiment.

## Running the Project

Clone the repository:

```
git clone https://github.com/Celsius273-web/ResNet18-Distillation.git
cd ResNet18-Distillation
```

Install the dependencies:

```
pip install -r requirements.txt
```

Run the training and evaluation script:

```
python resnet18distillation.py
```

CIFAR-10 downloads automatically through `torchvision` if the dataset is not already available.

The script trains the student model, evaluates test accuracy, measures model size and inference latency, and saves the trained student weights.

For interactive execution, open `Resnet18Distillation.ipynb` in Jupyter or Google Colab.

## Reproducibility Notes

The reported benchmark was run on an NVIDIA T4 GPU using Google Colab.

Latency results depend on the hardware and benchmark configuration, so they should not be interpreted as hardware-independent performance measurements.

The experiment does not report a final CIFAR-10 accuracy for the teacher. Therefore, this README does not claim a specific accuracy difference between the teacher and student.

## Dependencies

The project uses PyTorch, torchvision, NumPy, Matplotlib, and scikit-learn. Package versions are specified in `requirements.txt`.

## License

MIT License
