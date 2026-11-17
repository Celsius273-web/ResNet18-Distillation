# ResNet18 Knowledge Distillation on CIFAR-100

A ResNet18 teacher is distilled into a student with half the width (about 4x fewer parameters and FLOPs). The project measures what distillation buys over training the same student normally, using the same seeds, data, schedule and hyperparameters, and reports accuracy, agreement with the teacher, FLOPs, latency and memory.

This is a redo of an earlier CIFAR-10 version of this project. The earlier results were replaced by the ones below, which use CIFAR-100, three seeds per student and a fully trained teacher.

## Results

CIFAR-100 test set (10,000 images), 30 epochs, accuracy at the final epoch (no best-checkpoint selection). Student rows are mean ± sample standard deviation over 3 seeds (0, 1, 2).

| Model | Params | FLOPs (1 image) | Test accuracy |
|---|---|---|---|
| Teacher: ResNet18, width 64 | 11.22M | 1.111 G | 74.33% (one run) |
| Student, trained normally (baseline) | 2.82M | 0.279 G | 71.33 ± 0.52% |
| Student, distilled | 2.82M | 0.279 G | **73.70 ± 0.21%** |

- Distillation improves the student by **+2.37 points** over the same student trained normally.
- Every distilled run (73.46, 73.86, 73.77) beat every baseline run (71.35, 70.80, 71.84).
- The distilled student recovers about 79% of the 3.0-point gap between the baseline student and the teacher, with 3.98x fewer parameters and 3.99x fewer FLOPs.

**Agreement with the teacher's predictions on the test set**

| Seed | Baseline student | Distilled student |
|---|---|---|
| 0 | 76.33% | 80.70% |
| 1 | 75.92% | 80.68% |
| 2 | 76.96% | 80.64% |

**Inference cost, teacher vs student** (Tesla T4, fp32, 20 warm-up and 200 timed iterations, p50; seed-0 models)

| Batch size | Teacher | Student | Speedup | Peak memory (teacher / student) |
|---|---|---|---|---|
| 1 | 3.30 ms | 2.20 ms | 1.50x | 50.9 / 17.0 MB (3.00x) |
| 256 | 106.13 ms | 27.62 ms | 3.84x | 593.4 / 249.3 MB (2.38x) |

## Method

- **Models:** CIFAR-style ResNet18 (3x3 stem, no max-pool) written in PyTorch. The teacher uses base width 64 and the student base width 32.
- **Distillation loss:** `alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(softmax(teacher / T) || softmax(student / T))`, with T = 4 and alpha = 0.5. The teacher sees the same augmented batch as the student. The student learns from the teacher's full output distribution, including which wrong classes it considers close, not only the correct label. The KL term is scaled by T^2 to account for the softening effect of the temperature.
- **Training (identical for every run):** SGD, momentum 0.9, weight decay 5e-4, learning rate 0.1 with cosine annealing, batch size 128, 30 epochs, random crop with padding 4 and horizontal flip, fp16 mixed precision.
- **Teacher:** trained once with cross-entropy (seed 0) and frozen for all distillation runs.
- **Seeding:** each run is seeded before its model is built.
- **Test set use:** T and alpha were fixed in advance and not tuned on the test set. The test set is used only to report results.

## Limitations

- **No hyperparameter ablation.** T = 4 and alpha = 0.5 were fixed in advance and not tuned, so they are probably not optimal.
- **One teacher.** The teacher was trained once, so its run-to-run variance is not included in the student standard deviations.
- **Scope.** CIFAR-100 only, 30 epochs. Absolute accuracies are not comparable with published 200-epoch numbers; the comparison is only valid against the baseline trained here.
- **Other compression methods not compared.** INT8 quantization and other small architectures were not tried, so nothing here claims distillation beats them.
- **Precision mismatch.** Accuracy was evaluated under fp16 autocast, and latency was benchmarked in fp32.
- **Benchmark scope.** Latency and memory were measured on one GPU (T4) with the seed-0 models.

## Notes on how this was built

I wrote the data loading, the network and part of the training loop, and I worked through the distillation loss from Hinton et al., "Distilling the Knowledge in a Neural Network". I used AI assistance to write the FLOPs, latency and memory measurement code and to help with fp16 mixed precision and debugging.

## Files

- `ResnetDistill.ipynb`: the full notebook (data, models, training, evaluation, benchmarks).

## Reference

Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* 2015.
