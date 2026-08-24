# ResNet18-Knowledge-Distillation
Knowledge distillation implementation compressing ResNet18 with the CIFAR-10 dataset. The student model achieves 4x size reduction (10.7MB vs 42.7MB) with 87.89% test accuracy. 

Knowledge distillation implementation demonstrating model compression and optimization tradeoffs. This project trains a lightweight student ResNet18 to mimic a larger teacher model on CIFAR-10.

## Results:

Student Model Latency: 2.6423 ms

Teacher Model Latency: 2.9549 ms

Speedup: 1.12x

Student Model Size: 10.69 MB

Teacher Model Size: 42.69 MB

Size Reduction: 3.99x

Final Test Accuracy: 87.89%


<img width="617" height="390" alt="image" src="https://github.com/user-attachments/assets/87fda41e-c3b3-449d-8fae-22cf39a871b2" />



<img width="625" height="390" alt="image" src="https://github.com/user-attachments/assets/62b90bad-a3a2-4a6d-8012-bc46dcf0cf9b" />


## How it works:
The distillation process combines two loss components. Regular cross entropy loss ensures the student learns the actual labels. KL divergence loss forces the student to match the teacher's probability distributions using temperature scaling. This allows the students to mimic the patterns teacher had learned not just what information the labels provide. 

## Methodology:
Train/validation splits, learning rate scheduling, batch normalization, and explicit latency profiling with GPU synchronization. Inference latency measured over 100 runs with GPU warm up to eliminate noise.

## Why this matters:
Fast inference speeds saves resources when in use. Distillation trades minimal accuracy loss for substantial practical benefits making it worth while for many systems.

## How to Run
Clone repo: git clone https://github.com/Celsius273-web/ResNet18-Knowledge-Distillation

Install dependencies: pip install -r requirements.txt

Download CIFAR-10 (automatic on first run)

Run training: python distillation.py

Training takes 3-5 minutes on GPU (Colab recommended)

Results printed to console: model size, latency, accuracy

Trained weights saved as student_distilled.pth

View plots: training loss and accuracy curves in output

Load saved model: torch.load('student_distilled.pth')

Optional: Run distillation.ipynb in Colab for interactive execution
