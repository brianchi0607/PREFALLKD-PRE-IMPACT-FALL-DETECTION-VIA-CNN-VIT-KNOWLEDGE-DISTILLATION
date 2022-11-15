# PreFallKD:Pre-Impact Fall Detection via CNN-ViT Knowledge Distillation

# An overview of proposed PreFallKD Framework 
![PreFallKD](/images/PreFallKD_framework.png)
The input window is the 50 frames IMU signals, which include triaxial acceleration data, triaxial gyroscope data, and triaxial Euler angle data. The ViT-tiny is the teacher model and the lightweight CNN is the student model. The student model can learn the high dimension knowledge from the teacher model by Kullback-Leibler divergence loss function (KL Divergence Loss) and learn ground truth by Focal loss.
# Requirements
```
python == 3.9
pytorch >= 1.12.1
numpy >= 1.23.2
matplotlib >= 3.5.3
pandas >= 1.4.3
einops >= 0.4.1
``` 
