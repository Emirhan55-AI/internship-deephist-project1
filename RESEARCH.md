# Project Literature Review: Object Detection and Tracking

This document contains the results and decisions of the literature review conducted to determine the object detection and multi-object tracking models to be used within the scope of the project.

## 1. Object Detection

This section presents the literature review and results conducted to determine the object detection model to be used in the project.

### Reviewed Paper 1

**Reference:** Murat, Ayşe Aybilge, and Mustafa Servet Kiran. "A comprehensive review on YOLO versions for object detection." *Engineering Science and Technology, an International Journal* 70 (2025): 102161.

**Content of the Paper:**
This paper examines all major versions of the YOLO algorithm since its first publication. The study systematically analyzes the architectural differences between YOLO versions, as well as the strengths, weaknesses, and their contributions to performance [1].

**Findings of the Paper:**
The findings obtained from this paper formed the rationale for preferring YOLO within the scope of the current project. The paper also discusses the differences between **one-stage** and **two-stage** object detection systems. Considering the project’s real-time speed requirements and its potential to run on embedded systems, it was concluded that one-stage detection systems, which offer higher speed, are more suitable. In this context, the **YOLO algorithm**, one of the strongest among one-stage detection systems, was chosen as the foundation for this project.

### Reviewed Paper 2

**Reference:** Jegham, Nidhal, et al. "Evaluating the evolution of YOLO (You Only Look Once) models: A comprehensive benchmark study of YOLO11 and its predecessors." *arXiv e-prints* (2024): arXiv-2411.

**Content of the Paper:**
This paper is the first study to comprehensively evaluate all YOLO algorithm versions from YOLOv3 to YOLOv11. The study uses three datasets with different levels of difficulty and considers key metrics such as accuracy (mAP), speed (inference time), computational complexity (GFLOPs), and model size. According to the findings, the **YOLOv11 family** provides the most balanced results in terms of accuracy, speed, and efficiency compared to previous versions. In particular, the **YOLO11m** and **YOLO11s** models stand out with their high accuracy and low latency [2].

**Findings of the Paper and Model Selection:**
The findings obtained from this paper served as the main foundation for the model selection in the project. In a scenario such as a classroom environment—which can potentially be crowded and contain small objects (such as distant student heads)—the balance between speed and accuracy is of critical importance.

**Reasons for Eliminating Other Models:**

* **YOLOv3/v4:** Inefficient.
* **YOLOv5/v8:** Do not reach the efficiency level of YOLOv11.
* **YOLOv9:** Slow.
* **YOLOv10:** May experience accuracy issues in crowded environments.

### Selected Model: YOLOv11s

> This model has been identified as the **most suitable model** for projects requiring real-time operation on edge devices or standard hardware—such as student detection, counting, and tracking in a classroom—due to its small size, low latency (low GFLOPs), and sufficient accuracy.

---

## 3. Re-Identification (ReID) Model Selection

This section presents the literature review conducted to determine the ReID model for appearance-based feature extraction in multi-object tracking.

### Reviewed Paper 1

**Reference:** Zhou, Kaiyang, et al. "Omni-scale feature learning for person re-identification." *Proceedings of the IEEE/CVF international conference on computer vision*. 2019.

**Content of the Paper:**
This paper introduces OSNet (Omni-Scale Network), a lightweight deep learning architecture specifically designed for person re-identification. OSNet uses a unified aggregation gate to dynamically fuse multi-scale features, making it highly efficient for real-time applications while maintaining competitive accuracy [3].

**Findings of the Paper:**
The findings obtained from this paper formed the rationale for preferring OSNet within the scope of the current project. OSNet's architecture offers several key advantages for real-time tracking applications:

- **Lightweight Architecture:** OSNet achieves state-of-the-art performance with significantly fewer parameters compared to traditional ReID models (ResNet-50, PCB)
- **Omni-Scale Feature Learning:** The unified aggregation gate allows the model to capture features at multiple scales simultaneously, improving robustness to scale variations
- **Real-Time Performance:** OSNet's efficiency makes it suitable for real-time tracking applications without compromising accuracy
- **Generalization:** OSNet demonstrates strong generalization across different datasets and scenarios

### Reviewed Paper 2

**Reference:** Sun, Yifan, et al. "Beyond part models: Person retrieval with refined part pooling (and a strong convolutional baseline)." *Proceedings of the European conference on computer vision (ECCV)*. 2018.

**Content of the Paper:**
This paper introduces PCB (Part-based Convolutional Baseline), a part-based approach for person re-identification that divides the input image into horizontal parts and extracts features from each part separately [4].

**Findings of the Paper and Model Selection:**
While PCB achieves competitive accuracy, the study revealed several limitations for real-time tracking applications:

- **Computational Overhead:** PCB requires processing multiple parts separately, increasing inference time
- **Part Alignment:** The part-based approach requires additional preprocessing and alignment steps
- **Memory Requirements:** Storing features for multiple parts increases memory consumption

### Reviewed Paper 3

**Reference:** He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

**Content of the Paper:**
This paper introduces ResNet (Residual Network), a deep convolutional neural network architecture that uses residual connections to enable training of very deep networks [5].

**Findings of the Paper:**
ResNet-50, while widely used in ReID applications, presents several challenges for real-time tracking:

- **Heavy Architecture:** ResNet-50 contains 25.6M parameters, making it computationally expensive
- **Inference Time:** The deep architecture results in slower inference times, unsuitable for real-time applications
- **Resource Requirements:** High memory and computational requirements limit deployment on edge devices

### Selected Model: OSNet (Omni-Scale Network)

> This model has been identified as the **most suitable model** for real-time person re-identification in multi-object tracking applications due to its lightweight architecture, competitive accuracy, and efficient inference time. OSNet provides the optimal balance between performance and computational efficiency required for real-time student tracking in classroom environments.

**Key Advantages for Our Project:**
- **Real-Time Performance:** OSNet achieves inference times suitable for 30 FPS processing
- **Memory Efficiency:** Lower parameter count enables deployment on standard hardware
- **Accuracy:** Competitive performance with heavier models (ResNet-50, PCB)
- **Robustness:** Omni-scale features improve handling of scale variations and occlusions

**Model Specifications:**
- **Architecture:** OSNet (x0.25 variant for optimal speed-accuracy trade-off)
- **Parameters:** ~0.68M (compared to 25.6M in ResNet-50)
- **Input Size:** 256×128 pixels
- **Feature Dimension:** 512
- **Inference Time:** ~2-3ms per image (on CPU)

---

## 4. Multi-Object Tracking

This section presents the literature review conducted to determine the methodology to be used for assigning persistent identities to detected students and tracking them.

### IMPORTANT: Algorithm Selection Update

#### Why We Switched from ByteTrack to BoT-SORT

After the initial literature review, **ByteTrack** was selected as the tracking algorithm. However, during the implementation and testing phase, we switched to **BoT-SORT** (Boosting Online Tracking with SORT) based on empirical performance data and technical requirements. This section explains the rationale behind this decision.

#### Performance Comparison (BoxMOT Evolution Trials)

Based on 200 evolution trials conducted by the BoxMOT framework on each tracker, the following performance metrics were obtained:

| Metric         | BoT-SORT | ByteTrack | Improvement     |
| -------------- | -------- | --------- | --------------- |
| **HOTA** | 70.074   | 67.87     | +2.2%           |
| **MOTA** | 78.113   | 77.784    | +0.4%           |
| **IDF1** | 82.869   | 79.691    | **+3.2%** |

**Source:** [BoxMOT Best Results Wiki](https://github.com/mikel-brostrom/boxmot/wiki/Best-results-after-200-evolution-trials-on-each-tracker)

#### Key Technical Advantages of BoT-SORT

1. **Superior Identity Consistency (IDF1)**

   - BoT-SORT achieves **82.869 IDF1** compared to ByteTrack's 79.691
   - IDF1 is the most critical metric for our project goal: "assigning persistent digital identities to students"
   - Higher IDF1 means fewer **Identity Switches (ID Switches)**, which directly impacts tracking quality in classroom scenarios
2. **Integrated ReID (Re-Identification)**

   - BoT-SORT natively integrates **OSNet** for appearance-based feature extraction
   - Uses ReID embeddings to maintain identity consistency across occlusions
   - Better handles scenarios where students temporarily leave the frame or are occluded by others
   - Combines **motion (Kalman filtering)** and **appearance (ReID)** for robust association
3. **Camera Motion Compensation (CMC)**

   - BoT-SORT includes **ECC (Enhanced Correlation Coefficient)** for camera motion compensation
   - Handles camera shake or movement common in classroom surveillance systems
   - Improves tracking stability in real-world deployment scenarios
4. **Better Handling of Occlusions**

   - The combination of Kalman prediction, IoU matching, and ReID features makes BoT-SORT more robust to occlusions
   - Critical for classroom environments where students frequently overlap or move behind desks
5. **Optimized for Real-World Scenarios**

   - BoT-SORT's design specifically addresses challenges in crowded environments
   - Better suited for scenarios with similar-looking objects (students in uniforms)
   - Maintains tracking quality even with detection gaps

#### Why ByteTrack Was Initially Considered

ByteTrack was initially selected based on the literature review because:

- It's a state-of-the-art TBD (Tracking-by-Detection) algorithm
- It achieves excellent speed-accuracy balance
- It's widely used and well-documented
- The literature indicated it was suitable for real-time applications

#### Why We Switched to BoT-SORT

The switch to BoT-SORT was made after:

1. **Empirical Testing:** Performance comparison showed BoT-SORT's superiority in identity consistency
2. **Technical Requirements:** The need for robust ReID integration for classroom tracking
3. **Real-World Scenarios:** Better handling of occlusions and similar appearances
4. **Future-Proofing:** BoT-SORT's architecture allows for easier parameter tuning and optimization

#### Conclusion

The switch from ByteTrack to BoT-SORT was driven by **empirical performance data** and **technical requirements** specific to classroom tracking. While ByteTrack remains an excellent choice for general-purpose tracking, BoT-SORT's superior identity consistency (IDF1), integrated ReID capabilities, and camera motion compensation make it the optimal choice for our use case.

### Reviewed Paper 1

**Reference:** Luo, Wenhan, et al. "Multiple object tracking: A literature review." *Artificial Intelligence* 293 (2021): 103448.

**Content of the Paper:**
This paper provides a fundamental introduction to the MOT field by comprehensively explaining the problem (persistent ID assignment), main challenges (occlusion, similar appearances), system components (detection, feature extraction, data association), and main paradigms (**Online/Offline**, **TBD/JDT**) [6].

**Findings of the Paper:**
The conceptual framework provided by this paper perfectly aligns with the requirements of the project.

* Since the project must operate in real time, the **Online** tracking approach is mandatory.
* As it offers the flexibility to develop and optimize the detection and tracking modules separately, the **Tracking-by-Detection (TBD)** paradigm is the most logical starting point in terms of integration convenience with the selected YOLOv11s model.

### Reviewed Paper 2

**Reference:** Adžemović, Momir. "Deep Learning-Based Multi-Object Tracking: A Comprehensive Survey from Foundations to State-of-the-Art." *arXiv preprint arXiv:2506.13457* (2025).

**Content of the Paper:**
This highly up-to-date paper addresses the MOT field purely from a Deep Learning perspective. It thoroughly examines the evolution of the TBD paradigm (SORT -> DeepSORT -> Transformer-based) and the modern types of the JDT paradigms (Embedding-based -> Query-based). It especially emphasizes the importance of new evaluation metrics such as **HOTA**, and the accuracy of Transformer-based end-to-end trackers (e.g., TrackFormer). It compares the state-of-the-art (SOTA) methods in terms of speed and accuracy [7].

**Findings of the Paper:**
This paper continues from where Luo's paper left off and enables the selection of the most modern and effective tracking algorithm for our project.

* **TBD Paradigms Remain Strong:** The study shows that the TBD approach is not outdated; on the contrary, it still achieves state-of-the-art results in both speed and accuracy with new algorithms such as **ByteTrack** and **BoT-SORT**. Consequently, the **ByteTrack** algorithm was chosen.
* **Speed–Accuracy Balance:** Although Transformer-based trackers offer the highest accuracy, they are generally too slow for real-time performance. Considering the speed requirements of our project, these models are not a practical solution.
* **Identity Consistency Is Critical:** The paper reveals that metrics such as **IDF1** and **HOTA** are more meaningful than MOTA for the project's goal of assigning "persistent digital identities." In particular, minimizing the number of **Identity Switches (ID Switch)** is critical.
