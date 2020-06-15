## Abstract
The state of a physical system usually described by its po sition and movement relatively to its surroundings. This representation
is used widely in control and reinforcement learning to define the state and compute the cost/reward value. In real-world applications, such representations are available only in structured environments. State representation learning exploits the advances of deep learning to learn useful state representation from raw sensor data. To learn a model for a robotic manipulator using video data we are interested in representation learning from RGB images. We suggest a method for which we require the user to provide us with just a couple of videos demonstrating the task. Our approach uses a sequential contrastive loss to learn latent space mapping, and task-related descriptors in each state. Our framework intended to be used in robotics control scenarios, especially with model-based reinforcement learning algorithms. The resulted representation eliminates the need for engineered reward functions or any explicit access to positioning systems, aiming to improve the applicability of learning to control physical systems. Our framework emphasis reducing the learning time, and to work with low-resource scenarios.

### Paper to appear in the proceedings of [CAICS 2020](http://caics.ru)
[site](https://alonso94.github.io/SCL/)
### A video from training dataset
<p align="center">
  <img src="https://raw.githubusercontent.com/Alonso94/SCL/master/videos/out2.gif" width=25%>
</p>
