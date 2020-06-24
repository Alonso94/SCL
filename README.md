## Abstract
State estimation is an essential part of any control system. A perception system estimates a representation of the states using sensors data. Recently, an increasing interest in exploiting machine learning techniques to learn state representation. Representation learning allows estimating states in real-world scenarios. We are interested in learning representation from RGB images extracted from videos. And use the state representation to compute cost/reward suitable for control and reinforcement learning.
We propose a method in which the user has to provide just a couple of videos demonstrating the task. Our approach uses a sequential contrastive loss to learn a latent space mapping, and descriptors of the task-related objects. 
Our framework serves robotics control scenarios, especially model-based reinforcement learning algorithms. The resulted representation eliminates the need for engineered reward functions or any explicit access to positioning systems, aiming to improve the applicability of learning to control physical systems. Our framework allows for reducing the learning time and working with low-resource scenarios.

### A paper to appear in the proceedings of [CAICS 2020](http://caics.ru)
Project webpage : [https://alonso94.github.io/SCL/](https://alonso94.github.io/SCL/)
### Training videos
<p align="center">
  <img src="https://raw.githubusercontent.com/Alonso94/SCL/master/videos/out2.gif" width=25%>
</p>
### a goal image
<p align="center">
  <img src="https://raw.githubusercontent.com/Alonso94/SCL/master/target.png" width=25%>
</p>
