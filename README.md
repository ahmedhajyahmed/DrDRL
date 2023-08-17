This repository contains the framework and data of the paper "An Intentional Forgetting-Driven Self-Healing Method For Deep Reinforcement Learning Systems" by Ahmed Haj Yahmed, Rached Bouchoucha,  Houssem Ben Braiek, and Foutse Khomh published in the technical research track of the 38th IEEE/ACM International Conference on Automated Software Engineering (ASE2023). 

# Introduction of the framework
In this repository, we introduce Dr.DRL, an automated self-healing approach that employs the mechanism of intentional forgetting to enhance the adaptability of DRL agents in response to environmental drifts. This approach not only enhances the plasticity of DRL agents when confronted with environments drift but also significantly reduces the requisite number of episodes necessary for an agent to be effectively fine tuned to a the drifted environment, as compared to conventional continual learning methods.

# Repository Structure
* dr_drl : contains all the source code of DrDRL including
       - agents : the code of the agents DQN, PPO, SAC
       - agents : the code of the environments Acrobot, Cartpole, Mountaincar
       - runners : contains the scripts to run the continual learning and Dr.DRL
* experimental_results : contains the xlsx files of our experiments for each pair of agent environment
* requirements.txt : contains the required packages versions

# License
The software we developed is distributed under MIT license. See the license file.

# Contacts
You can contact Ahme Haj Yahmed (ahmed-haj-yahmed@polymtl.ca) or Rached Bouchoucha (rached.bouchoucha@polymtl.ca) for any questions.
