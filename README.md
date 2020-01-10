# path_integral_reinforcement_learning
# paper:A Path-Integral-Based Reinforcement Learning Algorithm for Path Following of an Autoassembly Mobile Robot
# DOI: 10.1109/TNNLS.2019.2955699
# https://ieeexplore.ieee.org/document/8941307
1. k_single.py: single training using PI2.
2. K1-K20.py: multiple propresses for all trainings so as for notable acceleration.
3. pi2_test.py: comparing with traditional nonlinear method only using Lyapunov techniques.

4. k_single_data.rar: results of single training
5. K_all.txt: all learned results using PI2 

6. crane_PI2_training.py: train the crane system.
7. crane_test.py: test the learned controller, comparing with traditional nonlinear method only using Lyapunov techniques.
8. crane_training_results.rar: results of the training of crane system.

9. PI2_path_following.m and MPC_path_following.m, comparison of real time performance.

10. pi2_tuning.py and pi_tuning_constant.py: comparing with constant P controller and tuning P controller.
