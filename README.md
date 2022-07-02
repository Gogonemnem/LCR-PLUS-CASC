# LCR+CASC
Code for Left-Center-Right Context-aware Aspect Category and Sentiment Classification. The code extends the code for CASC which can be found here: https://www.github.com/Raghu150999/UnsupervisedABSA
## Setup
Code is written in Python 3(.10) with all dependencies noted in requirements.txt. Pytorch and Tensorflow are the main packages, which require additional setup for optimal experience.

## Software explanation
As mentioned before, the code is based on source code written by others. Therefore, only new files are explained in this section. Furthermore, the code has been used as scripts. Thus, most of the code has to be run one by one to retrieve results.

The first part of the process is similar to the CASC procedure. Retrieve labels and scores for training data. However, our model uses a different method for scoring and labeling sentences. The rest of the process is quite different from the source code. First turn the SemEval .xml files into the same format of CASC source code. Afterwards, turn data into usable data. This step embeds all sentences using a version of DK-BERT. This step is required as Tensorflow and Pytorch are not compatible with each other. Afterwards the neural model can be trained (or hyperparameter optimized).

The following files are different from the source code:
- attention.py: The attention layers of HAABSA++.
- data.py: Loads the data in a usable format for the neural model.
- embedding.py: Embeds sentences using DK-BERT described in the paper. 
- example.py: An example how to use the code. The code also performs hyperparameter optimization.
- evaluate_test.py: Evaluates the performance of a model using SkLearn's classification report function.
- hypertrain.py: Reusable code to easily produce various hyperparameter optimized LCR-Rot-hop++ models.
- labeler_test.py: Our maximum score labeler function.
- lcr_rot_hop_plus_plus.py: Our double-task version of LCR-Rot-hop++.
- score_computer_test.py: Our maximum score function. This step also performs Aspect Term Extraction.
- semeval_reader.py: Data reader for the 2015 and 2016 restaurant SemEval datasets. Turns data into the same format used by the source code of CASC.
