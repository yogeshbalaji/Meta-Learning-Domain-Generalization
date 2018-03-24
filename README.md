# Meta-Learning-Domain-Generalization

An attempt to replicate the paper "Learning to Generalize: Meta-Learning for Domain Generalization".

Prerequisites:
- Tensorflow (Code only tested for version r1.6)

Sourceonly folder contains the code for Deep-all model. MLDG folder contains the code for the meta learning approach.
Please download PACS dataset (http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)

To run the sourceonly model, go to source_only folder and run 

	python finetune.py

Please change the data root (the path where the PACS dataset is stored before running the code)

Models giving best validation accuracy will be stored in results folder (default checkpoint path). To evaluate the trained model on a new domain, run

	python eval.py

Again, the data root path has to be changed.


To run the MLDG code, go to MLDG folder and run

	python main.py

In the curent version of code, we are not able to replicate the performance of MLDG as reported in the paper. Contributions are welcome.
