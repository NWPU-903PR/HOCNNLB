# HOCNNLB

HOCNNLB is a deep learning method used for predicting the RBP binding sites on lncRNA chains. HOCNNLB first employs a high-order one-hot encoding strategy to encode the lncRNA sequences by considering the dependence among nucleotides, then the encoded lncRNA sequences are fed into the convolutional neural network (CNN) to predict the RBP binding sites. We comprehensively evaluate HOCNNLB on 31 experimental datasets of 12 lncRNA binding proteins. 
=========================================================================================
HOCNNLB 2019/04/07 ver1.0.0  ICI/LIFT,China
=========================================================================================
System Requirements
The HOCNNLB is supported on Linux operating system
python 2.7
Keras 2.2.2 library and its backend is TensorFlow 
Sklearn

Content
./lncRBPdata.zip: the training and testing dataset with sequence and label indicating it is binding sites or not
./HOCNNLB.py: the python code, it can be ran to reproduce our results. 

Users's Guide
Input file requires fasta format

Execute Step
Step 1: Configure the hyperparameters(e.g. the filter number, filter size, pooling size and the neuron number of fully connected layer) and the name of RBP binding site dataset(e.g. 01_HITSCLIP_AGO2Karginov2013a_hg19).
Step 2: Run HOCNNLB,  train the HOCNNLB predictor using the training dataset and use the test dataset for independent testing.The final evaluation metrics are written into text formats (metrics_file.txt). 
Step 3: The final prediction results are presented in text formats (prediction.txt) and can be found in the out file directory.
where the input training file should be sequences.fa.gz with label info in each head per sequence.

Copyright Notice
Software, documentation and related materials: Copyright (c) 2019-2021 Institute of Control & Information(ICI), Northwestern Polytechnical University,China Key Laboratory of Information Fusion Technology(LIFT), Ministry of Education, China All rights reserved.
