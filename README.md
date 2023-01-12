# Sarcasm_Detection

Design: For this project, I used the BERT ForSequenceClassification model. I used the pretrained model (which includes GeLU activation function in the intermediate layers, BinaryCrossEntropy for the loss function, and an additional linear layer at the end for the classification task), but adjusted the dropout probabilities for the attention layer and hidden layer to .23 to try to combat overfitting. I also tried to reduce the number of epochs to 2 to prevent overfitting. I used the Adam optimizer with a learning rate of 3e-5 (through trial and error). For the evaluation of the model, I chose to use the classification_report from sklearn.metrics, as ______________.

Dataset: I used the "News Headlines Dataset For Sarcasm Detection" from Kaggle. For preprocessing, I isolated the headlines and labels of the data. I did an 85:15 train:validation split, since I found that the model was overfitting and wanted to try decreasing the training size to account for it. For the testing set, I couldn't figure out how to get it from the training/validation set, but the Kaggle dataset provided two versions with different articles, so I took some of the second version to use as a test set. 

Results: 

Limitations: OVERFITTING! I attempted to fix this issue by increasing the probability of dropout, but I found that after .23, the model would just stop converging and keep an accuracy of 50% (and output all 0s). Another potential issue is with the dataset- all of the articles were either from the Huffington Post (nonsarcastic) or The Onion (sarcastic), meaning the articles probably all had some sort of similarity in topic or style.

Future Direction: Moving forward, I'd like to fix the overfitting issue. One way that can help is to successfully implement the data augmentation (based on replacing words with synonyms). I also would like to create a custom model to move the placement of the dropouts (and overall have more flexibility with the layers) and alter the loss functions to include more regularization.
