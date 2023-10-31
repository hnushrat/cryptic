# cryptic

**\*\*\*\*\* Utility of each files \*\*\*\*\***

Several models have been trained for the task:
- **Auto Encoder**: This model aims at learning the representations of the binaries and then using them to classify.
- **LSTM**: This model uses the LSTM architecture, which takes each binary bit for processing and classifies.
- **Dense**: This is the vanilla neural network (ANN) for classification.
- **BERT**: This model tries to tackle the problem statement in a unique way by transforming bitstreams into tokens.


**How to use**: 
- Note:
  - The data file, "TrainingData.csv" needs to be in the same directory as the python files.
  - BERT model is in the "train_v2.py" and is written in pytorch.


**For the Auto Encoder model:**
- If you want to retrain it, run "auto_enc.py". This will save the best-trained encoder model as "best_encoder.h5".
- Next, run "encoded_inputs_lstm.py" to train the classification model on the encoded inputs. The model will be saved as "encoded_inputs_model.h5"

**For the LSTM model:**
- Run "lstm_tf.py" to train the classification model. The saved model will be named "best_classifier_2.h5"

**For the Dense model:**
- Run "dense_tf.py" to train the classification model. The saved model will be named "best_classifier_dense.h5"

**For the BERT model:**
- Run "train_v2.py" to train the classification model.

**All of the trials could only achieve an accuracy of approximately 50%**

## Packages
- python(3.8.18) for pytorch.
- python(3.10.13) for rest.
- torch(2.1.0+cu118)
- tensorflow(2.14.0)

## Contact
Please post a Github issue or contact (hnushrat@gmail.com) if you have any questions.
