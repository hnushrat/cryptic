# cryptic

**\*\*\*\*\* Utility of each files \*\*\*\*\***

Several models have been trained for the task:
- **Auto Encoder**: This model aims at learning the representations of the binaries and then using them to classify.
- **LSTM**: This model uses the LSTM architecture, which takes each binary bit for processing and classifies.
- **Dense**: This is the vanilla neural network (ANN) for classification.



**How to use**: 
- Note:
  - The data file, "TrainingData.csv" needs to be in the same directory as the python files.


**For the Auto Encoder model:**
- If you want to retrain it, run "auto_enc.py". This will save the best-trained encoder model as "best_encoder.h5".
- Next, run "encoded_inputs_lstm.py" to train the classification model on the encoded inputs. The model will be saved as "encoded_inputs_model.h5"

**For the LSTM model:**
- Run "lstm_tf.py" to train the classification model. The saved model will be named "best_classifier_2.h5"

                precision    recall  f1-score   support
           0       0.50      0.61      0.55     24828
           1       0.50      0.38      0.44     25172
    accuracy        .         .        0.50     50000
   macro avg       0.50      0.50      0.49     50000
weighted avg       0.50      0.50      0.49     50000


**For the Dense model:**
- Run "dense_tf.py" to train the classification model. The saved model will be named "best_classifier_dense.h5"

**All of the trials could only achieve an accuracy of approximately 50%**

## Packages
- python(3.10.13)
- tensorflow(2.14.0)

## Contact
Please post a Github issue or contact (hnushrat_t@isical.ac.in) if you have any questions.
