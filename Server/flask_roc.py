
# coding: utf-8

# In[2]:


from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

class ROC(Resource):
    def get(self, preprocessing, c):
        # you need to preprocess the data according to user preferences (only fit preprocessing on train data)
        # fit the model on the training set
        # predict probabilities on test set
        if preprocessing=='min-max':
            scaler = MinMaxScaler()
        elif preprocessing=='standardization':
            scaler = StandardScaler()
        else:
            return {'error: no scallar selected':'choose min-max or standardization for processing'}
        scaler.fit(X_train)
        Scaled_Xtrain = scaler.transform(X_train)
        Scaled_Xtest = scaler.transform(X_test)
        
        #model
        LR_Model=LogisticRegression(C=c).fit(Scaled_Xtrain,y_train)
        y_probabilties = LR_Model.predict_proba(Scaled_Xtest)
        
        #Storing the fpr, tpr, threshold values and  returning the result
        fpr, tpr, threshld = roc_curve(y_true=y_test,y_score=y_probabilties[:,1])
        
        result = [{'fpr':fpr[i].item(),'tpr':tpr[i].item(),'threshold':threshld[i].item()} for i in range(len(threshld))]
        return result

# Here you need to add the ROC resource, ex: api.add_resource(HelloWorld, '/')
# for examples see 
# https://flask-restful.readthedocs.io/en/latest/quickstart.html#a-minimal-api
print("adding resource")
api.add_resource(ROC,'/<string:preprocessing>/<float:c>')

if __name__ == '__main__':
    # load data
    df = pd.read_csv('data/transfusion.data')
    xDf = df.loc[:, df.columns != 'Donated']
    y = df['Donated']
    # get random numbers to split into train and test
    np.random.seed(1)
    r = np.random.rand(len(df))
    # split into train test
    X_train = xDf[r < 0.8]
    X_test = xDf[r >= 0.8]
    y_train = y[r < 0.8]
    y_test = y[r >= 0.8]
    app.run(debug=True)

