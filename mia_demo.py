from mia.estimators import prepare_attack_data
from mia.estimators import AttackModelBundle
import numpy as np
import torch
import sklearn.datasets
import torch.nn as nn
import torch.nn.functional as F
from mia.estimators import ShadowModelBundle


#data
X,y = sklearn.datasets.make_moons(200,noise=0.2) 
X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)

#model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier,self).__init__()
        self.fc1 = nn.Linear(2,3)
        self.fc2 = nn.Linear(3,2)
        
    def forward(self,x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x
                
    def predict(self,x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
    

attack_model_fn = BinaryClassifier()
target_model_fn = BinaryClassifier
NUM_CLASSES = 2
target_model = BinaryClassifier()
cutoff = int(0.5 * len(y))
data_in = [X[cutoff:], y[cutoff:]]
data_out = [X[:cutoff], y[:cutoff]]

smb = ShadowModelBundle(
    target_model_fn,
    shadow_dataset_size=cutoff,
    num_models=2,
)
X_shadow, y_shadow = smb.fit_transform(X, y)

# amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)
# amb.fit(X_shadow, y_shadow)

# attack_test_data, real_membership_labels = prepare_attack_data(
#     target_model, data_in, data_out
# )

# attack_guesses = amb.predict(attack_test_data)
# attack_accuracy = np.mean(attack_guesses == real_membership_labels)