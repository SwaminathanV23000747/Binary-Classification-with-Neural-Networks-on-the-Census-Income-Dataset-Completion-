# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset-Completion-


## Name: Swaminathan.V
## Reg no: 212223110057
## Overview
This project builds a binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually based on census data. The model uses embeddings for categorical features and batch-normalized continuous features.  

- **Dataset:** Census Income Dataset (income.csv)  
- **Input Features:** Categorical and continuous features such as Workclass, Education, Marital Status, Age, Hours per Week, etc.  
- **Output:** Binary label (<=50K or >50K)  
- **Model:** Single hidden layer neural network with embeddings for categorical features and batch normalization for continuous features.  
- **Training:** 300 epochs with Adam optimizer and CrossEntropyLoss.

---

### Program
```
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

torch.manual_seed(42)
data = pd.read_csv("/content/income.csv")

categorical_cols = data.select_dtypes(include='object').columns.tolist()
label_col = 'Income' # Corrected column name
categorical_cols = [c for c in categorical_cols if c != label_col]
continuous_cols = data.select_dtypes(include=['int64','float64']).columns.tolist()

label_enc = LabelEncoder()
data[label_col] = label_enc.fit_transform(data[label_col])

cat_encoders = {}
for c in categorical_cols:
    enc = LabelEncoder()
    data[c] = enc.fit_transform(data[c])
    cat_encoders[c] = enc

X_cats = data[categorical_cols].values
X_conts = data[continuous_cols].values
y = data[label_col].values

X_cat_train, X_cat_test, X_cont_train, X_cont_test, y_train, y_test = train_test_split(
    X_cats, X_conts, y, test_size=5000, random_state=42, stratify=y)

X_cat_train = torch.tensor(X_cat_train, dtype=torch.long)
X_cat_test = torch.tensor(X_cat_test, dtype=torch.long)
X_cont_train = torch.tensor(X_cont_train, dtype=torch.float)
X_cont_test = torch.tensor(X_cont_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

cat_sizes = [int(data[col].nunique()) for col in categorical_cols]
embeddings = [(size, min(50, (size+1)//2)) for size in cat_sizes]

class TabularModel(nn.Module):
    def __init__(self, emb_sizes, n_cont):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_sizes])
        self.emb_drop = nn.Dropout(0.4)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum([s for _, s in emb_sizes])
        self.fc1 = nn.Linear(n_emb + n_cont, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(50, 2)
    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat(x, dim=1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], dim=1)
        x = self.drop1(torch.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return x

model = TabularModel(embeddings, len(continuous_cols))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300
for epoch in range(epochs):
    model.train()
    y_pred = model(X_cat_train, X_cont_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    y_test_pred = model(X_cat_test, X_cont_test)
    test_loss = criterion(y_test_pred, y_test)
    y_pred_class = torch.argmax(y_test_pred, dim=1)
    acc = (y_pred_class == y_test).float().mean()

print(f"Test Loss: {test_loss.item():.4f}")
print(f"Accuracy: {acc.item()*100:.2f}%")

def predict_income(model, input_data):
    model.eval()
    cat_input = []
    cont_input = []
    for c in categorical_cols:
        val = input_data[c]
        val_enc = cat_encoders[c].transform([val])[0]
        cat_input.append(val_enc)
    for c in continuous_cols:
        cont_input.append(float(input_data[c]))
    cat_tensor = torch.tensor([cat_input], dtype=torch.long)
    cont_tensor = torch.tensor([cont_input], dtype=torch.float)
    with torch.no_grad():
        pred = model(cat_tensor, cont_tensor)
        prob = torch.softmax(pred, dim=1)[0][1].item()
        result = "Income >50K" if prob > 0.5 else "Income <=50K"
    return result, prob
```


### Output:

<img width="292" height="166" alt="image" src="https://github.com/user-attachments/assets/1aeadbee-ca23-4d92-a4cf-46319775ac67" />



### Result: 
Hence the program is completed successfully.
