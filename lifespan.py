LifeSpan Stage-III
Survival Prediction using Multimodal Features
Models: Cox PH, Random Survival Forest, DeepHit
Dataset: METABRIC (or similar survival dataset)
"""

# ======================================
# 1. IMPORTS
# ======================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ======================================
# 2. GLOBAL PARAMETERS
# ======================================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================
# 3. LOAD & PREPROCESS DATA
# ======================================
def load_survival_data(csv_path):
    """
    Expected columns:
    - time : survival time
    - event : 1=death, 0=censored
    - remaining columns : clinical + PatchSight + ImmuneMap features
    """
    df = pd.read_csv(csv_path)

    time = df["time"].values
    event = df["event"].values
    X = df.drop(columns=["time", "event"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, time, event, df


# ======================================
# 4. COX PROPORTIONAL HAZARDS
# ======================================
def train_cox_model(df):
    print("\nTraining Cox Proportional Hazards Model...")
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    return cph


# ======================================
# 5. RANDOM SURVIVAL FOREST
# ======================================
def train_rsf_model(X, time, event):
    print("\nTraining Random Survival Forest...")
    y = Surv.from_arrays(event=event.astype(bool), time=time)

    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1
    )
    rsf.fit(X, y)
    return rsf


# ======================================
# 6. DEEPHIT MODEL
# ======================================
class DeepHit(nn.Module):
    def __init__(self, input_dim):
        super(DeepHit, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_deephit(X, time, event):
    print("\nTraining DeepHit Model...")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    time_tensor = torch.tensor(time, dtype=torch.float32)
    event_tensor = torch.tensor(event, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, time_tensor, event_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DeepHit(X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for x, t, e in loader:
            x = x.to(DEVICE)
            t = t.to(DEVICE)
            e = e.to(DEVICE)

            pred = model(x).squeeze()
            loss = criterion(pred, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

    return model


# ======================================
# 7. RISK SCORE PREDICTION
# ======================================
def predict_risk_scores(cox, rsf, deephit, X):
    cox_risk = cox.predict_partial_hazard(pd.DataFrame(X))
    rsf_risk = rsf.predict(X)
    deephit.eval()

    with torch.no_grad():
        deephit_risk = deephit(
            torch.tensor(X, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    return cox_risk, rsf_risk, deephit_risk


# ======================================
# 8. MAIN EXECUTION
# ======================================
def main():
    print("Initializing LifeSpan Stage-III...")

    # Load data
    X, time, event, df = load_survival_data("data/metabric_survival.csv")

    # Split
    X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
        X, time, event, test_size=0.2, random_state=42
    )

    # Cox requires full dataframe
    df_train = pd.DataFrame(X_train)
    df_train["time"] = t_train
    df_train["event"] = e_train

    # Train models
    cox_model = train_cox_model(df_train)
    rsf_model = train_rsf_model(X_train, t_train, e_train)
    deephit_model = train_deephit(X_train, t_train, e_train)

    # Risk prediction
    cox_risk, rsf_risk, deephit_risk = predict_risk_scores(
        cox_model, rsf_model, deephit_model, X_test
    )

    print("\nSample Risk Scores:")
    print("Cox:", cox_risk[:5].values.flatten())
    print("RSF:", rsf_risk[:5])
    print("DeepHit:", deephit_risk[:5].flatten())

    torch.save(deephit_model.state_dict(), "deephit_lifespan.pth")
    print("\nDeepHit model saved as deephit_lifespan.pth")


if __name__ == "__main__":
    main()
