import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# import your classes
from  features import (
    NumericalHandeller,
    CategoricalHandeller,
    Encoder,
    ImputeStrategy,
    EncodingStrategy
)

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("titanic.csv")

# =========================
# 2. Drop useless columns
# =========================
df = df.drop(columns=["PassengerId", "Name"])

# =========================
# 3. Split features/target
# =========================
TARGET = "Survived"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Define handlers
# =========================

# Numerical columns
age_handler = NumericalHandeller(
    column="Age",
    impute_strategy=ImputeStrategy.MEAN
)

fare_handler = NumericalHandeller(
    column="Fare",
    impute_strategy=ImputeStrategy.MEDIAN
)

# Categorical columns
sex_handler = CategoricalHandeller(column="Sex")
embarked_handler = CategoricalHandeller(column="Embarked")


# =========================
# 5. Fit on TRAIN only
# =========================
age_handler.fit(X_train)
fare_handler.fit(X_train)
sex_handler.fit(X_train)
embarked_handler.fit(X_train)

# =========================
# 6. Transform TRAIN
# =========================
X_train = age_handler.transform(X_train)
X_train = fare_handler.transform(X_train)
X_train = sex_handler.transform(X_train)
X_train = embarked_handler.transform(X_train)

# =========================
# 7. Transform TEST
# =========================
X_test = age_handler.transform(X_test)
X_test = fare_handler.transform(X_test)
X_test = sex_handler.transform(X_test)
X_test = embarked_handler.transform(X_test)

# =========================
# 8. Encoding
# =========================

sex_encoder = Encoder("Sex", EncodingStrategy.LABEL)
embarked_encoder = Encoder("Embarked", EncodingStrategy.ONE_HOT)

sex_encoder.fit(X_train)
embarked_encoder.fit(X_train)

X_train = sex_encoder.transform(X_train)
X_train = embarked_encoder.transform(X_train)

X_test = sex_encoder.transform(X_test)
X_test = embarked_encoder.transform(X_test)

# =========================
# 9. Model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# 10. Predict + Evaluate
# =========================
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))