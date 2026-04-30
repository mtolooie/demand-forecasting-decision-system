from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

#load dataset
df = pd.read_csv('../data/train.csv')

#Preprocess

#filter data to business
df = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')]
#clean data
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df[['date', 'sales']]
df = df.dropna()

# make dates time_based
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
#df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek

df['lag_7'] = df['sales'].shift(7)
df['lag_14'] = df['sales'].shift(14)
df['lag_30'] = df['sales'].shift(30)

df = df.dropna()

#print(df.columns)

#separating features and target
x = df[['day', 'month', 'day_of_week', 'lag_7', 'lag_14', 'lag_30']]
y = df['sales']

# Train/Test split data without shuffling
split_index = int(len(df) * 0.8)

x_train = x.iloc[:split_index]
x_test = x.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Training model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(x_train, y_train)

#Make predictions
y_pred = model.predict(x_test)

#Evaluating prediction accuracy
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
print("average sales:", y_test.mean())

# Building sales thresholds for decision system
low_threshold = df['sales'].quantile(0.33)
high_threshold = df['sales'].quantile(0.66)

# Decision Function
def make_decision(value):
    if value < high_threshold:
        return "maintain stock"
    elif value < low_threshold:
        return "reduce stock"
    else:
        return "increase stock"

decisions = [make_decision(p) for p in y_pred]

#create correct decisions for real values
true_decisions = [make_decision(v) for v in y_test]

#compare decisions for accuracy
correct = 0
for pred, true in zip(decisions, true_decisions):
    if pred == true:
        correct += 1

decision_accuracy = correct / len(true_decisions)
print("decision_accuracy", decision_accuracy)
