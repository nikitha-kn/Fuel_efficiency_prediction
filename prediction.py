import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.model_selection import train_test_split

df = pd.read_csv('ml/Fuel_Prediction/auto_mpg.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

print(df['horsepower'].unique())

df = df[df['horsepower'] != '?']
df['horsepower'] = df['horsepower'].astype(int)
df.isnull().sum()
print(df.nunique())


ndf = df.select_dtypes(include = ['number'])
plt.subplots(figsize = (12, 10))
for i ,col in enumerate(['cylinders','origin']):
    plt.subplot(1,2,i+1)
    sns.countplot(data = ndf, x = col)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize = (10,8))
sns.heatmap(ndf.corr(), annot = True, cmap = 'coolwarm',fmt = '.2f', linewidths = 0.5)
plt.title('Correlation between different features')
plt.show()

df.drop('displacement', axis = 1, inplace = True)

X = ndf.drop('mpg', axis = 1)
y = ndf['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape, X_test.shape)

AUTO = tf.data.experimental.AUTOTUNE

train_ds = (tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values)).batch(32).prefetch(AUTO))
test_ds = (tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values)).batch(32).prefetch(AUTO))

model = keras.Sequential([
    layers.Dense(64, activation = 'relu', input_shape = [7]),
    layers.BatchNormalization(),
    layers.Dense(32, activation = 'relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 'mse'])

print(model.summary())
history = model.fit(train_ds, epochs = 50, validation_data = test_ds)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize = (12, 6))
    plt.subplot(1,2,1)
    plt.plot(hist['epoch'], hist['mae'], label = 'Train MAE')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE over Epochs')

    plt.subplot(1,2,2)
    plt.plot(hist['epoch'], hist['mse'], label = 'Train MSE')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE over Epochs')

    plt.tight_layout()
    plt.show()

plot_history(history)