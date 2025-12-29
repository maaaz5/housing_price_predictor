import pickle
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from app.models.regression import CustomLinearRegression




def load_and_prepare_data(csv_path):
  #loading the raw csv
  df = pd.read_csv(csv_path)

  df = df.dropna()

  df = pd.get_dummies(df,columns=['ocean_proximity'],prefix='ocean',dtype='int')

  y = df['median_house_value'].values.reshape(-1, 1)
  X = df.drop('median_house_value',axis=1).values

  feature_names = df.drop('median_house_value',axis=1).columns.to_list()

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)


  return X_scaled, y, feature_names, scaler




def train_model(csv_path, learning_rate=0.01, n_iterations=10000, model_save_path='models/housing_model.pkl', scaler_save_path='models/scaler.pkl'):

  try:
    X, y, feature_names, scaler = load_and_prepare_data(csv_path)

    model = CustomLinearRegression()

    model.fit(X, y, learning_rate=learning_rate, n_iterations=n_iterations)

    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)

    model.save_model(model_save_path)

    with open(scaler_save_path, 'wb') as f:
      pickle.dump(scaler, f)
    
    feature_names_path = 'models/feature_names.pkl'
    with open(feature_names_path, 'wb') as f:
      pickle.dump(feature_names, f)
    
    final_cost = model.history[-1] if model.history is not None else None

    return {
      'status':'success',
      'message':'Model Trained as it should',
      'final_cost':float(final_cost) if final_cost is not None else None,
      'iterations':n_iterations,
      'learning_rate': learning_rate,
      'samples_trained': len(y),
      'features_count': len(feature_names),
      'model_saved_at': model_save_path,
      'scaler_saved_at':scaler_save_path
    }
  except Exception as e:
    return {
      "status":"error",
      "message":f"Training failed : {str(e)}"
    }