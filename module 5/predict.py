import pickle

model_file = 'model_C=1.0.bin'


with open(model_file, 'rb') as f_in:
#with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

customer = {
    'gender' : 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity' : 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport' : 'no',
    'streamingtv' : 'no',
    'streamingmovies' : 'no',
    'contract' : 'month-to-month',
    'paperlessbilling' : 'yes',
    'paymentmethod' : 'electronic_check',
    'tenure' : 1 , # 10, 20
    'monthlycharges' : 29.85,
    'totalcharges' : 29.85
}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]


print('input', customer)
print('churn probability', y_pred)