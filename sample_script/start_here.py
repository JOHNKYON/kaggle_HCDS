"""
This is a sample script published on kaggle, see https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
for more information
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestClassifier


import warnings


def main():
    warnings.filterwarnings('ignore')

    app_train = pd.read_csv('../data/application_train.csv')

    app_test = pd.read_csv('../data/application_test.csv')

    assert isinstance(app_train, pd.DataFrame)
    missing_values = missing_values_table(app_train)

    label_encoder = LabelEncoder()
    le_count = 0

    for col in app_train:
        if app_train[col].dtype == 'object':
            # If 2 or fewer
            if len(list(app_train[col].unique())) <= 2:
                # Train
                label_encoder.fit(app_train[col])
                # Transform
                app_train[col] = label_encoder.transform(app_train[col])
                app_test[col] = label_encoder.transform(app_test[col])

                # Track how many columns were label encoded.
                le_count += 1

    # print('%d columns were label encoded.' % le_count)

    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)

    train_labels = app_train['TARGET']
    app_train, app_test = app_train.align(app_test, join='inner', axis=1)

    app_train['TARGET'] = train_labels

    correlations = app_train.corr()['TARGET'].sort_values()

    # Display correlations
    # print('Most Positive Correlations: \n', correlations.tail(15))
    # print('\nMost Negative Correlations: \n', correlations.head(15))

    app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

    # Featrue engineering
    app_train_poly, app_test_poly = feature_engineer(app_train, app_test)

    # Print out the new shapes
    # print('Training data with polynomial features shape: ', app_train_poly.shape)
    # print('Testing data with polynomial features shape:  ', app_test_poly.shape)

    # Drop the target from the training data
    if 'TARGET' in app_train:
        train = app_train.drop(columns=['TARGET'])
    else:
        train = app_train.copy()
    features = list(train.columns)

    # Copy of the testing data
    test = app_test.copy()

    print("Imputing begins")
    poly_features_names = list(app_train_poly.columns)

    # Impute the polynomial features
    imputer = Imputer(strategy='median')

    poly_features = imputer.fit_transform(app_train_poly)
    poly_features_test = imputer.transform(app_test_poly)
    print("Imputing finished.")

    # Scale the polynomial features
    scaler = MinMaxScaler(feature_range=(0, 1))

    poly_features = scaler.fit_transform(poly_features)
    poly_features_test = scaler.transform(poly_features_test)

    random_forest_poly = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)

    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)

    print("Model training begins")
    random_forest_poly.fit(poly_features, train_labels)
    print("Model training finished")

    print("Prediction begins")
    predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]
    print("Prediction finished")

    submit = app_test[["SK_ID_CURR"]]
    submit['TARGET'] = predictions

    # Save the submit into a csv
    submit.to_csv('../data/output/random_forest_baseline.csv', index=False)


def feature_engineer(app_train, app_test):
    print("Feature engineering begins")

    assert isinstance(app_train, pd.DataFrame)
    assert isinstance(app_test, pd.DataFrame)

    # Make a new dataframe for polynomial features
    poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
    poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

    imputer = Imputer(strategy='median')

    poly_target = poly_features['TARGET']

    poly_features = poly_features.drop(columns=['TARGET'])

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    from sklearn.preprocessing import PolynomialFeatures

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    # print('Polynomial Features shape: ', poly_features.shape)

    # Create a dataframe of the features
    poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                             'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Add in the target
    poly_features['TARGET'] = poly_target

    # Find the correlations with the target
    poly_corrs = poly_features.corr()['TARGET'].sort_values()

    # Display most negative and most positive
    # print(poly_corrs.head(10))
    # print(poly_corrs.tail(5))

    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test,
                                      columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                  'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Merge polynomial features into training dataframe
    poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
    app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')

    # Merge polnomial features into testing dataframe
    poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
    app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

    # Align the dataframes
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

    print("Feature Engineering finished")
    return app_train_poly, app_test_poly


def missing_values_table(df, prt=False):
    """

    :param df: DataFrame
    :param prt: boolean
    :return:
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Persentage of missing values
    mis_val_precent = 100 * df.isnull().sum() / len(df)

    # Make a table
    mis_val_table = pd.concat([mis_val, mis_val_precent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )

    # Sort by percentage
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print
    if prt:
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


if __name__ == '__main__':
    main()
