from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import warnings

from scripts.model_training.light_gbm import light_gbm

from scripts.feature_engineer import feature_engineer

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

# Read in all data
train = pd.read_csv('../data/application_train.csv')
test = pd.read_csv('../data/application_test.csv')
bureau = pd.read_csv('../data/bureau.csv')
bureau_balance = pd.read_csv('../data/bureau_balance.csv')
credit_card_balance = pd.read_csv('../data/credit_card_balance.csv')
installments_payments = pd.read_csv('../data/installments_payments.csv')
pos_cash_balance = pd.read_csv('../data/POS_CASH_balance.csv')
previous_application = pd.read_csv('../data/previous_application.csv')

train, test = feature_engineer(train, test, bureau, bureau_balance, credit_card_balance,
                               installments_payments, pos_cash_balance, previous_application)

submission, fi, metrics = light_gbm(train, test)

print(fi)
print(metrics)

submission.to_csv('lightGBM.csv', index=False)
