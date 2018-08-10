"""This is a util that read the data and sample 100 rows from each and save them."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd


def sampling():
    """
    Sample each data file, extract 100 rows from each and save them into new files.
    :return:
    """

    # application_train.csv
    file = pd.read_csv("data/application_train.csv", nrows=100)
    file.to_csv("data/sample/train_first_100.csv", index=False)

    # bureau.csv
    file = pd.read_csv("data/bureau.csv", nrows=100)
    file.to_csv("data/sample/bureau_first_100.csv", index=False)

    # bureau_balance.csv
    file = pd.read_csv("data/bureau_balance.csv", nrows=100)
    file.to_csv("data/sample/bureaubalance_first_100.csv", index=False)

    # credit_card_balance.csv
    file = pd.read_csv("data/credit_card_balance.csv", nrows=100)
    file.to_csv("data/sample/creditcard_first_100.csv", index=False)

    # installments_payments.csv
    file = pd.read_csv("data/installments_payments.csv", nrows=100)
    file.to_csv("data/sample/installpayments_first_100.csv", index=False)

    # POS_CASH_balance.csv
    file = pd.read_csv("data/POS_CASH_balance.csv", nrows=100)
    file.to_csv("data/sample/poscash_first_100.csv", index=False)

    # previous_application.csv
    file = pd.read_csv("data/previous_application.csv", nrows=100)
    file.to_csv("data/sample/previousapp_100.csv", index=False)


if __name__ == "__main__":
    sampling()
