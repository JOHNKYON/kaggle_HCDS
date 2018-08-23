from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.feature_engineer import count_categorical
from utils.feature_engineer import agg_numeric
from utils.feature_engineer import missing_values_table
from utils.feature_engineer import light_gbm

import warnings

warnings.filterwarnings("ignore")


def feature_engineer(train, test, bureau, bureau_balance, credit_card_balance,
                     installments_payments, pos_cash_balance, previous_application):
    """
    This function read all the data from the competition and do manual feature engineer to it.
    :param train:
    :param test:
    :param bureau:
    :param bureau_balance:
    :param credit_card_balance:
    :param installments_payments:
    :param pos_cash_balance:
    :param previous_application:
    :return: (Dataframe) train
             (Datafarme) test
    """
    bureau_counts = count_categorical(bureau, group_var='SK_ID_CURR', df_name='bureau')
    bureau_agg = agg_numeric(bureau.drop(columns=['SK_ID_BUREAU']), group_var='SK_ID_CURR', df_name='bureau')
    bureau_balance_counts = count_categorical(bureau_balance, group_var='SK_ID_BUREAU', df_name='bureau_balance')
    bureau_balance_agg = agg_numeric(bureau_balance, group_var='SK_ID_BUREAU', df_name='bureau_balance')
    credit_card_balance_counts = count_categorical(credit_card_balance,
                                                   group_var='SK_ID_CURR', df_name='credit_card_balance')
    credit_card_balance_agg = agg_numeric(credit_card_balance.drop(columns=['SK_ID_PREV']),
                                          group_var='SK_ID_CURR', df_name='credit_card_balance')
    # Reason: Installments_payments_counts table contains no object value.
    # installments_payments_counts = count_categorical(installments_payments,
    #                                                  group_var='SK_ID_CURR', df_name='installments_payments')
    installments_payments_agg = agg_numeric(installments_payments.drop(columns=['SK_ID_PREV']),
                                            group_var='SK_ID_CURR', df_name='installments_payments')
    pos_cash_balance_counts = count_categorical(pos_cash_balance, group_var='SK_ID_CURR', df_name='pos_cash_balance')
    pos_cash_balance_agg = agg_numeric(pos_cash_balance.drop(columns=['SK_ID_PREV']),
                                       group_var='SK_ID_CURR', df_name='pos_cash_balance')
    previous_application_counts = count_categorical(previous_application,
                                                    group_var='SK_ID_CURR', df_name='previous_application_counts')
    previous_application_agg = agg_numeric(previous_application.drop(columns=['SK_ID_PREV']),
                                           group_var='SK_ID_CURR', df_name='previous_application')

    # Dataframe grouped by the loan
    bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts,
                                              right_index=True, left_on='SK_ID_BUREAU', how='outer')

    # Merge to include the SK_ID_CURR
    bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on='SK_ID_BUREAU', how='left')

    # Aggregate the stats for each client
    bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns=['SK_ID_BUREAU']),
                                           group_var='SK_ID_CURR', df_name='client')

    original_features = list(train.columns)
    print('Original Number of Features: ', len(original_features))

    # TODO: We can also first deal with pos_cash_balance and credit card balance before merge.

    # Merge with the value counts of bureau
    train = train.merge(bureau_counts, on='SK_ID_CURR', how='left')

    # Merge with the stats of bureau
    train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')

    # Merge with the monthly information grouped by client
    train = train.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')

    # Merge with credit card balance counts
    train = train.merge(credit_card_balance_counts, on='SK_ID_CURR', how='left')

    # Merge with credit card balance agg
    train = train.merge(credit_card_balance_agg, on='SK_ID_CURR', how='left')

    # Merge with installments payments agg
    train = train.merge(installments_payments_agg, on='SK_ID_CURR', how='left')

    # Merge with pos_cash_balance counts
    train = train.merge(pos_cash_balance_counts, on='SK_ID_CURR', how='left')

    # Merge with pos_cash_balance agg
    train = train.merge(pos_cash_balance_agg, on='SK_ID_CURR', how='left')

    # Merge with previous_application counts
    train = train.merge(previous_application_counts, on='SK_ID_CURR', how='left')

    # Merge with previous application agg
    train = train.merge(previous_application_agg, on='SK_ID_CURR', how='left')

    new_features = list(train.columns)
    print('Number of features using previous loans from other institutions data: ', len(new_features))

    missing_train = missing_values_table(train)

    missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > 90])

    # Test
    # Merge with the value counts of bureau
    test = test.merge(bureau_counts, on='SK_ID_CURR', how='left')

    # Merge with the stats of bureau
    test = test.merge(bureau_agg, on='SK_ID_CURR', how='left')

    # Merge with the monthly information grouped by client
    test = test.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')

    # Merge with credit card balance counts
    test = test.merge(credit_card_balance_counts, on='SK_ID_CURR', how='left')

    # Merge with credit card balance agg
    test = test.merge(credit_card_balance_agg, on='SK_ID_CURR', how='left')

    # Merge with installments payments agg
    test = test.merge(installments_payments_agg, on='SK_ID_CURR', how='left')

    # Merge with pos_cash_balance counts
    test = test.merge(pos_cash_balance_counts, on='SK_ID_CURR', how='left')

    # Merge with pos_cash_balance agg
    test = test.merge(pos_cash_balance_agg, on='SK_ID_CURR', how='left')

    # Merge with previous_application counts
    test = test.merge(previous_application_counts, on='SK_ID_CURR', how='left')

    # Merge with previous application agg
    test = test.merge(previous_application_agg, on='SK_ID_CURR', how='left')

    print('Shape of Training Data: ', train.shape)
    print('Shape of Testing Data: ', test.shape)

    train_labels = train['TARGET']

    # Align the dataframes, this will remove the 'TARGET' column
    train, test = train.align(test, join='inner', axis=1)
    train['TARGET'] = train_labels

    print('Training Data Shape: ', train.shape)
    print('Testing Data Shape ', test.shape)

    missing_test = missing_values_table(test)
    missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > 90])
    len(missing_test_vars)

    missing_columns = list(set(missing_test_vars + missing_train_vars))
    print('There are %d columns with more than 90%% missing in either the training or testing data.'
          % len(missing_columns))

    # Drop the missing columns
    train = train.drop(columns=missing_columns)
    test = test.drop(columns=missing_columns)

    train.to_csv('train_all_raw.csv', index=False)
    test.to_csv('test_all_raw.csv', index=False)

    # Calculate all correlations in dataframe
    corrs = train.corr()

    corrs = corrs.sort_values('TARGET', ascending=False)

    # Set the threshold
    threshold = 0.8

    # Empty dictionary to hold correlated variables
    above_threshold_vars = {}

    # For each column, record the variables that are above the threshold
    for col in corrs:
        above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])

    # Track columns to remove and columns already examined
    cols_to_remove = []
    cols_seen = []
    cols_to_remove_paire = []

    # Iterate through columns and correlated columns
    for key, value in above_threshold_vars.items():
        # Keep track of columns already examined
        cols_seen.append(key)
        for x in value:
            if x == key:
                next
            else:
                # Only want to remove on in a pair
                if x not in cols_seen:
                    cols_to_remove.append(x)
                    cols_to_remove_paire.append(key)

    cols_to_remove = list(set(cols_to_remove))
    print('Number of columns to remove: ', len(cols_to_remove))

    train_corrs_removed = train.drop(columns=cols_to_remove)
    test_corrs_removed = test.drop(columns=cols_to_remove)

    print('Training Corrs Removed Shape: ', train_corrs_removed.shape)
    print('Test Corrs Removed ShapeL ', test_corrs_removed.shape)

    train_corrs_removed.to_csv('train_all_corrs_removed.csv', index=False)
    test_corrs_removed.to_csv('test_all_corrs_removed.csv', index=False)

    return train_corrs_removed, test_corrs_removed
