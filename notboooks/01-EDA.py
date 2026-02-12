# %%
# ============================== IMPORT ==============================
# Manipulation des données : numpy and pandas
import pandas as pd

# File system manangement
import os

# Visualisation : matplotlib and seaborn

# %%
# ============================= DATA LOAD =============================
print(os.listdir("../datas/"))

# %%
# Home Credit columns description
HomeCredit_columns_description = pd.read_csv(
    "../datas/HomeCredit_columns_description.csv"
)
print("Home Credit columns description shape: ", HomeCredit_columns_description.shape)
HomeCredit_columns_description.head()

# %%
# Training data
app_train = pd.read_csv("../datas/application_train.csv")
print("Training data shape: ", app_train.shape)
app_train.head()

# %%
# Test Data
app_test = pd.read_csv("../datas/application_test.csv")
print("Test data shape: ", app_test.shape)
app_test.head()

# %%
# Bureau Balance
bureau_balance = pd.read_csv("../datas/bureau_balance.csv")
print("Bureau Balance shape: ", bureau_balance.shape)
bureau_balance.head()

# %%
# Bureau
bureau = pd.read_csv("../datas/bureau.csv")
print("Bureau Balance shape: ", bureau.shape)
bureau.head()

# %%
# Credit Card Balance
credit_card_balance = pd.read_csv("../datas/credit_card_balance.csv")
print("Credit Card Balance shape: ", credit_card_balance.shape)
credit_card_balance.head()

# %%
# Installements payments
installments_payments = pd.read_csv("../datas/installments_payments.csv")
print("Installements payments shape: ", installments_payments.shape)
installments_payments.head()

# %%
# POS CASH Balance
POS_CASH_balance = pd.read_csv("../datas/POS_CASH_balance.csv")
print("POS CASH Balance shape: ", POS_CASH_balance.shape)
POS_CASH_balance.head()

# %%
# Previous application
previous_application = pd.read_csv("../datas/previous_application.csv")
print("Previous application shape: ", previous_application.shape)
previous_application.head()

# %%
# Sample Submission
sample_submission = pd.read_csv("../datas/sample_submission.csv")
print("Sample Submission shape: ", sample_submission.shape)
sample_submission.head()
