SET search_path TO home_credit, public;

TRUNCATE TABLE home_credit.application_train;
TRUNCATE TABLE home_credit.application_test;
TRUNCATE TABLE home_credit.bureau;
TRUNCATE TABLE home_credit.bureau_balance;
TRUNCATE TABLE home_credit.credit_card_balance;
TRUNCATE TABLE home_credit.pos_cash_balance;

COPY home_credit.application_train
FROM '/data/home-credit/application_train.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);

COPY home_credit.application_test
FROM '/data/home-credit/application_test.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);

COPY home_credit.bureau
FROM '/data/home-credit/bureau.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);

COPY home_credit.bureau_balance
FROM '/data/home-credit/bureau_balance.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);

COPY home_credit.credit_card_balance (
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_ATM_CURRENT",
    "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_OTHER_CURRENT",
    "AMT_DRAWINGS_POS_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_PAYMENT_CURRENT",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "AMT_RECEIVABLE_PRINCIPAL",
    "AMT_RECIVABLE",
    "AMT_TOTAL_RECEIVABLE",
    "CNT_DRAWINGS_ATM_CURRENT",
    "CNT_DRAWINGS_CURRENT",
    "CNT_DRAWINGS_OTHER_CURRENT",
    "CNT_DRAWINGS_POS_CURRENT",
    "CNT_INSTALMENT_MATURE_CUM",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF"
)
FROM PROGRAM $$
awk -F',' 'NR == 1 {print; next} {
  OFS=",";
  if (NF < 23) {
    for (i = NF + 1; i <= 23; i++) {
      $i = "";
    }
  }
  print
}' /data/home-credit/credit_card_balance.csv
$$
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);

COPY home_credit.pos_cash_balance
FROM '/data/home-credit/POS_CASH_balance.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);

TRUNCATE TABLE home_credit.previous_application;
TRUNCATE TABLE home_credit.installments_payments;

COPY home_credit.previous_application
FROM '/data/home-credit/previous_application.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);

COPY home_credit.installments_payments
FROM '/data/home-credit/installments_payments.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    NULL ''
);
