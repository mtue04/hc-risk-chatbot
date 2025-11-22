# Airflow Data Pipeline Guide

## Overview

The feature engineering pipeline is implemented as an **Airflow DAG** using **Polars** for high-performance data processing, running entirely in **Docker containers**.

## Architecture Flow

```
PostgreSQL (Raw Data)
       ↓
Airflow Scheduler (triggers DAG)
       ↓
┌──────────────────────────────────────────┐
│  Feature Engineering DAG                  │
│  ┌────────────────────────────────────┐  │
│  │ 1. Extract (Polars + PostgreSQL)   │  │
│  │    - Read all tables               │  │
│  │    - Convert to Polars DataFrames  │  │
│  └────────────────────────────────────┘  │
│              ↓                            │
│  ┌────────────────────────────────────┐  │
│  │ 2. Transform (Polars)              │  │
│  │    - Ratios & interactions         │  │
│  │    - Bureau aggregations           │  │
│  │    - Previous app features         │  │
│  │    - Installments analysis         │  │
│  │    - 170 total features            │  │
│  └────────────────────────────────────┘  │
│              ↓                            │
│  ┌────────────────────────────────────┐  │
│  │ 3. Load (Polars + PostgreSQL)      │  │
│  │    - Write to home_credit.features │  │
│  └────────────────────────────────────┘  │
│              ↓                            │
│  ┌────────────────────────────────────┐  │
│  │ 4. Feast Apply                     │  │
│  │    - Register feature definitions  │  │
│  └────────────────────────────────────┘  │
│              ↓                            │
│  ┌────────────────────────────────────┐  │
│  │ 5. Feast Materialize               │  │
│  │    - Copy features to Redis        │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
       ↓
Redis Online Store (ready for serving)
       ↓
Model API (/predict/applicant)
```