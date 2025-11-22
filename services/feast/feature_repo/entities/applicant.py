"""
Applicant entity definition for Home Credit feature store.

This entity represents a credit applicant identified by SK_ID_CURR.
"""

from feast import Entity, ValueType

applicant = Entity(
    name="applicant",
    join_keys=["SK_ID_CURR"],
    value_type=ValueType.INT64,
    description="Home Credit applicant identified by SK_ID_CURR",
)
