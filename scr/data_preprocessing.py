import pandas as pd
import numpy as np


def aggregate_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate item-level data into order-level features.
    """

    order_items_agg = order_items.groupby('order_id').agg(
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        total_items=('order_item_id', 'count')
    ).reset_index()

    return order_items_agg


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate payment information at order level.
    """

    payments_agg = payments.groupby('order_id').agg(
        total_payment=('payment_value', 'sum'),
        payment_types=('payment_type', 'nunique'),
        installments=('payment_installments', 'max')
    ).reset_index()

    return payments_agg


def build_master_dataset(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items_agg: pd.DataFrame,
    payments_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge multiple tables into one master dataset.
    """

    df = orders.merge(
        customers,
        on='customer_id',
        how='left'
    )

    df = df.merge(
        order_items_agg,
        on='order_id',
        how='left'
    )

    df = df.merge(
        payments_agg,
        on='order_id',
        how='left'
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business and ML-related features.
    """

    # Datetime conversion
    df['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp']
    )

    df['order_delivered_customer_date'] = pd.to_datetime(
        df['order_delivered_customer_date']
    )

    # Order value
    df['order_value'] = df['total_payment']

    # Delivery time
    df['delivery_time'] = (
        df['order_delivered_customer_date']
        - df['order_purchase_timestamp']
    ).dt.days

    # Time features
    df['order_year'] = df['order_purchase_timestamp'].dt.year
    df['order_month'] = df['order_purchase_timestamp'].dt.month

    # Customer total orders
    customer_orders = df.groupby(
        'customer_unique_id'
    ).agg(
        total_orders=('order_id', 'count')
    ).reset_index()

    df = df.merge(
        customer_orders,
        on='customer_unique_id',
        how='left'
    )

    # Recency feature
    last_purchase = df.groupby(
        'customer_unique_id'
    )['order_purchase_timestamp'].max().reset_index()

    last_purchase.columns = [
        'customer_unique_id',
        'last_purchase_date'
    ]

    df = df.merge(
        last_purchase,
        on='customer_unique_id',
        how='left'
    )

    df['recency_days'] = (
        df['order_purchase_timestamp'].max()
        - df['last_purchase_date']
    ).dt.days

    return df


def filter_delivered_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only delivered orders.
    """

    return df[df['order_status'] == 'delivered'].copy()