{{ config(
    materialized='table'
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'fct_order_log') }}
),

renamed_columns AS (
    SELECT
        ID_ORDER_LOG AS order_log_id,
        ID_ORDERS AS order_id,
        TS_ORDERED AS ts_ordered,
        TS_SHIPPED_WAREHOUSE AS ts_shipped_warehouse,
        TS_SHIPPED_STORE AS ts_shipped_store,
        TS_DELIVERED AS ts_delivered,
        TS_RETURNED AS ts_returned,
        TS_LAST_UPDATE AS ts_last_update,
        ID_DIM_DATE_ORDERED AS date_ordered_id,
        ID_DIM_DATE_SHIPPED_WH AS date_shipped_wh_id,
        ID_DIM_DATE_SHIPPED_STORE AS date_shipped_store_id,
        ID_DIM_DATE_DELIVERED AS date_delivered_id,
        ID_DIM_DATE_RETURNED AS date_returned_id,
        ID_DIM_DATE_LAST_UPDATE AS date_last_update_id
    FROM source
)

SELECT * FROM renamed_columns