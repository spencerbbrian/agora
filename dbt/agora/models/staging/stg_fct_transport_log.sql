{{ config(
    materialized='table',
    tags=['staging']
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'fct_transport_log') }}
),

renamed_columns AS (
    SELECT
        ID_TRANSPORT_LOG AS id_transport_log,
        ID_WAREHOUSE AS id_warehouse,
        ID_STORE AS id_store,
        ID_SUPPLIERS AS id_suppliers,
        JOURNEY_STATUS AS journey_status,
        TS_STOCK_SHIPPED AS ts_stock_shipped,
        TS_STOCK_RECEIVED AS ts_stock_received,
        TS_LAST_UPDATE AS ts_last_update,
        ID_DIM_DATE_STOCK_SHIPPED AS id_dim_date_stock_shipped,
        ID_DIM_DATE_STOCK_RECEIVED AS id_dim_date_stock_received,
        ID_DIM_DATE_LAST_UPDATE AS id_dim_date_last_update
    FROM source
)

SELECT * FROM renamed_columns