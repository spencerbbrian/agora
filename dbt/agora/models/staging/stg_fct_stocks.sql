{{ config(
    materialized='table'
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'fct_stocks') }}
),

renamed_columns AS (
    SELECT
        ID_PRODUCTS AS product_id,
        ID_WAREHOUSE AS warehouse_id,
        STOCK AS stock,
        MIN_STOCK AS min_stock,
        ID_DIM_DATE_LAST_STOCKED AS date_last_stocked_id,
        ID_DIM_DATE_LAST_UPDATED AS date_last_updated_id
    FROM source
)

SELECT * FROM renamed_columns;