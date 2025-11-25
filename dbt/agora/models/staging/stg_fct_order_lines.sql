{{ config(
    materialized='table'
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'fct_order_lines') }}
),

renamed_columns AS (
    SELECT
        ID_ORDER_LINE AS order_line_id,
        ID_ORDERS AS order_id,
        ID_PRODUCTS AS product_id,
        QUANTITY AS quantity,
        REQUESTED_QTY AS requested_qty,
        RETURNED_FLAG AS returned_flag
    FROM source
)

SELECT * FROM renamed_columns