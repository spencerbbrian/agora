{{ config(
    materialized='table',
    tags=['staging']
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'fct_orders') }}
),

renamed_columns AS (
    SELECT
        ID_ORDERS AS order_id,
        ID_STORE AS store_id,
        ID_SUPPLIERS AS supplier_id
    FROM source
)

SELECT * FROM renamed_columns;