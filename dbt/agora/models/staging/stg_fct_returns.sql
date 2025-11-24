{{ config(
    materialized='table'
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'fct_returns') }}
),

renamed_columns AS (
    SELECT
        ID_RETURNS AS return_id,
        ID_ORDERS AS order_id,
        ID_DIM_DATE_RETURNED AS date_returned_id,
        REASON AS reason
    FROM source
)

SELECT * FROM renamed_columns;