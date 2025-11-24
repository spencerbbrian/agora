{{ config(
    materialized='table',
    tags=['staging']
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'dim_stores') }}
),

renamed_columns AS (
    SELECT
        ID_STORE AS store_id,
        CITY AS city,
        REGION AS region,
        ID_WAREHOUSE_SUPPLYING AS supplying_warehouse
    FROM source
)

SELECT * FROM renamed_columns