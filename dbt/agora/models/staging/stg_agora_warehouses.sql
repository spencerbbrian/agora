{{ config(
    materialized='table',
    tags=['staging']
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'dim_warehouses') }}
),

renamed_columns AS (
    SELECT
        ID_WAREHOUSE AS warehouse_id,
        WH_NAME AS wh_name,
        CITY AS city,
        REGION AS region,
        CAPACITY AS capacity
    FROM source
)

SELECT * FROM renamed_columns;