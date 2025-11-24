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
        STORE_NAME AS store_name,
        STORE_LOCATION AS store_location
    FROM source
)

SELECT * FROM renamed_columns;