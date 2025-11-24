{{ config(
    materialized='table',
    tags=['staging']
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'dim_brand') }}
),

renamed_columns AS (
    SELECT
        ID_BRAND AS brand_id,
        BRAND_NAME AS brand_name,
        BRAND_CATEGORY AS brand_category
    FROM source
)

SELECT * FROM renamed_columns;