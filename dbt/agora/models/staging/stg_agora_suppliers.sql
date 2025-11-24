{{ config(
    materialized='table',
    tags=['staging']
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'dim_suppliers') }}
),

renamed_columns AS (
    SELECT
        ID_SUPPLIERS AS supplier_id,
        SUPPLIER_NAME AS supplier_name
    FROM source
)

SELECT * FROM renamed_columns;