{{ config(
    materialized='table'
) }}

WITH source AS (
    SELECT * FROM {{ source('agora', 'dim_products') }}
),

renamed_columns AS (
    SELECT
        ID_PRODUCTS AS product_id,
        ID_BRAND AS brand_id,
        "Name" AS product_name,
        "Price" AS price,
        "Description" AS description,
        "Rate" AS rating,
        "Rating_count" AS rating_count,
        "Gender" AS gender,
        "Product_Type" AS type,
        "Character_x" AS character,
        "Fragrance_Family" AS fragrance_family,
        "Size" AS size,
        "Year" AS year,
        "Ingredients" AS ingredients,
        "Concentration" AS concentration,
        "Top_note" AS top_note,
        "Middle_note" AS middle_note,
        "Base_note" AS base_note
    FROM source
)

SELECT * FROM renamed_columns