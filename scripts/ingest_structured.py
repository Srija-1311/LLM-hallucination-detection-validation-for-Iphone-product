import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "foxconn_iphone_kb"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}


# ============================================================
# 2. DATASET PATH
# ============================================================

# Change this path to the location of your dataset.

DATASET_ROOT = Path(
    "C:/Users/srija/OneDrive/ドキュメント/foxconn synthetic data/foxconn synthetic data"
)

STRUCTURED_ROOT = DATASET_ROOT / "structured_data"


# ============================================================
# 3. CONNECT TO POSTGRESQL
# ============================================================

def get_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to PostgreSQL successfully.")
        return conn

    except Exception as e:
        print("Could not connect to PostgreSQL.")
        print("Error:", e)
        raise


def get_product_id(cur, product_name):
    """
    Convert a product name from the CSV into the normalized
    product_id stored in the products table.
    """
    cur.execute(
        """
        SELECT product_id
        FROM products
        WHERE product_name = %s
        """,
        (product_name,)
    )

    result = cur.fetchone()

    if not result:
        raise ValueError(
            f"Product '{product_name}' does not exist in the products table."
        )

    return result[0]

# ============================================================
# 4. LOAD PRODUCT MASTER
# ============================================================

def load_products(conn):
    """
    Creates/loads the normalized products table from the
    product_master CSV.

    Expected product_master records include attributes such as:
        Product_ID
        Product_Family
        Model_Year
        Synthetic_Dataset
    """

    file_path = (
        STRUCTURED_ROOT
        / "product_master"
        / "product_master.csv"
    )

    print(f"\nReading: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Records found: {len(df)}")

    # Display the actual data so you can verify it.
    print(df.to_string(index=False))

    product_id = None
    product_name = None
    product_family = None
    model_year = None
    synthetic_flag = True

    for _, row in df.iterrows():

        attribute = str(row["attribute_or_type"]).strip()
        value = str(row["value"]).strip()

        attribute_lower = attribute.lower()

        if attribute_lower == "product_id":
            product_id = value

        elif attribute_lower == "product_family":
            product_family = value

        elif attribute_lower == "model_year":
            try:
                model_year = int(float(value))
            except ValueError:
                model_year = None

        elif attribute_lower in (
            "synthetic_dataset",
            "synthetic_dataset_flag"
        ):
            synthetic_flag = value.lower() in (
                "true",
                "yes",
                "1"
            )

    # The dataset represents the iPhone 17.
    # Use product_id from the dataset rather than inventing one.
    if product_id is None:
        raise ValueError(
            "Product_ID was not found in product_master.csv"
        )

    product_name = "iPhone 17"

    insert_sql = """
        INSERT INTO products (
            product_id,
            product_name,
            product_family,
            model_year,
            synthetic_dataset_flag,
            source_type
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (product_id)
        DO UPDATE SET
            product_name = EXCLUDED.product_name,
            product_family = EXCLUDED.product_family,
            model_year = EXCLUDED.model_year,
            synthetic_dataset_flag =
                EXCLUDED.synthetic_dataset_flag,
            source_type = EXCLUDED.source_type;
    """

    with conn.cursor() as cur:

        cur.execute(
            insert_sql,
            (
                product_id,
                product_name,
                product_family,
                model_year,
                synthetic_flag,
                "synthetic",
            ),
        )

    conn.commit()

    print(
        f"Product inserted/updated: "
        f"{product_id} - {product_name}"
    )

    return product_id


# ============================================================
# 5. GENERIC TABLE LOADER
# ============================================================

def load_standard_table(
    conn,
    csv_path,
    table_name,
    value_type="text"
):
    """
    Loads CSVs having the common structure:

        record_id
        product
        attribute_or_type
        value
        unit_or_status
        source_reference
        source_type

    into their corresponding PostgreSQL tables.
    """

    print(f"\n{'=' * 60}")
    print(f"Loading: {table_name}")
    print(f"File: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"Records found: {len(df)}")

    required_columns = [
        "record_id",
        "product",
        "attribute_or_type",
        "value",
        "unit_or_status",
        "source_reference",
        "source_type",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {missing}"
        )

    rows = []

    with conn.cursor() as cur:

        for _, row in df.iterrows():

            record_id = str(row["record_id"]).strip()

            product_name = str(row["product"]).strip()
            product_id = get_product_id(cur, product_name)

            attribute = str(row["attribute_or_type"]).strip()

            value = row["value"]

            if pd.isna(value):
                value = None
            else:
                value = str(value).strip()

            unit_status = row["unit_or_status"]

            if pd.isna(unit_status):
                unit_status = None
            else:
                unit_status = str(unit_status).strip()

            source_reference = row["source_reference"]

            if pd.isna(source_reference):
                source_reference = None
            else:
                source_reference = str(source_reference).strip()

            source_type = row["source_type"]

            if pd.isna(source_type):
                source_type = "synthetic"
            else:
                source_type = str(source_type).strip()

            if value_type == "numeric" and value is not None:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(
                        f"Expected numeric value in {csv_path.name}, "
                        f"record {record_id}: {value}"
                    )

            rows.append(
                (
                    record_id,
                    product_id,
                    attribute,
                    value,
                    unit_status,
                    source_reference,
                    source_type,
                )
            )

    insert_sql = f"""
        INSERT INTO {table_name} (
            record_id,
            product_id,
            attribute_or_type,
            value,
            unit_or_status,
            source_reference,
            source_type
        )
        VALUES %s
        ON CONFLICT (record_id)
        DO UPDATE SET
            product_id = EXCLUDED.product_id,
            attribute_or_type = EXCLUDED.attribute_or_type,
            value = EXCLUDED.value,
            unit_or_status = EXCLUDED.unit_or_status,
            source_reference = EXCLUDED.source_reference,
            source_type = EXCLUDED.source_type;
    """

    with conn.cursor() as cur:

        execute_values(
            cur,
            insert_sql,
            rows
        )

    conn.commit()

    print(
        f"Successfully inserted/updated "
        f"{len(rows)} records into {table_name}"
    )


# ============================================================
# 6. TRACEABILITY LOADER
# ============================================================

def load_traceability(conn):

    csv_path = (
        STRUCTURED_ROOT
        / "traceability"
        / "traceability.csv"
    )

    print(f"\n{'=' * 60}")
    print("Loading: traceability")

    df = pd.read_csv(csv_path)

    print(f"Records found: {len(df)}")

    rows = []

    with conn.cursor() as cur:  
        for _, row in df.iterrows():
            record_id = str(row["record_id"]).strip()

            product_name = str(row["product"]).strip()
            product_id = get_product_id(cur, product_name)

            attribute = str(
                row["attribute_or_type"]
            ).strip()

            # continue with the rest of your existing code

            evidence_reference = str(
                row["value"]
            ).strip()

            responsible_source = row["unit_or_status"]

            if pd.isna(responsible_source):
                responsible_source = None
            else:
                responsible_source = str(
                    responsible_source
                ).strip()

            source_reference = row["source_reference"]

            if pd.isna(source_reference):
                source_reference = None
            else:
                source_reference = str(
                    source_reference
                ).strip()

            source_type = row["source_type"]

            if pd.isna(source_type):
                source_type = "synthetic"
            else:
                source_type = str(
                    source_type
                ).strip()

            rows.append(
                (
                    record_id,
                    product_id,
                    attribute,
                    evidence_reference,
                    responsible_source,
                    source_reference,
                    source_type,
                )
            )

    sql = """
        INSERT INTO traceability (
            record_id,
            product_id,
            attribute_or_type,
            evidence_reference,
            responsible_source,
            source_reference,
            source_type
        )
        VALUES %s
        ON CONFLICT (record_id)
        DO UPDATE SET
            product_id = EXCLUDED.product_id,
            attribute_or_type =
                EXCLUDED.attribute_or_type,
            evidence_reference =
                EXCLUDED.evidence_reference,
            responsible_source =
                EXCLUDED.responsible_source,
            source_reference =
                EXCLUDED.source_reference,
            source_type =
                EXCLUDED.source_type;
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows)

    conn.commit()

    print(
        f"Successfully inserted/updated "
        f"{len(rows)} traceability records"
    )


# ============================================================
# 7. CLAIM EVIDENCE LOADER
# ============================================================

def load_claim_evidence(conn):

    csv_path = (
        STRUCTURED_ROOT
        / "claim_evidence"
        / "claim_evidence.csv"
    )

    print(f"\n{'=' * 60}")
    print("Loading: claim_evidence")

    df = pd.read_csv(csv_path)

    print(f"Records found: {len(df)}")

    rows = []

    with conn.cursor() as cur:
        for _, row in df.iterrows():

            record_id = str(row["record_id"]).strip()
            product_name = str(row["product"]).strip()
            product_id = get_product_id(cur, product_name)

            attribute = str(
                row["attribute_or_type"]
            ).strip()

            evidence_status = str(
                row["value"]
            ).strip().upper()

            evidence_type = row["unit_or_status"]

            if pd.isna(evidence_type):
                evidence_type = None
            else:
                evidence_type = str(
                    evidence_type
                ).strip()

            source_reference = row["source_reference"]

            if pd.isna(source_reference):
                source_reference = None
            else:
                source_reference = str(
                    source_reference
                ).strip()

            source_type = row["source_type"]

            if pd.isna(source_type):
                source_type = "synthetic"
            else:
                source_type = str(
                    source_type
                ).strip()

            rows.append(
                (
                    record_id,
                    product_id,
                    attribute,
                    evidence_status,
                    evidence_type,
                    source_reference,
                    source_type,
                )
            )

    sql = """
        INSERT INTO claim_evidence (
            record_id,
            product_id,
            attribute_or_type,
            evidence_status,
            evidence_type,
            source_reference,
            source_type
        )
        VALUES %s
        ON CONFLICT (record_id)
        DO UPDATE SET
            product_id = EXCLUDED.product_id,
            attribute_or_type =
                EXCLUDED.attribute_or_type,
            evidence_status =
                EXCLUDED.evidence_status,
            evidence_type =
                EXCLUDED.evidence_type,
            source_reference =
                EXCLUDED.source_reference,
            source_type =
                EXCLUDED.source_type;
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows)

    conn.commit()

    print(
        f"Successfully inserted/updated "
        f"{len(rows)} claim-evidence records"
    )


# ============================================================
# 8. MAIN INGESTION PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("FOXCONN iPHONE STRUCTURED DATA INGESTION")
    print("=" * 70)

    if not STRUCTURED_ROOT.exists():
        raise FileNotFoundError(
            f"Structured dataset not found:\n"
            f"{STRUCTURED_ROOT}"
        )

    conn = get_connection()

    try:

        # ----------------------------------------------------
        # 1. Product master
        # ----------------------------------------------------

        product_id = load_products(conn)

        print(
            f"\nUsing product_id: {product_id}"
        )

        # ----------------------------------------------------
        # 2. Standard text table
        # ----------------------------------------------------

        load_standard_table(
            conn,
            STRUCTURED_ROOT
            / "product_master"
            / "product_master.csv",
            "product_master_records",
            value_type="text",
        )

        # ----------------------------------------------------
        # 3. Component specifications
        # ----------------------------------------------------

        load_standard_table(
            conn,
            STRUCTURED_ROOT
            / "component_specs"
            / "component_specs.csv",
            "component_specs",
            value_type="text",
        )

        # ----------------------------------------------------
        # 4. Quality metrics
        # ----------------------------------------------------

        load_standard_table(
            conn,
            STRUCTURED_ROOT
            / "quality_metrics"
            / "quality_metrics.csv",
            "quality_metrics",
            value_type="numeric",
        )

        # ----------------------------------------------------
        # 5. Manufacturing records
        # ----------------------------------------------------

        load_standard_table(
            conn,
            STRUCTURED_ROOT
            / "manufacturing_records"
            / "manufacturing_records.csv",
            "manufacturing_records",
            value_type="numeric",
        )

        # ----------------------------------------------------
        # 6. Defect records
        # ----------------------------------------------------

        load_standard_table(
            conn,
            STRUCTURED_ROOT
            / "defect_records"
            / "defect_records.csv",
            "defect_records",
            value_type="numeric",
        )

        # ----------------------------------------------------
        # 7. Supplier records
        # ----------------------------------------------------

        load_standard_table(
            conn,
            STRUCTURED_ROOT
            / "supplier_records"
            / "supplier_records.csv",
            "supplier_records",
            value_type="numeric",
        )

        # ----------------------------------------------------
        # 8. Test results
        # ----------------------------------------------------

        load_standard_table(
            conn,
            STRUCTURED_ROOT
            / "test_results"
            / "test_results.csv",
            "test_results",
            value_type="text",
        )

        # ----------------------------------------------------
        # 9. Traceability
        # ----------------------------------------------------

        load_traceability(conn)

        # ----------------------------------------------------
        # 10. Claim evidence
        # ----------------------------------------------------

        load_claim_evidence(conn)

        print("\n")
        print("=" * 70)
        print("STRUCTURED DATA INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:

        conn.rollback()

        print("\nINGESTION FAILED")
        print("Error:", e)

        raise

    finally:
        conn.close()

        print("\nPostgreSQL connection closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()