from airflow.decorators import dag, task
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.utils.dates import days_ago

@dag(
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["test"],
)
def test_gcp_connection():
    
    @task
    def run_bigquery_query():
        query = """
        SELECT *
        FROM `envilink.temp.your_table_id`
        LIMIT 10
        """

        # Define the BigQuery job
        insert_job = BigQueryInsertJobOperator(
            task_id="run_bigquery_query",
            configuration={
                "query": {
                    "query": query,
                    "useLegacySql": False  # Ensure you use standard SQL
                }
            },
            gcp_conn_id="gcp",
        )

        return insert_job.execute()
    
    # Run the task
    run_bigquery_query()

# Instantiate the DAG
dag_instance = test_gcp_connection()