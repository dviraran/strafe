# Database Setup
DB_NAME = '****'
PG_USERNAME = '****'
PG_PASSWORD = '****'
# 614b4ffe9c1992068d618404
# Schemas
OMOP_CDM_SCHEMA = 'omop_all'  # schema holding standard OMOP tables   cur options: SmallSynpuf or bigsynpuf
CDM_AUX_SCHEMA = 'cdm_aux'  # schema to hold auxilliary tables not tied to a particular schema
CDM_VERSION = 'v6'  # set to 'v5.x.x' if on v5

# SQL Paths
SQL_PATH_COHORTS = 'sql/Cohorts'  # path to SQL scripts that generate cohorts
SQL_PATH_FEATURES = 'sql/Features'  # path to SQL scripts that generate features

# Cache
DEFAULT_SAVE_LOC = '/tmp/'  # where to save temp files

# Only used in ORM code
omop_schema = 'cdm'
user_schema = 'eol_cohort_comparison'

# Sql connection parameters

#HOST = 'localhost'
HOST = '****'
#HOST = 'localhost'
DIALECT = 'postgresql'
DRIVER = 'psycopg2'
