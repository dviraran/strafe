select 
    b.example_id,
    a.person_id,
    a.condition_concept_id as concept_name,
    date '1900-01-01' - b.end_date + date (a.condition_start_date) as feature_start_date,      -- dummy date to right-align features
    b.start_date as person_start_date,
    b.end_date as person_end_date
from 
    {cdm_schema}.condition_occurrence a
inner join
    {cohort_table} b
on 
    a.person_id = b.person_id

