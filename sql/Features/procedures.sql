select 
    b.example_id,
    a.person_id,
    a.procedure_concept_id as concept_name,
    a.procedure_date as feature_start_date,
    b.start_date as person_start_date,
    b.end_date as person_end_date
from 
    {cdm_schema}.procedure_occurrence a
inner join
    {cohort_table} b
on 
    a.person_id = b.person_id
where
    a.procedure_date <= b.end_date


