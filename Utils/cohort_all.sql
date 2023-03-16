create table {schema_name}.{cohort_table_name} as

select row_number() over (order by person.person_id) - 1 as example_id,
       person.person_id as person_id,
       obs.start_date as start_date,
       obs.end_date as end_date,
       1 as outcome_date,
       1 as y

from
{schema_name}.person
join
(select person_id, observation_period_start_date as start_date, observation_period_end_date as end_date
from {schema_name}.observation_period) obs
on person.person_id = obs.person_id

