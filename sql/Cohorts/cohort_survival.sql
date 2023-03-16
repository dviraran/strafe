create table {schema_name}.{cohort_table_name} as

select row_number() over (order by ckd2.person_id) - 1 as example_id,
       ckd2.person_id as person_id,
       ckd2.ckd2_time as ckd2_time,
       obs.start_date as obs_start_time,
       obs.end_date as obs_end_time,
       ckd4.ckd4_time as ckd4_time,
       obs.start_date as start_date,
       ckd2.ckd2_time as end_date,
       case when ckd4.ckd4_time is not null then ckd4.ckd4_time else null end as outcome_date,
       case when ckd4.ckd4_time is not null and ckd4.ckd4_time<ckd2.ckd2_time+{Days_after} then 1 else 0 end as y,
       case when ckd4.ckd4_time is not null then 1 else 0 end as e,
       case when ckd4.ckd4_time is not null then ckd4_time-ckd2_time else obs.end_date-ckd2.ckd2_time end as t,
       p.year_of_birth as year_of_birth,
       case when p.gender_concept_id=8507 then 1 else 0 end as gender


from
(select person_id, min(condition_start_date) as ckd2_time
from {schema_name}.condition_occurrence
where condition_concept_id in {CKD2_codes}
group by person_id)  ckd2
join
(select person_id, observation_period_start_date as start_date, observation_period_end_date as end_date
from {schema_name}.observation_period) obs
on ckd2.person_id = obs.person_id
left join
(select person_id, min(condition_start_date) as ckd4_time
from {schema_name}.condition_occurrence
where condition_concept_id in {CKD4_codes}
group by person_id) ckd4
on obs.person_id = ckd4.person_id
left join
(select person_id, gender_concept_id, year_of_birth from {schema_name}.person) p
on ckd2.person_id=p.person_id

left join
(select person_id, count(*) as num_visits from {schema_name}.visit_occurrence group by person_id ) v
on ckd2.person_id=v.person_id

where ckd2.ckd2_time> {Days_before} + obs.start_date and ckd2.ckd2_time is not null and
      ((ckd4.ckd4_time is null) or (ckd2.ckd2_time<ckd4.ckd4_time and ckd4.ckd4_time is not null) ) and v.num_visits>5
      and t>0



