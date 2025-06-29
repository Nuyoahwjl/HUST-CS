-- 本关任务：在均不计兼职的情况下，统计各单位的薪资总额、月平均薪资、最高薪资、最低薪资、中位薪资
-- 计算各单位的薪资总额(total_amount), 月平均薪资（average_wage）、最高及最低薪资(max_wage，min_wage)、中位薪资（mid_wage）。查询结果按总金额降序排序。
-- 1）月平均工资是薪资总额除以人数及月份。
-- 2）最高及最低薪资是实际月薪资，非平均后的最高及最低薪资。
-- 3）中位薪资是该单位员工人数中处于中间位置的人的月平均工资。例如，单位X共有9人，月平均薪资排第5人的工资即为中位薪资。若为偶数时，是中间两位的工资的平均数。
-- 4) 确保只统计与有效客户关联的记录(客户表可能存在悬浮元组)
-- 输出格式：w_org, total_amount, average_wage, max_wage, min_wage, mid_wage

with full_wage as (
        select 
                w.w_org,
                w.w_c_id as c_id,
                w.w_amount,
                date_format(w.w_time, '%Y-%m') as ym
        from 
                wage w
        join 
                client c on c.c_id = w.w_c_id
        where 
                w.w_type = 1
),

org_totals as (
        select 
                w_org,
                sum(w_amount) as total_amount,
                count(distinct c_id) as emp_cnt,
                count(distinct ym) as month_cnt,
                max(w_amount) as max_wage,
                min(w_amount) as min_wage,
                sum(w_amount) / (count(distinct c_id) * count(distinct ym)) as average_wage_raw
        from 
                full_wage
        group by 
                w_org
),

emp_month_avg as (
        select 
                w_org,
                c_id,
                avg(w_amount) as emp_avg_wage
        from 
                full_wage
        group by 
                w_org, c_id
),

emp_rank as (
        select 
                w_org,
                emp_avg_wage,
                row_number() over (partition by w_org order by emp_avg_wage) as rn_asc,
                count(*) over (partition by w_org) as emp_cnt
        from 
                emp_month_avg
),

median_pick as (
        select 
                w_org,
                emp_avg_wage
        from 
                emp_rank
        where 
                (emp_cnt % 2 = 1 and rn_asc = (emp_cnt + 1) / 2)
                or (emp_cnt % 2 = 0 and rn_asc in (emp_cnt / 2, emp_cnt / 2 + 1))
),

median_final as (
        select 
                w_org,
                avg(emp_avg_wage) as mid_wage
        from 
                median_pick
        group by 
                w_org
)

select 
        o.w_org,
        o.total_amount,
        round(o.average_wage_raw, 2) as average_wage,
        o.max_wage,
        o.min_wage,
        round(m.mid_wage, 2) as mid_wage
from 
        org_totals o
join 
        median_final m using (w_org)
order by 
        o.total_amount desc;
