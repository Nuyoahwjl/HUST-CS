-- 用一条SQL语句将new_wage表的全部酬劳信息插入到薪资表(wage)。
--（1）对于兼职人员，发放单位采用一事一酬。因此，某月内，对于某人，
--      new_wage可能有多条待支付酬劳记录。插入wage表时需按月汇总后再插入，w_memo也同时合并。
--（2）当多条记录中的c_id_card，w_amount，w_org，w_time都相同时，认为是重复记录，只保留一条。
--（3）全职的酬劳时间按照原w_time插入，兼职的酬劳时间按照当月的最早w_time插入。
--（4）缴纳税标志默认为N
--（5）w_type为1代表全职,为2代表兼职
--（6）插入wage的顺序为（w_c_id, w_amount, w_org, w_time, w_type, w_memo, w_tax），w_id会自动增加1。
insert into wage
    (w_c_id, w_amount, w_org, w_time, w_type, w_memo, w_tax)
select *
from (
    select c.c_id as w_c_id,
           d.w_amount as w_amount,
           d.w_org as w_org,
           d.w_time as w_time,
           1 as w_type,
           group_concat(d.w_memo order by d.id) as w_memo,
           'N' as w_tax
    from (
        select n.*,
               row_number() over (
               partition by n.c_id_card,
                    n.w_amount,
                    n.w_org,
                    n.w_time
               order by n.id
               ) as rn
        from new_wage n
        where n.w_type = 1
          ) as d
    join client c on c.c_id_card = d.c_id_card
    where d.rn = 1
    group by c.c_id, d.w_amount, d.w_org, d.w_time

    union all

    select c.c_id as w_c_id,
           sum(d.w_amount) as w_amount,
           d.w_org as w_org,
           min(d.w_time) as w_time,
           2 as w_type,
           group_concat(d.w_memo order by d.w_time) as w_memo,
           'N' as w_tax
    from (
        select n.*,
               row_number() over (
               partition by n.c_id_card,
                    n.w_amount,
                    n.w_org,
                    n.w_time
               order by n.id
               ) as rn
        from new_wage n
        where n.w_type = 2
          ) as d
    join client c on c.c_id_card = d.c_id_card
    where d.rn = 1
    group by c.c_id, d.w_org, date_format(d.w_time, '%Y-%m')
      ) as ready2insert

order by w_c_id, w_org, w_time;








insert into wage (w_c_id, w_amount, w_org, w_time, w_type, w_memo, w_tax)
select
    c.c_id,
    sub.w_amount,
    sub.w_org,
    sub.w_time,
    sub.w_type,
    sub.w_memo,
    'N' as w_tax
from (
    -- 全职：去重后，保留最小id作为排序依据
    select
        min(nw.id) as sort_id,
        nw.c_id_card,
        nw.w_amount,
        nw.w_org,
        nw.w_time,
        1 as w_type,
        nw.w_memo
    from new_wage nw
    where nw.w_type = 1
    group by
        nw.c_id_card,
        nw.w_amount,
        nw.w_org,
        nw.w_time,
        nw.w_memo

    union all

    -- 兼职：先对原表去重，再分组聚合
    select
        min(id) as sort_id,
        c_id_card,
        sum(w_amount) as w_amount,
        w_org,
        min(w_time) as w_time,
        2 as w_type,
        group_concat(distinct w_memo order by w_time separator '；') as w_memo
    from (
        -- 去除重复记录
        select distinct
            id,
            c_id_card,
            w_amount,
            w_org,
            w_time,
            w_memo,
            date_format(w_time, '%Y-%m') as ym
        from new_wage
        where w_type = 2
    ) as distinct_parttime
    group by
        c_id_card,
        w_org,
        ym
) as sub
join client c on sub.c_id_card = c.c_id_card
order by sub.sort_id;