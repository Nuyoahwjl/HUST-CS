-- 1) 查询销售总额前三的理财产品
select * from(
    select 
        pyear, 
        rank() over(partition by pyear order by sumamount desc) as rk,
        p_id,
        sumamount
    from (
        select 
            year(pro_purchase_time) as pyear,
            p_id,
            sum(pro_quantity * p_amount) as sumamount
        from property, finances_product
        where pro_pif_id = p_id
        and pro_type = 1
        and year(pro_purchase_time) in (2010, 2011)
        group by p_id, pyear
    ) as t1
) as t2
where rk<=3
order by pyear, rk, p_id;