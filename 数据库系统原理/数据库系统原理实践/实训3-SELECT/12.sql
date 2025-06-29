 -- 12) 综合客户表(client)、资产表(property)、理财产品表(finances_product)、保险表(insurance)和
 --     基金表(fund)，列出客户的名称、身份证号以及投资总金额（即投资本金，
 --     每笔投资金额=商品数量*该产品每份金额)，注意投资金额按类型需要查询不同的表，
 --     投资总金额是客户购买的各类资产(理财,保险,基金)投资金额的总和，总金额命名为total_amount。
 --     查询结果按总金额降序排序。
select c_name, c_id_card, ifnull(sum(pro_amount), 0) as total_amount
from client 
    left join (
        select pro_c_id, pro_quantity * p_amount as pro_amount
        from property, finances_product
        where pro_pif_id = p_id
        and pro_type = 1
        union all
        select pro_c_id, pro_quantity * i_amount as pro_amount
        from property, insurance
        where pro_pif_id = i_id
        and pro_type = 2   
        union all
        select pro_c_id, pro_quantity * f_amount as pro_amount
        from property, fund
        where pro_pif_id = f_id
        and pro_type = 3
    ) as pro
    on pro.pro_c_id = c_id
group by c_id
order by total_amount desc;