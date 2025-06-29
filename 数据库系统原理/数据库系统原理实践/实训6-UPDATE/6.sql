use finance1;
-- 在金融应用场景数据库中，已在表property(资产表)中添加了客户身份证列，列名为pro_id_card，类型为char(18)，该列目前全部留空(null)。

-- 请用一条update语句，根据client表中提供的身份证号(c_id_card)，填写property表中对应的身份证号信息(pro_id_card)。
update property,client 
set pro_id_card=c_id_card 
where pro_c_id = c_id;
