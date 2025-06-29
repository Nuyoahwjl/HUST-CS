use finance1;

-- 在金融应用场景数据库中，编程实现一个转账操作的存储过程sp_transfer_balance，实现从一个帐户向另一个帐户转账。
-- 请补充代码完成该过程：
create procedure sp_transfer(
                     in applicant_id int,      
                     in source_card_id char(30),
                     in receiver_id int, 
                     in dest_card_id char(30),
                     in amount numeric(10,2),
                     out return_code int)
begin

set autocommit = off;
start transaction;
    update bank_card set b_balance = b_balance-amount where b_number = source_card_id and b_c_id = applicant_id and b_type = "储蓄卡";
    update bank_card set b_balance = b_balance+amount where b_number = dest_card_id and b_c_id = receiver_id and b_type = "储蓄卡";
    update bank_card set b_balance = b_balance-amount where b_number = dest_card_id and b_c_id = receiver_id and b_type = "信用卡";

    if not exists(select * from bank_card where b_number = source_card_id and b_c_id = applicant_id and b_type = "储蓄卡" and b_balance >= 0) then
        set return_code = 0;
        rollback;
    elseif not exists(select * from bank_card where b_number = dest_card_id and b_c_id = receiver_id) then
        set return_code = 0;
        rollback;
    else
        set return_code = 1;
        commit;
    end if;
set autocommit = true;

end$$

delimiter ;
