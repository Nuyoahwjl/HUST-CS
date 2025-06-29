### 表1 client（客户表）

| 字段名称     | 数据类型       | 约束                   | 说明             |
|--------------|----------------|------------------------|------------------|
| c_id         | INTEGER        | PRIMARY KEY            | 客户编号         |
| c_name       | VARCHAR(100)   | NOT NULL               | 客户名称         |
| c_mail       | CHAR(30)       | UNIQUE                 | 客户邮箱         |
| c_id_card    | CHAR(20)       | UNIQUE NOT NULL        | 客户身份证       |
| c_phone      | CHAR(20)       | UNIQUE NOT NULL        | 客户手机号       |
| c_password   | CHAR(20)       | NOT NULL               | 客户登录密码     |

---

### 表2 wage（薪资表）

| 字段名称   | 数据类型        | 约束                   | 说明                       |
|------------|-----------------|------------------------|----------------------------|
| w_id       | INTEGER         | PRIMARY KEY            | 薪资发放记录编号           |
| w_c_id     | INTEGER         | NOT NULL, FOREIGN KEY  | 客户编号（关联 client 表） |
| w_amount   | NUMERIC(10,2)   | 无                     | 酬劳                       |
| w_org      | CHAR(30)        | NOT NULL               | 酬劳发放单位               |
| w_time     | DATE            | 无                     | 酬劳发放时间               |
| w_type     | INTEGER         | NOT NULL               | 酬劳类型：全职(1)、兼职(2) |
| w_tax      | CHAR            | NOT NULL               | 缴纳税标志：Y，N           |
| w_memo     | CHAR(30)        | 无                     | 酬劳说明                   |

---

### 表3 new_wage（薪资表）

| 字段名称     | 数据类型        | 约束                   | 说明                 |
|--------------|-----------------|------------------------|----------------------|
| id           | INTEGER         | PRIMARY KEY            | 序号                 |
| c_id_card    | CHAR(20)        | UNIQUE NOT NULL        | 身份证               |
| w_amount     | NUMERIC(10,2)   | 无                     | 酬劳金额             |
| w_org        | CHAR(30)        | NOT NULL               | 发放单位             |
| w_time       | DATE            | 无                     | 支付酬劳时间         |
| w_type       | INTEGER         | NOT NULL               | 酬劳类型：全职、兼职 |
| w_memo       | CHAR(30)        | 无                     | 酬劳发放说明         |