### 表1 `client`（客户表）

| 字段名称      | 数据类型      | 约束             | 说明          |
|---------------|---------------|------------------|---------------|
| `c_id`        | INTEGER       | PRIMARY KEY      | 客户编号      |
| `c_name`      | VARCHAR(100)  | NOT NULL         | 客户名称      |
| `c_mail`      | CHAR(30)      | UNIQUE           | 客户邮箱      |
| `c_id_card`   | CHAR(20)      | UNIQUE NOT NULL  | 客户身份证    |
| `c_phone`     | CHAR(20)      | UNIQUE NOT NULL  | 客户手机号    |
| `c_password`  | CHAR(20)      | NOT NULL         | 客户登录密码  |

---

### 表2 `bank_card`（银行卡）

| 字段名称      | 数据类型      | 约束                 | 说明                                     |
|---------------|---------------|----------------------|------------------------------------------|
| `b_number`    | CHAR(30)      | PRIMARY KEY          | 银行卡号                                 |
| `b_type`      | CHAR(20)      | 无                   | 银行卡类型（储蓄卡/信用卡）              |
| `b_c_id`      | INTEGER       | NOT NULL FOREIGN KEY | 所属客户编号，引用自 `client` 表的 `c_id`|
| `b_balance`   | NUMERIC(10,2) | NOT NULL             | 余额（信用卡余额系指已透支的金额）       |

---

### 表3 `finances_product`（理财产品表）

| 字段名称          | 数据类型          | 约束          | 说明         |
|-------------------|-------------------|---------------|--------------|
| `p_name`          | VARCHAR(100)      | NOT NULL      | 产品名称     |
| `p_id`            | INTEGER           | PRIMARY KEY   | 产品编号     |
| `p_description`   | VARCHAR(4000)     | 无            | 产品描述     |
| `p_amount`        | INTEGER           | 无            | 购买金额     |
| `p_year`          | INTEGER           | 无            | 理财年限     |

---

### 表4 `insurance`（保险表）

| 字段名称          | 数据类型          | 约束          | 说明         |
|-------------------|-------------------|---------------|--------------|
| `i_name`          | VARCHAR(100)      | NOT NULL      | 保险名称     |
| `i_id`            | INTEGER           | PRIMARY KEY   | 保险编号     |
| `i_amount`        | INTEGER           | 无            | 保险金额     |
| `i_person`        | CHAR(20)          | 无            | 适用人群     |
| `i_year`          | INTEGER           | 无            | 保险年限     |
| `i_project`       | VARCHAR(200)      | 无            | 保障项目     |

---

### 表5 `fund`（基金表）

| 字段名称          | 数据类型          | 约束                | 说明               |
|-------------------|-------------------|---------------------|--------------------|
| `f_name`          | VARCHAR(100)      | NOT NULL            | 基金名称           |
| `f_id`            | INTEGER           | PRIMARY KEY         | 基金编号           |
| `f_type`          | CHAR(20)          | 无                  | 基金类型           |
| `f_amount`        | INTEGER           | 无                  | 基金金额           |
| `risk_level`      | CHAR(20)          | NOT NULL            | 风险等级           |
| `f_manager`       | INTEGER           | NOT NULL            | 基金管理者         |

---

### 表6 `property`（资产表）

| 字段名称           | 数据类型   | 约束                  | 说明                                       |
|--------------------|------------|-----------------------|--------------------------------------------|
| `pro_id`           | INTEGER    | PRIMARY KEY           | 资产编号                                   |
| `pro_c_id`         | INTEGER    | NOT NULL FOREIGN KEY  | 所属客户编号（引用自 `client` 表的 `c_id`）|
| `pro_pif_id`       | INTEGER    | NOT NULL              | 业务约束（关联具体产品编号）               |
| `pro_type`         | INTEGER    | NOT NULL              | 商品类型：`1`=理财产品；`2`=保险；`3`=基金 |
| `pro_status`       | CHAR(20)   | 无                    | 商品状态（“可用”、“冻结”）                 |
| `pro_quantity`     | INTEGER    | 无                    | 商品数量                                   |
| `pro_income`       | INTEGER    | 无                    | 商品收益                                   |
| `pro_purchase_time`| DATE       | 无                    | 购买时间                                   |

---
