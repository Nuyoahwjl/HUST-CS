import java.sql.*;
import java.util.Scanner;

public class Transfer {
    static final String JDBC_DRIVER = "com.mysql.cj.jdbc.Driver";
    static final String DB_URL = "jdbc:mysql://127.0.0.1:3306/finance?allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=UTF8&useSSL=false&serverTimezone=UTC";
    static final String USER = "root";
    static final String PASS = "123123";
    /**
     * 转账操作
     *
     * @param connection 数据库连接对象
     * @param sourceCard 转出账号
     * @param destCard 转入账号
     * @param amount  转账金额
     * @return boolean
     *   true  - 转账成功
     *   false - 转账失败
     */
    public static boolean transferBalance(Connection connection,
                             String sourceCard,
                             String destCard, 
                             double amount){
        boolean code = false;
        try {
            connection.setAutoCommit(false);
            connection.setTransactionIsolation(4);
        } catch(SQLException e) {
            e.printStackTrace();
        }
        try {
            String sqlFindCard = "SELECT * FROM bank_card WHERE b_number = ?;";
            String sqlUpdateAmount = "UPDATE bank_card SET b_balance = b_balance - ? WHERE b_number = ? AND b_type = ?;";
            //转出
            PreparedStatement statement = connection.prepareStatement(sqlUpdateAmount);
            statement.setDouble(1, amount);
            statement.setString(2, sourceCard);
            statement.setString(3, "储蓄卡");
            int result = statement.executeUpdate();
            //转入
            statement = connection.prepareStatement(sqlUpdateAmount);
            statement.setDouble(1, -amount);
            statement.setString(2, destCard);
            statement.setString(3, "储蓄卡");
            result = statement.executeUpdate();
            statement = connection.prepareStatement(sqlUpdateAmount);
            statement.setDouble(1, amount);
            statement.setString(2, destCard);
            statement.setString(3, "信用卡");
            result = statement.executeUpdate();
            //检查账户是否正确
            statement = connection.prepareStatement(sqlFindCard);
            statement.setString(1, sourceCard);
            ResultSet resultSet = statement.executeQuery();
            if(resultSet.next()) {
                if(resultSet.getString("b_type").equals("储蓄卡") && resultSet.getDouble("b_balance") >= 0.) {
                    statement = connection.prepareStatement(sqlFindCard);
                    statement.setString(1, destCard);
                    resultSet = statement.executeQuery();
                    if(resultSet.next()) {
                        code = true;
                        connection.commit();
                    } 
                    else {
                        code = false;
                        connection.rollback();
                    }
                }
                else {
                    code = false;
                    connection.rollback();
                }
            }
            else {
                code = false;
                connection.rollback();
            }
        } catch(SQLException e) {
            connection.rollback();
            e.printStackTrace();
        } finally {
            return code;
        }

    }

    // 不要修改main() 
    public static void main(String[] args) throws Exception {

        Scanner sc = new Scanner(System.in);
        Class.forName(JDBC_DRIVER);

        Connection connection = DriverManager.getConnection(DB_URL, USER, PASS);

        while(sc.hasNext())
        {
            String input = sc.nextLine();
            if(input.equals(""))
                break;

            String[]commands = input.split(" ");
            if(commands.length ==0)
                break;
            String payerCard = commands[0];
            String  payeeCard = commands[1];
            double  amount = Double.parseDouble(commands[2]);
            if (transferBalance(connection, payerCard, payeeCard, amount)) {
              System.out.println("转账成功。" );
            } else {
              System.out.println("转账失败,请核对卡号，卡类型及卡余额!");
            }
        }
    }

}
