import java.sql.*;
import java.util.Scanner;

public class ChangePass {
    static final String JDBC_DRIVER = "com.mysql.cj.jdbc.Driver";
    static final String DB_URL = "jdbc:mysql://127.0.0.1:3306/finance?allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=UTF8&useSSL=false&serverTimezone=UTC";
    static final String USER = "root";
    static final String PASS = "123123";
    /**
     * 修改客户密码
     *
     * @param connection 数据库连接对象
     * @param mail 客户邮箱,也是登录名
     * @param password 客户登录密码
     * @param newPass  新密码
     * @return
     *   1 - 密码修改成功
     *   2 - 用户不存在
     *   3 - 密码不正确
     *  -1 - 程序异常(如没能连接到数据库等）
     */
    public static int passwd(Connection connection,
                             String mail,
                             String password, 
                             String newPass){
        int code = -1;
        try {
            String sqlStatementSelect = "SELECT c_password FROM client WHERE c_mail = \""+mail+"\";";
            String sqlStatementUpdate = "UPDATE client SET c_password = \""+newPass+"\" WHERE c_mail = \""+mail+"\";";
            Statement statement = connection.createStatement();
            ResultSet resultSet = statement.executeQuery(sqlStatementSelect);
            if(resultSet.next()) {
                String oldPassword = resultSet.getString("c_password");
                if(oldPassword.equals(password)) {
                    int result = statement.executeUpdate(sqlStatementUpdate);
                    if(result == 1) {
                        code = 1;
                    }
                    else {
                        code = -1;
                    }
                }
                else {
                    code = 3;
                }
            }
            else {
                code = 2;
            }
        } catch(SQLException e){
            code = -1;
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
            String email = commands[0];
            String pass = commands[1];
            String pwd1 = commands[2];
            String pwd2 = commands[3];
            if (pwd1.equals(pwd2)) {
              int n = passwd(connection, email, pass, pwd1);  
              System.out.println("return: " + n);
            } else {
              System.out.println("两次输入的密码不一样!");
            }
        }
    }

}
