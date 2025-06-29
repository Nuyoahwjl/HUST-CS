import java.sql.*;

public class Transform {
    static final String JDBC_DRIVER = "com.mysql.cj.jdbc.Driver";
    static final String DB_URL = "jdbc:mysql://127.0.0.1:3306/sparsedb?allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=UTF8&useSSL=false&serverTimezone=UTC";
    static final String USER = "root";
    static final String PASS = "123123";

    /**
     * 向sc表中插入数据
     *
     */
    public static int insertSC(Connection connection, int sno,
                               String col_name, int col_value){
        int result = 0;
        try {
            Statement statement = connection.createStatement();
            String sqlStatement = "INSERT INTO sc VALUES("+sno+",\""+col_name+"\","+col_value+");";
            result = statement.executeUpdate(sqlStatement);
        } catch(SQLException e) {
            e.printStackTrace();
        } finally {
            return result;
        }
    }

    public static void main(String[] args) {
        try {
            Class.forName(JDBC_DRIVER);
            Connection connection = DriverManager.getConnection(DB_URL, USER, PASS);
            Statement statement = connection.createStatement();
            ResultSet resultSetAll = statement.executeQuery("SELECT * FROM entrance_exam;");
            while(resultSetAll.next()) {
                int sno = resultSetAll.getInt("sno");
                int chinese = resultSetAll.getInt("chinese");
                if(chinese != 0){
                    int ret = insertSC(connection, sno, "chinese", chinese);
                }
                int math = resultSetAll.getInt("math");
                if(math != 0){
                    int ret = insertSC(connection, sno, "math", math);
                }
                int english = resultSetAll.getInt("english");
                if(english != 0){
                    int ret = insertSC(connection, sno, "english", english);
                }
                int physics = resultSetAll.getInt("physics");
                if(physics != 0){
                    int ret = insertSC(connection, sno, "physics", physics);
                }
                int chemistry = resultSetAll.getInt("chemistry");
                if(chemistry != 0){
                    int ret = insertSC(connection, sno, "chemistry", chemistry);
                }
                int biology = resultSetAll.getInt("biology");
                if(biology != 0){
                    int ret = insertSC(connection, sno, "biology", biology);
                }
                int history = resultSetAll.getInt("history");
                if(history != 0){
                    int ret = insertSC(connection, sno, "history", history);
                }
                int geography = resultSetAll.getInt("geography");
                if(geography != 0){
                    int ret = insertSC(connection, sno, "geography", geography);
                }
                int politics = resultSetAll.getInt("politics");
                if(politics != 0){
                    int ret = insertSC(connection, sno, "politics", politics);
                }
            } 
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
    
}