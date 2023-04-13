const mysql = require('mysql')
//连接数据库
const content = mysql.createConnection({
    host: 'localhost', //数据域名
    user:'root', //数据库软件账号
    password:'', //数据库软件密码
    database:'', //数据库名称
    port:'3306',//端口号，默认3306
})
//编写sql语句
const sql = "select * from userinfo"
//执行sql语句
content.query(sql, function (error, success){
    if(error){
        console.log("fail");
        return
    }
    console.log(success);
})