-- ------------------------------------------------------------
-- 🚀 docker-compose up 시 자동으로 실행될 초기 SQL 목록
-- ------------------------------------------------------------
ALTER USER IF EXISTS 'root'@'%' IDENTIFIED WITH mysql_native_password BY '1234';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;