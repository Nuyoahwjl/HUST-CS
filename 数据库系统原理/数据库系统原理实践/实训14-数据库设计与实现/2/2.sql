-- 创建数据库
CREATE DATABASE IF NOT EXISTS cinema_db;
USE cinema_db;

-- 创建电影表
CREATE TABLE movie (
    movie_ID INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    type VARCHAR(50),
    runtime INT,  -- 单位：分钟
    release_date DATE,
    director VARCHAR(100),
    starring VARCHAR(255)
) ENGINE=InnoDB;

-- 创建客户表
CREATE TABLE customer (
    c_ID INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    phone VARCHAR(20) UNIQUE
) ENGINE=InnoDB;

-- 创建影厅表
CREATE TABLE hall (
    hall_ID INT AUTO_INCREMENT PRIMARY KEY,
    mode VARCHAR(50) NOT NULL,  -- 如IMAX/3D/普通厅
    capacity INT NOT NULL,
    location VARCHAR(200) NOT NULL
) ENGINE=InnoDB;

-- 创建排片表
CREATE TABLE schedule (
    schedule_ID INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    time TIME NOT NULL,
    price DECIMAL(6,2) NOT NULL,
    number INT DEFAULT 0,  -- 已售出票数
    movie_ID INT NOT NULL,
    hall_ID INT NOT NULL,
    FOREIGN KEY (movie_ID) REFERENCES movie(movie_ID) ON DELETE CASCADE,
    FOREIGN KEY (hall_ID) REFERENCES hall(hall_ID) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 创建票务表
CREATE TABLE ticket (
    ticket_ID INT AUTO_INCREMENT PRIMARY KEY,
    seat_num VARCHAR(10) NOT NULL,  -- 如'A01'
    c_ID INT,
    schedule_ID INT NOT NULL,
    FOREIGN KEY (c_ID) REFERENCES customer(c_ID) ON DELETE SET NULL,
    FOREIGN KEY (schedule_ID) REFERENCES schedule(schedule_ID) ON DELETE CASCADE
) ENGINE=InnoDB;