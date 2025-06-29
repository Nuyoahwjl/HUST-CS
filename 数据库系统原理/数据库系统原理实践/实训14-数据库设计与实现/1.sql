-- 请将你实现flight_booking数据库的语句写在下方：
CREATE DATABASE IF NOT EXISTS flight_booking;
USE flight_booking;

DROP TABLE IF EXISTS user;
CREATE TABLE user(
	user_id INT PRIMARY KEY AUTO_INCREMENT,
	firstname VARCHAR(50) NOT NULL,
	lastname VARCHAR(50) NOT NULL,
	dob DATE NOT NULL,
	sex CHAR(1) NOT NULL,
	email VARCHAR(50),
	phone VARCHAR(30),
	username VARCHAR(20) NOT NULL UNIQUE,
	password CHAR(32) NOT NULL,
	admin_tag TINYINT NOT NULL DEFAULT 0
);

DROP TABLE IF EXISTS passenger;
CREATE TABLE passenger(
	passenger_id INT PRIMARY KEY AUTO_INCREMENT,
	id CHAR(18) NOT NULL UNIQUE,
	firstname VARCHAR(50) NOT NULL,
	lastname VARCHAR(50) NOT NULL,
	mail VARCHAR(50),
	phone VARCHAR(20) NOT NULL,
	sex CHAR(1) NOT NULL,
	dob DATE
);

DROP TABLE IF EXISTS airport;
CREATE TABLE airport(
	airport_id INT PRIMARY KEY AUTO_INCREMENT,
	iata CHAR(3) NOT NULL UNIQUE,
	icao CHAR(4) NOT NULL UNIQUE,
	name VARCHAR(50) NOT NULL,
	city VARCHAR(50),
	country VARCHAR(50),
	latitude DECIMAL(11,8),
	longitude DECIMAL(11,8),
	index(name)
);

DROP TABLE IF EXISTS airline;
CREATE TABLE airline(
	airline_id INT PRIMARY KEY AUTO_INCREMENT,
	name VARCHAR(30) NOT NULL,
	iata CHAR(2) NOT NULL UNIQUE,
	airport_id INT NOT NULL,
	CONSTRAINT FK_airline_airport FOREIGN KEY (airport_id) REFERENCES airport(airport_id)
);

DROP TABLE IF EXISTS airplane;
CREATE TABLE airplane(
	airplane_id INT PRIMARY KEY AUTO_INCREMENT,
	type VARCHAR(50) NOT NULL,
	capacity SMALLINT NOT NULL,
	identifier VARCHAR(50) NOT NULL,
	airline_id INT NOT NULL,
	CONSTRAINT FK_airplane_airline FOREIGN KEY (airline_id) REFERENCES airline(airline_id)
);

DROP TABLE IF EXISTS flightschedule;
CREATE TABLE flightschedule(
	flight_no CHAR(8) PRIMARY KEY,
	departure TIME NOT NULL,
	arrival TIME NOT NULL,
	duration SMALLINT NOT NULL,
	monday TINYINT DEFAULT 0,
	tuesday TINYINT DEFAULT 0,
	wednesday TINYINT DEFAULT 0,
	thursday TINYINT DEFAULT 0,
	friday TINYINT DEFAULT 0,
	saturday TINYINT DEFAULT 0,
	sunday TINYINT DEFAULT 0,
	airline_id INT NOT NULL,
	`from` INT NOT NULL,
	`to` INT NOT NULL,
	CONSTRAINT FK_flightschedule_airline FOREIGN KEY (airline_id) REFERENCES airline(airline_id),
	CONSTRAINT FK_flightschedule_airport1 FOREIGN KEY (`from`) REFERENCES airport(airport_id),
	CONSTRAINT FK_flightschedule_airport2 FOREIGN KEY (`to`) REFERENCES airport(airport_id)
);

DROP TABLE IF EXISTS flight;
CREATE TABLE flight(
	flight_id INT PRIMARY KEY AUTO_INCREMENT,
	departure DATETIME NOT NULL,
	arrival DATETIME NOT NULL,
	duration SMALLINT NOT NULL,
	airline_id INT NOT NULL,
	airplane_id INT NOT NULL,
	flight_no CHAR(8) NOT NULL,
	`from` INT NOT NULL,
	`to` INT NOT NULL,
	CONSTRAINT FK_flight_airline FOREIGN KEY (airline_id) REFERENCES airline(airline_id),
	CONSTRAINT FK_flight_airplane FOREIGN KEY (airplane_id) REFERENCES airplane(airplane_id),
	CONSTRAINT FK_flight_flightschedule FOREIGN KEY (flight_no) REFERENCES flightschedule(flight_no),
	CONSTRAINT FK_flight_airport1 FOREIGN KEY (`from`) REFERENCES airport(airport_id),
	CONSTRAINT FK_flight_airport2 FOREIGN KEY (`to`) REFERENCES airport(airport_id)
);

DROP TABLE IF EXISTS ticket;
CREATE TABLE ticket(
	ticket_id INT PRIMARY KEY AUTO_INCREMENT,
	seat CHAR(4),
	price DECIMAL(10,2) NOT NULL,
	flight_id INT NOT NULL,
	passenger_id INT NOT NULL,
	user_id INT NOT NULL,
	CONSTRAINT FK_ticket_flight FOREIGN KEY (flight_id) REFERENCES flight(flight_id),
	CONSTRAINT FK_ticket_passenger FOREIGN KEY (passenger_id) REFERENCES passenger(passenger_id),
	CONSTRAINT FK_ticket_user FOREIGN KEY (user_id) REFERENCES user(user_id)
);
