-- MySQL schema and CRUD operations for PG MINI PROJECT

CREATE DATABASE IF NOT EXISTS mini_project;
USE mini_project;

CREATE TABLE IF NOT EXISTS action_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    action_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_action VARCHAR(255) NOT NULL,
    customer_id VARCHAR(100),
    customer_name VARCHAR(255),
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    action_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_action VARCHAR(255) NOT NULL,
    customer_id VARCHAR(100),
    customer_name VARCHAR(255),
    features TEXT,
    prediction VARCHAR(255) NOT NULL,
    probability DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- action_log CRUD
INSERT INTO action_log (action_time, user_action, customer_id, customer_name, details)
VALUES (NOW(), 'start_prediction', 'C123', 'Jane Doe', 'User started prediction flow');

SELECT id, action_time, user_action, customer_id, customer_name, details, created_at
FROM action_log;

SELECT id, action_time, user_action, customer_id, customer_name, details, created_at
FROM action_log
WHERE id = 123;

UPDATE action_log
SET user_action = 'update_prediction',
    customer_id = 'C123',
    customer_name = 'Jane Doe',
    details = 'Changed prediction settings'
WHERE id = 123;

DELETE FROM action_log
WHERE id = 123;

-- prediction_log CRUD
INSERT INTO prediction_log (action_time, user_action, customer_id, customer_name, features, prediction, probability)
VALUES (NOW(), 'predict_cancer', 'C123', 'Jane Doe', 'age=60,glucose=120,...', 'positive', 0.87);

SELECT id, action_time, user_action, customer_id, customer_name, features, prediction, probability, created_at
FROM prediction_log;

SELECT id, action_time, user_action, customer_id, customer_name, features, prediction, probability, created_at
FROM prediction_log
WHERE id = 456;

UPDATE prediction_log
SET customer_id = 'C123',
    customer_name = 'Jane Doe',
    prediction = 'negative',
    probability = 0.12,
    features = 'age=60,glucose=110,...'
WHERE id = 456;

DELETE FROM prediction_log
WHERE id = 456;
